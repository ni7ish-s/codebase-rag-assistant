"""P6 tests: CLI, HTTP API, and watcher behaviour (all with the fake provider)."""

from __future__ import annotations

import json

import pytest

from coderag.api import CodeRAG
from coderag.surfaces.cli import main as cli_main
from tests.conftest import write


@pytest.fixture
def repo_with_code(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    store = tmp_path / "store"
    write(repo / "auth.py", "def authenticate(token):\n    return token == 'ok'\n")
    monkeypatch.setenv("CODERAG_PROVIDER", "fake")
    common = ["--watched-dir", str(repo), "--store-dir", str(store)]
    return repo, store, common


# --- CLI ---


def test_cli_index_then_search(repo_with_code, capsys):
    repo, store, common = repo_with_code
    assert cli_main(["index", "--quiet", *common]) == 0
    assert "Indexed" in capsys.readouterr().out

    assert cli_main(["search", "authenticate", "-k", "3", *common]) == 0
    out = capsys.readouterr().out
    assert "auth.py:1" in out


def test_cli_search_json(repo_with_code, capsys):
    repo, store, common = repo_with_code
    cli_main(["index", "--quiet", *common])
    capsys.readouterr()
    rc = cli_main(["search", "authenticate", "--json", *common])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload[0]["path"] == "auth.py"


def test_cli_status(repo_with_code, capsys):
    repo, store, common = repo_with_code
    cli_main(["index", "--quiet", *common])
    capsys.readouterr()
    cli_main(["status", *common])
    status = json.loads(capsys.readouterr().out)
    assert status["provider"] == "fake"
    assert status["total_files"] == 1


def test_cli_search_without_index(repo_with_code, capsys):
    repo, store, common = repo_with_code
    rc = cli_main(["search", "anything", *common])
    assert rc == 1
    assert "No results" in capsys.readouterr().out


def test_cli_eval_graph(repo_with_code, tmp_path, capsys):
    repo, store, common = repo_with_code
    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        json.dumps(
            {
                "query": "authenticate",
                "relevant_files": ["auth.py"],
                "relevant_symbols": ["authenticate"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # --compare path: forces graph on and adds a hybrid+graph row.
    rc = cli_main(
        ["eval", "--dataset", str(ds), "--compare", "--graph", "--quiet", *common]
    )
    assert rc == 0
    assert "hybrid+graph" in capsys.readouterr().out
    # single-mode path: the label reflects the enabled stage.
    rc = cli_main(["eval", "--dataset", str(ds), "--graph", "--quiet", *common])
    assert rc == 0
    assert "graph" in capsys.readouterr().out


# --- HTTP API ---


def test_http_api_search_and_status(repo_with_code):
    from fastapi.testclient import TestClient

    from coderag.surfaces.http_api import create_app

    repo, store, _ = repo_with_code
    from coderag.config import Config

    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store))
    cr.index()
    client = TestClient(create_app(cr))

    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["total_files"] == 1

    r = client.get("/search", params={"q": "authenticate", "k": 3})
    body = r.json()
    assert body["count"] >= 1
    assert body["results"][0]["path"] == "auth.py"

    r = client.get("/file", params={"path": "auth.py"})
    assert "authenticate" in r.json()["content"]

    r = client.get("/file", params={"path": "../../etc/passwd"})
    assert r.status_code == 404  # path traversal blocked


def test_http_index_endpoint(repo_with_code):
    from fastapi.testclient import TestClient

    from coderag.config import Config
    from coderag.surfaces.http_api import create_app

    repo, store, _ = repo_with_code
    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store))
    client = TestClient(create_app(cr))
    r = client.post("/index", json={"full": False})
    assert r.status_code == 200
    assert r.json()["total_files"] == 1


# --- watcher ---


def test_watcher_apply_handles_edit_and_delete(repo_with_code):
    from coderag.config import Config
    from coderag.watch import _apply

    repo, store, _ = repo_with_code
    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store))
    cr.index()
    n0 = cr.store.total_chunks()

    new = repo / "extra.py"
    write(new, "def extra():\n    return 1\n")
    _apply(cr, str(new))
    assert cr.store.total_chunks() > n0

    new.unlink()
    _apply(cr, str(new))
    assert "extra.py" not in cr.store.all_file_paths()
    assert cr.store.total_chunks() == n0  # back to the pre-edit count


def test_watcher_handler_collects_only_code_paths():
    import threading

    from watchdog.events import (
        DirModifiedEvent,
        FileCreatedEvent,
        FileModifiedEvent,
        FileMovedEvent,
    )

    from coderag.watch import _Handler

    pending: set = set()
    handler = _Handler(pending, threading.Lock())

    handler.on_modified(FileModifiedEvent("/repo/auth.py"))  # code file -> noted
    handler.on_created(FileCreatedEvent("/repo/data.bin"))  # unknown ext -> ignored
    handler.on_modified(DirModifiedEvent("/repo/pkg"))  # directory -> ignored
    handler.on_moved(FileMovedEvent("/repo/old.py", "/repo/new.py"))  # both ends noted

    assert pending == {"/repo/auth.py", "/repo/old.py", "/repo/new.py"}


# --- HTTP API security ---


def test_http_api_requires_key_when_configured(repo_with_code):
    from fastapi.testclient import TestClient

    from coderag.config import Config
    from coderag.surfaces.http_api import create_app

    repo, store, _ = repo_with_code
    cr = CodeRAG(
        Config(provider="fake", watched_dir=repo, store_dir=store, api_key="s3cret")
    )
    cr.index()
    client = TestClient(create_app(cr))

    # Protected endpoints require the key when one is configured.
    assert client.get("/search", params={"q": "x"}).status_code == 401  # no key
    assert (
        client.get(
            "/search", params={"q": "x"}, headers={"X-API-Key": "nope"}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/search", params={"q": "x"}, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/search", params={"q": "x"}, headers={"Authorization": "Bearer s3cret"}
        ).status_code
        == 200
    )

    # /status is the health-probe target and is intentionally exempt from auth, so
    # a server with an API key set can still pass its own liveness/readiness checks.
    assert client.get("/status").status_code == 200
    assert client.get("/status", headers={"X-API-Key": "nope"}).status_code == 200


def test_file_endpoint_serves_only_indexed_files(repo_with_code):
    from fastapi.testclient import TestClient

    from coderag.config import Config
    from coderag.surfaces.http_api import create_app

    repo, store, _ = repo_with_code
    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store))
    cr.index()
    # A non-code file under the root is never indexed, so it must not be readable.
    write(repo / "secret.env", "TOKEN=supersecret\n")
    client = TestClient(create_app(cr))

    assert client.get("/file", params={"path": "secret.env"}).status_code == 404
    r = client.get("/file", params={"path": "auth.py"})
    assert r.status_code == 200 and "authenticate" in r.json()["content"]


def test_cors_is_disabled_by_default(repo_with_code):
    from fastapi.testclient import TestClient

    from coderag.config import Config
    from coderag.surfaces.http_api import create_app

    repo, store, _ = repo_with_code
    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store))
    cr.index()
    client = TestClient(create_app(cr))
    r = client.get("/status", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
