"""Tests for the server-rendered web UI surface (optional ``[ui]`` extra).

Offline and deterministic via the ``fake`` provider, exercised headlessly with FastAPI's
``TestClient`` — the same pattern as the HTTP API tests. Skipped when the UI/engine extras
aren't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")
pytest.importorskip("fastapi")
pytest.importorskip("jinja2")
pytest.importorskip("pygments")

from fastapi.testclient import TestClient  # noqa: E402

from coderag.api import CodeRAG  # noqa: E402
from coderag.config import Config  # noqa: E402
from coderag.surfaces.webui import create_ui_app  # noqa: E402
from tests.conftest import write  # noqa: E402


@pytest.fixture
def ui(tmp_path):
    repo = tmp_path / "repo"
    store = tmp_path / "store"
    write(repo / "auth.py", "def authenticate(token):\n    return token == 'ok'\n")
    write(repo / "util.py", "class Helper:\n    def run(self):\n        return 42\n")
    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store))
    cr.index()
    return cr, TestClient(create_ui_app(cr))


def test_home_empty_state_lists_examples(ui):
    _, client = ui
    r = client.get("/")
    assert r.status_code == 200
    assert "CodeRAG" in r.text
    assert "<form" in r.text  # search form present
    assert "?q=" in r.text  # clickable example queries


def test_search_renders_highlighted_hits(ui):
    _, client = ui
    r = client.get("/", params={"q": "authenticate"})
    assert r.status_code == 200
    assert "auth.py" in r.text
    assert 'class="highlight"' in r.text  # Pygments output present
    assert "/file?path=" in r.text  # citation links into the file viewer
    assert "data-search-ms=" not in r.text  # speed badge is demo-only


def test_filters_narrow_results(ui):
    _, client = ui
    # A path filter that matches nothing should yield the no-results message.
    r = client.get("/", params={"q": "authenticate", "path": "does-not-exist"})
    assert r.status_code == 200
    assert "No results" in r.text
    # Filtering to the matching path keeps the hit.
    r = client.get("/", params={"q": "authenticate", "path": "auth.py"})
    assert "auth.py" in r.text


def test_file_view_and_traversal_guard(ui):
    _, client = ui
    r = client.get("/file", params={"path": "auth.py"})
    assert r.status_code == 200
    assert "authenticate" in r.text
    # Path traversal / non-indexed files are rejected by the facade.
    assert client.get("/file", params={"path": "../../etc/passwd"}).status_code == 404


def test_browse_lists_indexed_files(ui):
    _, client = ui
    r = client.get("/browse")
    assert r.status_code == 200
    assert "auth.py" in r.text and "util.py" in r.text
    # The path filter narrows the listing.
    r = client.get("/browse", params={"path": "util"})
    assert "util.py" in r.text and "auth.py" not in r.text


def test_status_page_and_healthz(ui):
    _, client = ui
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/status")
    assert r.status_code == 200
    assert "total_chunks" in r.text  # status() keys rendered


def test_reindex_redirects_to_status(ui):
    _, client = ui
    r = client.post("/reindex", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/status")


def test_answer_is_graceful_without_a_backend(ui):
    _, client = ui
    # No OPENAI/ANTHROPIC creds in the fake config → a friendly message, not a 500.
    r = client.get("/answer", params={"q": "authenticate"})
    assert r.status_code == 200
    assert "unavailable" in r.text.lower()


def test_pygments_stylesheet_served(ui):
    _, client = ui
    r = client.get("/assets/pygments.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]
    assert ".highlight" in r.text


def test_social_share_meta_tags(ui):
    _, client = ui
    r = client.get("/")
    assert r.status_code == 200
    # Absolute, opaque 1200x630 card pointed at by both Open Graph and Twitter, with
    # the dimensions/alt that let scrapers render the preview without a re-fetch.
    assert '<meta property="og:image" content="https://' in r.text
    assert "/static/og-image.png" in r.text
    assert '<meta property="og:image:width" content="1200">' in r.text
    assert '<meta property="og:image:height" content="630">' in r.text
    assert '<meta property="og:image:type" content="image/png">' in r.text
    assert 'name="twitter:card" content="summary_large_image"' in r.text
    assert 'name="twitter:image:alt"' in r.text
    # The referenced card is actually served by the static mount.
    img = client.get("/static/og-image.png")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"


def test_store_distinct_helpers(ui):
    cr, _ = ui
    assert "python" in cr.store.distinct_languages()
    kinds = cr.store.distinct_kinds()
    assert any(k in kinds for k in ("function", "class", "method"))


# --- demo mode ---


def _demo_client(tmp_path, *, max_answers=2, cooldown=0):
    repo = tmp_path / "repo"
    store = tmp_path / "store"
    write(repo / "auth.py", "def authenticate(token):\n    return token == 'ok'\n")
    cr = CodeRAG(
        Config(
            provider="fake",
            watched_dir=repo,
            store_dir=store,
            demo_mode=True,
            demo_max_answers=max_answers,
            demo_cooldown_seconds=cooldown,
        )
    )
    cr.index()
    return cr, TestClient(create_ui_app(cr))


def test_demo_mode_banner_caps_and_hidden_reindex(tmp_path):
    cr, client = _demo_client(tmp_path)
    r = client.get("/", params={"q": "authenticate"})
    assert "Demo mode" in r.text
    assert 'action="/reindex"' not in r.text  # reindex hidden in the public demo
    assert 'max="8"' in r.text  # results capped (k control on the search view) at 8
    # POST /reindex is a no-op in demo mode: it redirects and does not index.
    n0 = cr.store.total_chunks()
    rr = client.post("/reindex", follow_redirects=False)
    assert rr.status_code == 303 and "disabled" in rr.headers["location"].lower()
    assert cr.store.total_chunks() == n0


def test_demo_mode_shows_search_speed_badge(tmp_path):
    cr, client = _demo_client(tmp_path)
    r = client.get("/", params={"q": "authenticate"})
    # Demo mode surfaces retrieval speed and frames it as separate from AI answers.
    assert "data-search-ms=" in r.text
    assert "instant local" in r.text.lower()  # reworded demo banner
    # The empty landing (no query, no results) shows no badge.
    assert "data-search-ms=" not in client.get("/").text


def test_demo_answer_quota_is_enforced(tmp_path):
    # No LLM backend → each allowed answer streams an "unavailable" notice, but it
    # still charges the soft per-session quota (the gate charges on attempt).
    cr, client = _demo_client(tmp_path, max_answers=2, cooldown=0)
    assert "unavailable" in client.get("/answer", params={"q": "q1"}).text.lower()
    assert "unavailable" in client.get("/answer", params={"q": "q2"}).text.lower()
    assert "limit reached" in client.get("/answer", params={"q": "q3"}).text.lower()


def test_demo_answer_cooldown_blocks_rapid_followups(tmp_path):
    cr, client = _demo_client(tmp_path, max_answers=5, cooldown=30)
    client.get("/answer", params={"q": "a"})  # charges, starts the cooldown
    assert "wait" in client.get("/answer", params={"q": "b"}).text.lower()


def test_demo_cookie_is_minted_not_reflected(tmp_path):
    # First visit (no cookie) → the server mints and sets one.
    cr, client = _demo_client(tmp_path)
    client.cookies.clear()
    r = client.get("/")
    assert "coderag_demo=" in r.headers.get("set-cookie", "")
    # A request that already carries a (crafted) cookie must NOT have that value
    # echoed back via Set-Cookie — the server never reflects client input.
    client.cookies.clear()
    r2 = client.get("/", cookies={"coderag_demo": "attacker-controlled"})
    assert "set-cookie" not in {k.lower() for k in r2.headers}
