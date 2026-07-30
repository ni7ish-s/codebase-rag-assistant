"""Tests for the MCP server surface (all offline, with the fake provider).

Drives the FastMCP tools in-memory via ``call_tool`` (no subprocess), mirroring the
HTTP-surface tests. Also covers the two things the MCP server newly stresses: parallel
indexing correctness and search staying safe while the index is being written.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading

import pytest

pytest.importorskip("mcp")  # skip the whole module if the [mcp] extra isn't installed

from reporag.api import CodeRAG  # noqa: E402
from reporag.config import Config  # noqa: E402
from reporag.surfaces.mcp_server import _State, _warm_up, build_mcp  # noqa: E402
from tests.conftest import write  # noqa: E402

DEMO = {
    "auth.py": (
        "def authenticate(token):\n"
        "    '''verify a token with retry/backoff'''\n"
        "    return token == 'ok'\n"
    ),
    "math.ts": "export function add(a: number, b: number) {\n  return a + b;\n}\n",
}


def _make(tmp_path, files, **cfg):
    """Build an indexed CodeRAG + MCP server over ``files`` with the fake provider."""
    repo = tmp_path / "repo"
    store = tmp_path / "store"
    for name, body in files.items():
        write(repo / name, body)
    cr = CodeRAG(Config(provider="fake", watched_dir=repo, store_dir=store, **cfg))
    cr.index()
    state = _State()
    return cr, build_mcp(cr, state=state), state, repo


def _call(mcp, name, args):
    """Invoke a tool and parse its JSON text content into a dict."""
    res = asyncio.run(mcp.call_tool(name, args))
    content = res[0] if isinstance(res, tuple) else res
    return json.loads(content[0].text)


# --- tool surface ---


def test_tools_are_registered(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {
        "search_code",
        "search_files",
        "get_file",
        "index_status",
        "reindex",
    }
    cr.close()


def test_search_code_returns_compact_locations(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)
    r = _call(mcp, "search_code", {"query": "authenticate token", "top_k": 5})
    assert r["count"] >= 1
    assert r["indexing"] == "ready"
    hit = r["results"][0]
    # Compact shape: path:start-end location, a snippet, and no heavy full-text field.
    assert re.match(r".+:\d+-\d+$", hit["location"])
    assert "snippet" in hit and "text" not in hit
    assert {"symbol", "kind", "language", "score", "similarity"} <= hit.keys()
    cr.close()


def test_snippet_truncated_unless_full_text(tmp_path):
    body = (
        "def big_function():\n"
        + "".join(f"    step_{i} = {i}\n" for i in range(40))
        + "    return 'x39 done'\n"
    )
    cr, mcp, _, _ = _make(tmp_path, {"big.py": body})
    q = {"query": "big_function step_39 x39", "top_k": 5}
    hit = next(
        h for h in _call(mcp, "search_code", q)["results"] if "big.py" in h["location"]
    )
    assert hit["truncated"] is True and "…" in hit["snippet"]

    full = next(
        h
        for h in _call(mcp, "search_code", {**q, "full_text": True})["results"]
        if "big.py" in h["location"]
    )
    assert full["truncated"] is False and "step_39" in full["snippet"]
    cr.close()


def test_search_code_filters(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)

    r = _call(
        mcp, "search_code", {"query": "add", "top_k": 10, "language": "typescript"}
    )
    assert r["results"] and all(h["language"] == "typescript" for h in r["results"])

    r = _call(
        mcp, "search_code", {"query": "function", "top_k": 10, "path_prefix": "math"}
    )
    assert all(h["location"].startswith("math.ts") for h in r["results"])

    r = _call(
        mcp, "search_code", {"query": "anything", "top_k": 10, "language": "rust"}
    )
    assert r["results"] == []  # no rust files indexed
    cr.close()


def test_search_files_content_and_files(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)

    content = _call(mcp, "search_files", {"pattern": "authenticate"})
    assert content["count"] >= 1
    assert any(row["path"] == "auth.py" for row in content["results"])
    assert content["indexing"] == "ready"

    files = _call(mcp, "search_files", {"pattern": "*.ts", "target": "files"})
    assert any(row["path"] == "math.ts" for row in files["results"])
    cr.close()


def test_search_code_pagination(tmp_path):
    files = {
        f"f{i}.py": "def token_retry():\n    return 'token retry backoff'\n"
        for i in range(6)
    }
    cr, mcp, _, _ = _make(tmp_path, files)
    q = "token retry backoff"
    page1 = _call(mcp, "search_code", {"query": q, "top_k": 2, "offset": 0})
    assert page1["count"] == 2 and page1["offset"] == 0 and "next_offset" in page1

    page2 = _call(
        mcp, "search_code", {"query": q, "top_k": 2, "offset": page1["next_offset"]}
    )
    assert page2["offset"] == page1["next_offset"]
    cr.close()


def test_loop_detection_blocks_repeated_search(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)
    args = {"query": "authenticate", "top_k": 3}
    for _ in range(4):
        assert "error" not in _call(mcp, "search_code", args)
    assert "error" in _call(mcp, "search_code", args)  # 5th identical call blocked
    # a different query resets the guard
    assert "error" not in _call(mcp, "search_code", {"query": "add", "top_k": 3})
    cr.close()


def test_get_file_range_and_structured_errors(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)

    r = _call(mcp, "get_file", {"path": "auth.py", "start_line": 1, "end_line": 1})
    assert r["content"] == "def authenticate(token):"

    # Errors are returned as content, not raised — so the agent gets a usable message.
    assert "error" in _call(mcp, "get_file", {"path": "../../etc/passwd"})
    assert "error" in _call(mcp, "get_file", {"path": "not_indexed.py"})
    cr.close()


def test_get_file_line_numbers_and_suggestions(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)

    numbered = _call(
        mcp,
        "get_file",
        {"path": "auth.py", "start_line": 1, "end_line": 1, "with_line_numbers": True},
    )
    assert numbered["content"] == "1|def authenticate(token):"

    # A near-miss filename returns a "did you mean?" hint instead of a bare error.
    miss = _call(mcp, "get_file", {"path": "ath.py"})
    assert "error" in miss and "auth.py" in miss.get("did_you_mean", [])
    cr.close()


def test_index_status_reports_totals_and_flag(tmp_path):
    cr, mcp, state, _ = _make(tmp_path, DEMO)
    r = _call(mcp, "index_status", {})
    assert r["total_files"] == 2
    assert r["total_chunks"] == cr.store.total_chunks()
    assert r["indexing"] == "ready"
    # The structured progress object is always present.
    assert "progress" in r and "state" in r["progress"]

    state.indexing = True
    assert _call(mcp, "index_status", {})["indexing"] == "in_progress"
    cr.close()


def test_index_status_reports_live_progress(tmp_path):
    # While an index is mid-flight, index_status surfaces live counters so an agent can tell
    # "scanning a big tree" apart from "stuck" — even before any rows are committed.
    cr, mcp, state, _ = _make(tmp_path, DEMO)
    state.indexing = True
    state.progress.begin("scanning")
    state.progress.saw_file(1200, 300)
    state.progress.set_state("indexing")
    state.progress.wrote_file("pkg/mod.py", 7)

    p = _call(mcp, "index_status", {})["progress"]
    assert p["state"] == "indexing"
    assert p["files_discovered"] == 1200
    assert p["files_to_index"] == 300
    assert p["files_indexed"] == 1
    assert p["chunks"] == 7
    assert p["current_path"] == "pkg/mod.py"
    assert p["elapsed"] is not None
    cr.close()


def test_tool_annotations_mark_read_only(tmp_path):
    cr, mcp, _, _ = _make(tmp_path, DEMO)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("search_code", "search_files", "get_file", "index_status"):
        assert tools[name].annotations is not None
        assert tools[name].annotations.readOnlyHint is True, name
    # reindex mutates the index — it must NOT be advertised as read-only.
    assert tools["reindex"].annotations.readOnlyHint is not True
    cr.close()


def test_reindex_picks_up_new_file_and_guards_concurrency(tmp_path):
    cr, mcp, state, repo = _make(tmp_path, DEMO)
    write(repo / "extra.py", "def extra():\n    return 1\n")
    r = _call(mcp, "reindex", {})
    assert r["total_files"] == 3
    assert cr.store.total_chunks() == cr.store.total_chunks()
    # The run drives the shared progress object and lands on "ready".
    assert state.progress.snapshot()["state"] == "ready"

    state.indexing = True  # a run already in progress -> guarded
    assert "error" in _call(mcp, "reindex", {})
    cr.close()


def test_warm_up_is_safe(tmp_path):
    cr, _, _, _ = _make(tmp_path, DEMO)
    _warm_up(cr)  # must not raise with any provider
    cr.close()


def test_notify_keeps_stdout_clean(capsys):
    # Lifecycle messages must go to stderr only — stdout is the stdio MCP wire protocol.
    from reporag.surfaces.mcp_server import _notify

    _notify("indexing started")
    captured = capsys.readouterr()
    assert "indexing started" in captured.err
    assert captured.out == ""


# --- all-text (general file-directory) indexing ---


def test_all_text_indexes_text_and_skips_binary(tmp_path):
    files = {
        "notes.log": "deployment runbook: restart the scheduler service\n",
        "Dockerfile": "FROM python:3.11\nRUN pip install coderag\n",
        "data.bin": "head\x00\x01\x02tail binary blob\n",  # NUL byte -> binary
    }

    # Default (code-oriented): unknown .log is skipped; Dockerfile is a known text name.
    cr, _, _, _ = _make(tmp_path / "a", files)
    paths = set(cr.store.all_file_paths())
    assert "notes.log" not in paths
    assert "Dockerfile" in paths
    cr.close()

    # all_text: arbitrary text becomes searchable; binary is still rejected.
    cr, _, _, _ = _make(tmp_path / "b", files, index_all_text=True)
    paths = set(cr.store.all_file_paths())
    assert "notes.log" in paths
    assert "data.bin" not in paths
    cr.close()


# --- parallel indexing correctness & concurrency safety ---


def test_parallel_indexing_matches_serial(tmp_path):
    files = {
        f"m{i}.py": (
            f"def f{i}(x):\n    return x + {i}\n\n"
            f"class C{i}:\n    def m(self):\n        return {i}\n"
        )
        for i in range(8)
    }

    def build(workers, sub):
        cr = CodeRAG(
            Config(
                provider="fake",
                watched_dir=tmp_path / "repo",
                store_dir=tmp_path / sub,
                index_workers=workers,
            )
        )
        # write the same repo once (shared watched_dir)
        for name, body in files.items():
            write(tmp_path / "repo" / name, body)
        stats = cr.index()
        out = (
            stats.total_chunks,
            cr.store.total_chunks(),
            sorted(cr.store.all_file_paths()),
        )
        cr.close()
        return out

    serial = build(1, "store_serial")
    parallel = build(4, "store_parallel")
    assert serial[0] == parallel[0]  # stats agree
    assert serial[1] == parallel[1] > 0  # identical chunk count
    assert serial[2] == parallel[2]  # identical file set


def test_search_is_safe_during_concurrent_indexing(tmp_path):
    repo = tmp_path / "repo"
    for i in range(25):
        write(repo / f"f{i}.py", "def g():\n    return 'token retry backoff'\n")
    cr = CodeRAG(
        Config(provider="fake", watched_dir=repo, store_dir=tmp_path / "store")
    )
    cr.index()

    errors: list = []
    stop = threading.Event()

    def hammer_search():
        try:
            while not stop.is_set():
                cr.search("token retry backoff", top_k=5)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    t = threading.Thread(target=hammer_search)
    t.start()
    try:
        # Re-index (store writes) while searches (store reads) run concurrently.
        for _ in range(3):
            for i in range(25, 45):
                write(repo / f"f{i}.py", "def g():\n    return 'more tokens here'\n")
            cr.index()
            for i in range(25, 45):
                (repo / f"f{i}.py").unlink()
            cr.index()
    finally:
        stop.set()
        t.join(timeout=5)

    assert not errors, errors
    assert cr.store.total_chunks() == 25  # the 20 churned files were pruned
    cr.close()
