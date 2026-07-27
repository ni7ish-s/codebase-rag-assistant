"""P3 tests: incremental indexing, the no-duplicate invariant, and pruning."""

from __future__ import annotations

from pathlib import Path

from coderag.api import CodeRAG
from coderag.types import IndexProgress
from tests.conftest import write


def _cr(config) -> CodeRAG:
    config.watched_dir.mkdir(parents=True, exist_ok=True)
    return CodeRAG(config)


def test_index_creates_chunks(config):
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    write(config.watched_dir / "b.py", "def beta():\n    return 2\n")
    stats = cr.index()
    assert stats.files_indexed == 2
    assert stats.total_chunks >= 2
    assert cr.store.total_chunks() == stats.total_chunks


def test_unchanged_files_are_skipped(config):
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    cr.index()
    stats2 = cr.index()  # nothing changed
    assert stats2.files_indexed == 0
    assert stats2.files_skipped == 1


def test_editing_a_file_does_not_duplicate(config):
    cr = _cr(config)
    path = config.watched_dir / "a.py"
    write(path, "def alpha():\n    return 1\n")
    cr.index()
    chunks_before = cr.store.total_chunks()
    assert chunks_before >= 1

    # Edit and reindex.
    write(path, "def alpha():\n    return 100\n\ndef gamma():\n    return 3\n")
    stats = cr.index()
    assert stats.chunks_removed >= 1  # old chunks were deleted first
    # The new content is searchable; the stale content is gone (no duplicates).
    rows = cr.store.hydrate(cr.store.chunk_ids_for_path("a.py"))
    joined = "\n".join(r["text"] for r in rows.values())
    assert "return 100" in joined
    assert "return 1\n" not in joined or "return 100" in joined


def test_deleted_file_is_pruned(config):
    cr = _cr(config)
    a = config.watched_dir / "a.py"
    b = config.watched_dir / "b.py"
    write(a, "def alpha():\n    return 1\n")
    write(b, "def beta():\n    return 2\n")
    cr.index()
    chunks_with_b = cr.store.total_chunks()

    b.unlink()
    stats = cr.index()
    assert stats.files_removed == 1
    assert "b.py" not in cr.store.all_file_paths()
    assert cr.store.total_chunks() < chunks_with_b  # b's chunks are gone


def test_ignored_dirs_are_skipped(config):
    cr = _cr(config)
    write(config.watched_dir / "src" / "a.py", "def alpha():\n    return 1\n")
    write(config.watched_dir / "node_modules" / "x.js", "function x(){return 1;}\n")
    write(config.watched_dir / ".git" / "hooks.py", "def hook():\n    return 1\n")
    cr.index()
    paths = cr.store.all_file_paths()
    assert "src/a.py" in paths
    assert not any("node_modules" in p for p in paths)
    assert not any(".git" in p for p in paths)


def test_dependency_and_cache_dirs_are_skipped(config):
    cr = _cr(config)
    write(config.watched_dir / "src" / "a.py", "def alpha():\n    return 1\n")
    write(config.watched_dir / "site-packages" / "dep.py", "def dep():\n    return 1\n")
    write(config.watched_dir / ".cache" / "c.py", "def c():\n    return 1\n")
    cr.index()
    paths = cr.store.all_file_paths()
    assert "src/a.py" in paths
    assert not any("site-packages" in p for p in paths)
    assert not any(".cache" in p for p in paths)


def test_full_rebuild_resets(config):
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    cr.index()
    n1 = cr.store.total_chunks()
    stats = cr.index(full=True)
    assert stats.total_chunks == n1  # same content, rebuilt cleanly
    assert cr.store.total_chunks() == n1


def test_get_file_line_numbers_use_chunk_convention(config):
    # A bare \r is a line break for str.splitlines() but NOT for the chunker's
    # split("\n"); get_file must use the chunker's convention so a returned range
    # matches the line numbers a SearchHit reports.
    cr = _cr(config)
    content = "line1\rstill_line1\nline2\nline3\n"
    write(config.watched_dir / "f.txt", content)
    cr.index()
    expected = content.split("\n")
    assert cr.get_file("f.txt", 1, 1) == expected[0]  # "line1\rstill_line1"
    assert cr.get_file("f.txt", 2, 2) == expected[1]  # "line2"
    assert cr.get_file("f.txt", 1, 2) == "\n".join(expected[:2])


def test_stat_skip_avoids_reread_of_unchanged_files(config, monkeypatch):
    # On a re-index, an untouched file must be skipped via the cheap (size, mtime) check
    # WITHOUT reading its bytes — the dominant cost saver for a large tree.
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    cr.index()

    orig_read = Path.read_bytes

    def boom(self: Path) -> bytes:
        if self.name == "a.py":
            raise AssertionError("re-read an unchanged file instead of stat-skipping")
        return orig_read(self)

    monkeypatch.setattr(Path, "read_bytes", boom)
    stats = cr.index()
    assert stats.files_indexed == 0
    assert stats.files_skipped == 1


def test_index_progress_is_reported(config, capsys):
    # progress=True narrates the run on stderr so a long index isn't a silent wait.
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    cr.indexer.index(progress=True)
    err = capsys.readouterr().err
    assert "Scanning" in err  # discovery phase is announced
    assert "✓ Indexed" in err  # final summary line


def test_index_progress_is_silent_when_off(config, capsys):
    # progress=False (the default, and what the MCP background index uses) stays quiet.
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    cr.indexer.index(progress=False)
    err = capsys.readouterr().err
    assert "Scanning" not in err and "✓ Indexed" not in err


def test_index_reports_live_progress(config):
    # A live IndexProgress is updated as the run proceeds and ends at "ready" — this is what
    # makes the MCP index_status legible instead of showing 0 while the tree is scanned.
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    write(config.watched_dir / "b.py", "def beta():\n    return 2\n")
    prog = IndexProgress()
    assert prog.snapshot()["state"] == "idle"

    cr.indexer.index(live=prog)
    snap = prog.snapshot()
    assert snap["state"] == "ready"
    assert snap["files_indexed"] == 2
    assert snap["files_discovered"] == 2
    assert snap["chunks"] >= 2
    assert snap["started_at"] is not None
    assert snap["elapsed"] is not None
    assert snap["last_error"] is None


def test_index_progress_records_failure(config, monkeypatch):
    # If a run raises after begin(), the caller marks the progress "failed" with the error.
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    prog = IndexProgress()

    def boom(*args, **kwargs):
        raise RuntimeError("embedding exploded")

    monkeypatch.setattr(cr.indexer, "_embed_and_write", boom)
    prog.begin("scanning")
    try:
        cr.indexer.index(live=prog)
    except RuntimeError as exc:
        prog.finish("failed", str(exc))
    snap = prog.snapshot()
    assert snap["state"] == "failed"
    assert "embedding exploded" in (snap["last_error"] or "")


def test_index_survives_reopen(config, tmp_path):
    cr = _cr(config)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    cr.index()
    n = cr.store.total_chunks()
    cr.close()

    cr2 = CodeRAG(config)
    assert cr2.store.total_chunks() == n  # persisted across reopen
