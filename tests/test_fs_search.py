"""Tests for exact filesystem search (:mod:`coderag.fs_search`).

The pure-Python path is the authoritative implementation and is what these tests force
(``use_ripgrep=False``), so the suite never depends on ripgrep being installed. One
consistency test compares the ripgrep fast path against the Python path when rg is present.
"""

from __future__ import annotations

import shutil

import pytest

from coderag.fs_search import search_files
from tests.conftest import write


@pytest.fixture
def tree(tmp_path):
    write(tmp_path / "a.py", "import os\n\ndef alpha():\n    return os.getpid()\n")
    write(tmp_path / "pkg" / "b.py", "def beta():\n    return 'alpha beta'\n")
    write(tmp_path / "notes.txt", "alpha mention in text\n")
    write(tmp_path / "node_modules" / "dep.py", "def alpha():\n    pass\n")  # ignored
    write(tmp_path / ".git" / "cfg", "alpha\n")  # ignored
    return tmp_path


def test_content_search_finds_matches_and_skips_ignored(tree):
    r = search_files(tree, r"def alpha", target="content", use_ripgrep=False)
    paths = {row["path"] for row in r["results"]}
    assert "a.py" in paths
    assert not any("node_modules" in p or p.startswith(".git") for p in paths)
    assert r["ripgrep"] is False


def test_file_glob_restricts_content_search(tree):
    r = search_files(
        tree, r"alpha", target="content", file_glob="*.py", use_ripgrep=False
    )
    assert r["results"]
    assert all(row["path"].endswith(".py") for row in r["results"])


def test_target_files_glob(tree):
    r = search_files(tree, "*.py", target="files", use_ripgrep=False)
    paths = {row["path"] for row in r["results"]}
    assert {"a.py", "pkg/b.py"} <= paths
    assert not any("node_modules" in p or p.startswith(".git") for p in paths)


def test_output_modes(tree):
    files_only = search_files(
        tree, r"alpha", target="content", output_mode="files_only", use_ripgrep=False
    )
    assert all(set(row) == {"path"} for row in files_only["results"])

    counts = search_files(
        tree, r"alpha", target="content", output_mode="count", use_ripgrep=False
    )
    assert counts["results"] and all("count" in row for row in counts["results"])


def test_context_lines(tree):
    r = search_files(
        tree, r"return os\.getpid", target="content", context=1, use_ripgrep=False
    )
    row = r["results"][0]
    assert "before" in row and "after" in row
    assert any("def alpha" in b for b in row["before"])


def test_pagination(tmp_path):
    for i in range(10):
        write(tmp_path / f"f{i}.py", "needle\n")
    r = search_files(tmp_path, "needle", target="content", limit=4, use_ripgrep=False)
    assert r["count"] == 4 and r["truncated"] is True and r["next_offset"] == 4
    assert "offset=4" in r["hint"]

    last = search_files(
        tmp_path, "needle", target="content", limit=4, offset=8, use_ripgrep=False
    )
    assert last["count"] == 2 and last["truncated"] is False and "hint" not in last


def test_redaction(tmp_path):
    # Obvious low-entropy placeholder (not a real secret); gitleaks:allow keeps the
    # secret-scanner from flagging this test fixture.
    fake = "xxxxxxxxxxxx"
    write(tmp_path / "s.py", f'token = "{fake}"\n')  # gitleaks:allow
    masked = search_files(tmp_path, "token", target="content", use_ripgrep=False)
    assert "***" in masked["results"][0]["line"]

    raw = search_files(
        tmp_path, "token", target="content", redact=False, use_ripgrep=False
    )
    assert fake in raw["results"][0]["line"]


def test_invalid_regex_returns_error(tmp_path):
    assert "error" in search_files(tmp_path, "(", target="content", use_ripgrep=False)


def test_binary_files_skipped(tmp_path):
    write(tmp_path / "ok.py", "needle\n")
    (tmp_path / "blob.py").write_bytes(b"needle\x00\x01\x02")
    r = search_files(tmp_path, "needle", target="content", use_ripgrep=False)
    assert {row["path"] for row in r["results"]} == {"ok.py"}


def test_ripgrep_matches_python_path(tree):
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    rg = search_files(tree, r"alpha", target="content", use_ripgrep=True)
    py = search_files(tree, r"alpha", target="content", use_ripgrep=False)
    assert rg["ripgrep"] is True and py["ripgrep"] is False
    key = lambda res: {(r["path"], r["line_number"]) for r in res["results"]}  # noqa: E731
    assert key(rg) == key(py)
