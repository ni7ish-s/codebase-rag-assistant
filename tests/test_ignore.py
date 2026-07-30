"""Tests for the shared ignore-aware walker (:func:`coderag._ignore.walk_files`)."""

from __future__ import annotations

from reporag._ignore import walk_files
from reporag.config import DEFAULT_IGNORE_GLOBS
from tests.conftest import write


def _rels(root, **kw) -> set[str]:
    return {rel for _, rel in walk_files(root, DEFAULT_IGNORE_GLOBS, **kw)}


def test_gitignore_negation_and_dir_only(tmp_path):
    write(tmp_path / ".gitignore", "*.log\nout/\n!keep.log\n")
    write(tmp_path / "a.py", "x\n")
    write(tmp_path / "debug.log", "x\n")
    write(tmp_path / "keep.log", "x\n")
    write(tmp_path / "out" / "x.py", "x\n")  # "out" is not a built-in default ignore
    rels = _rels(tmp_path)
    assert "a.py" in rels
    assert "keep.log" in rels  # re-included by negation
    assert "debug.log" not in rels  # ignored by *.log
    assert not any(r.startswith("out/") for r in rels)  # dir pruned by "out/"


def test_nested_gitignore_scopes_to_subtree(tmp_path):
    write(tmp_path / "a.txt", "x\n")
    write(tmp_path / "sub" / ".gitignore", "*.txt\n")
    write(tmp_path / "sub" / "b.txt", "x\n")
    write(tmp_path / "sub" / "c.py", "x\n")
    rels = _rels(tmp_path)
    assert "a.txt" in rels  # root .txt unaffected by sub/.gitignore
    assert "sub/c.py" in rels
    assert "sub/b.txt" not in rels  # ignored by the nested rule


def test_gitignore_can_be_disabled(tmp_path):
    write(tmp_path / ".gitignore", "*.log\n")
    write(tmp_path / "debug.log", "x\n")
    assert "debug.log" not in _rels(tmp_path, use_gitignore=True)
    assert "debug.log" in _rels(tmp_path, use_gitignore=False)


def test_indexer_and_fs_search_agree_on_gitignore(tmp_path):
    # The shared-walker invariant: semantic index and exact search see the same files.
    from reporag.api import CodeRAG
    from reporag.config import Config

    repo = tmp_path / "repo"
    write(repo / ".gitignore", "ignored/\n*.log\n")
    write(repo / "keep.py", "def k():\n    return 1\n")
    write(repo / "ignored" / "x.py", "def x():\n    return 1\n")
    write(repo / "note.log", "hi\n")
    cr = CodeRAG(
        Config(provider="fake", watched_dir=repo, store_dir=tmp_path / "store")
    )
    cr.index()
    indexed = set(cr.store.all_file_paths())
    assert "keep.py" in indexed
    assert not any(p.startswith("ignored/") for p in indexed)

    res = cr.search_files("*", target="files", use_ripgrep=False)
    found = {row["path"] for row in res["results"]}
    assert "keep.py" in found
    assert not any(p.startswith("ignored/") for p in found)
    assert "note.log" not in found  # .log ignored, so exact search skips it too
    cr.close()
