"""Tests for the ChromaDB store — the single backend (metadata + BM25 + vectors)."""

from __future__ import annotations

from coderag.embeddings.fake_provider import FakeEmbeddingProvider
from coderag.store.chroma_store import ChromaStore
from coderag.types import Chunk


def _chunk(text: str, sym: str = "f", kind: str = "function", start: int = 1) -> Chunk:
    return Chunk(
        text=text,
        start_line=start,
        end_line=start + 1,
        language="python",
        symbol=sym,
        kind=kind,
    )


def _store(tmp_path):
    prov = FakeEmbeddingProvider()
    return ChromaStore(tmp_path / "store", prov.dim), prov


def _add(st, prov, rel, chunks, *, replace=False, chash="h", mtime=1.0, size=10):
    vecs = prov.embed_documents([c.text for c in chunks])
    return st.write_file(
        rel, "python", chash, mtime, size, chunks, vecs, replace=replace
    )


def test_write_stats_lexical_and_hybrid(tmp_path):
    st, prov = _store(tmp_path)
    _add(
        st,
        prov,
        "auth.py",
        [_chunk("def authenticate(token): retry backoff", "authenticate")],
    )
    _add(st, prov, "math.py", [_chunk("def add(a, b): return a + b", "add")])
    st.optimize()

    s = st.stats()
    assert s.total_files == 2 and s.total_chunks == 2
    assert st.total_chunks() == 2

    lex = st.lexical_search("authenticate", 5)
    assert lex
    top = lex[0][0]
    assert st.hydrate([top])[top]["path"] == "auth.py"

    hits = st.search("authenticate token", prov, top_k=5)
    assert hits and {h.path for h in hits} <= {"auth.py", "math.py"}
    assert all(h.start_line >= 1 and h.text for h in hits)


def test_change_detection_metadata(tmp_path):
    st, prov = _store(tmp_path)
    _add(st, prov, "a.py", [_chunk("x")], chash="h1", mtime=12.5, size=4096)
    meta = st.get_file_meta("a.py")
    assert meta is not None
    assert meta["content_hash"] == "h1"
    assert meta["size"] == 4096 and abs(meta["mtime"] - 12.5) < 1e-9
    assert set(st.all_file_metas()) == {"a.py"}
    assert st.all_file_paths() == ["a.py"]
    assert st.get_file_meta("missing.py") is None


def test_replace_does_not_duplicate(tmp_path):
    st, prov = _store(tmp_path)
    _add(st, prov, "a.py", [_chunk("def alpha(): return 1", "alpha")])
    assert st.total_chunks() == 1
    added, removed = _add(
        st,
        prov,
        "a.py",
        [
            _chunk("def alpha(): return 100", "alpha"),
            _chunk("def gamma(): return 3", "gamma"),
        ],
        replace=True,
        chash="h2",
    )
    assert added == 2 and removed == 1
    assert st.total_chunks() == 2
    rows = st.hydrate(st.chunk_ids_for_path("a.py"))
    joined = "\n".join(r["text"] for r in rows.values())
    assert "return 100" in joined
    assert "return 1\n" not in joined


def test_delete_file(tmp_path):
    st, prov = _store(tmp_path)
    _add(st, prov, "a.py", [_chunk("a")])
    _add(st, prov, "b.py", [_chunk("b")])
    assert st.delete_file("a.py") == 1
    assert st.all_file_paths() == ["b.py"]
    assert st.total_chunks() == 1


def test_bootstrap_clears_on_model_change(tmp_path):
    st, prov = _store(tmp_path)
    assert st.bootstrap(prov.dim, "fake-16") is False
    _add(st, prov, "a.py", [_chunk("a")])
    st.optimize()
    assert st.total_chunks() == 1
    assert st.bootstrap(prov.dim, "fake-16") is False  # unchanged
    assert st.total_chunks() == 1
    assert st.bootstrap(prov.dim, "other-model") is True  # model changed -> cleared
    assert st.total_chunks() == 0


def test_symbol_index_caches_and_invalidates(tmp_path):
    st, prov = _store(tmp_path)
    _add(st, prov, "a.py", [_chunk("def compute_tax(): pass", "compute_tax")])
    idx = st.symbol_index()
    assert "compute_tax" in idx
    assert st.symbol_index() is idx  # cached while nothing changed
    _add(st, prov, "b.py", [_chunk("def brand_new(): pass", "brand_new")])
    idx2 = st.symbol_index()
    assert idx2 is not idx and "brand_new" in idx2


def test_distinct_and_fts_sanitization(tmp_path):
    st, prov = _store(tmp_path)
    _add(st, prov, "a.py", [_chunk("def parse_config(): return 1", "parse_config")])
    st.optimize()
    assert st.distinct_languages() == ["python"]
    assert "function" in st.distinct_kinds()
    assert st.lexical_search("parse_config", 5)  # plain token
    assert st.lexical_search("parse_config::*", 5)  # operators sanitized, no raise
    assert st.lexical_search("", 5) == []  # empty query


def test_maybe_reindex_builds_ann_on_incremental_tail(tmp_path):
    """An incrementally-grown corpus must end up on the ANN index, not brute-force.

    Regression: incremental writes only flushed, so rows piled into an unindexed tail
    that every vector query brute-forced (sub-50ms retrieval -> hundreds of ms). A pass
    over the reindex threshold should (re)build the index and drain the tail.
    """
    from coderag.store.chroma_store import _ANN_MIN_ROWS

    st, prov = _store(tmp_path)
    # Index past the ANN minimum without ever calling optimize() (a watcher session).
    n_files = (_ANN_MIN_ROWS // 4) + 5
    for f in range(n_files):
        chunks = [
            _chunk(f"def fn_{f}_{i}(): return {i}", f"fn_{f}_{i}") for i in range(4)
        ]
        _add(st, prov, f"m{f}.py", chunks)
    st.flush()

    # No ANN index yet -> brute-force.
    assert st.index_kind == "chromadb"
    assert st._vector_index_stats(st._db.open_table("chunks")) is None

    assert st.maybe_reindex() is True
    assert st.index_kind == "chromadb-ann"
    indexed, unindexed = st._vector_index_stats(st._db.open_table("chunks"))
    assert indexed == st.total_chunks() and unindexed == 0

    # No new rows -> cheap no-op, index stays.
    assert st.maybe_reindex() is False
    assert st.index_kind == "chromadb-ann"

    # State is recovered from disk on reopen (not just in-memory).
    reopened = ChromaStore(tmp_path / "store", prov.dim)
    assert reopened.index_kind == "chromadb-ann"


def test_clear_empties_store(tmp_path):
    st, prov = _store(tmp_path)
    _add(st, prov, "a.py", [_chunk("a")])
    st.optimize()
    assert st.total_chunks() == 1
    st.clear()
    assert st.total_chunks() == 0
    assert st.all_file_paths() == []
