"""The single embedded store: LanceDB holds chunk metadata, text (BM25), and vectors (ANN).

This replaces the former SQLite store + separate FAISS index. One LanceDB database at
``store_dir`` with two tables:

* ``files``  — one row per indexed file (``path``, ``content_hash``, ``mtime``, ``size``,
  ``language``): drives incremental change detection.
* ``chunks`` — one row per chunk (``id``, ``path``, ``symbol``, ``kind``, ``language``,
  ``start_line``, ``end_line``, ``text``, ``vector``): both BM25 (over ``text``) and vector
  ANN (over ``vector``) live here, so there is no FAISS↔SQLite coordination to maintain.

``chunks.path`` is denormalized so a file's chunks are deleted with a single ``delete(
"path = …")`` (LanceDB has no foreign keys). The integer ``chunks.id`` is the fusion/hydrate
key (it replaces the FAISS id). Writes are buffered and flushed in batches — LanceDB is
columnar and many tiny appends create severe fragment/version bloat. Reads query committed
data only; the writer owns the buffer (guarded by a lock), so a background index stays safe
alongside live queries (partial results until ``optimize`` runs).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from coderag.retrieval.fusion import reciprocal_rank_fusion
from coderag.types import Chunk, IndexStats, SearchHit

logger = logging.getLogger(__name__)

_CHUNKS = "chunks"
_FILES = "files"
_META_FILE = "meta.json"
_SCHEMA_VERSION = 1
_FLUSH_ROWS = 8192
# LanceDB needs enough rows to train a vector ANN index; below this, brute-force is exact
# and fast, so we skip indexing (also keeps tiny test corpora on the exact path).
_ANN_MIN_ROWS = 256
# Incremental writes append to an *unindexed tail* that every vector query brute-forces.
# Once that tail grows past this many rows, query latency starts to degrade noticeably
# (a few thousand rows already costs tens of ms), so an incremental pass rebuilds the ANN
# index instead of letting the tail grow unbounded. See ``maybe_reindex``.
_ANN_REINDEX_TAIL = 4096
_HYDRATE_COLS = [
    "id",
    "path",
    "symbol",
    "kind",
    "language",
    "start_line",
    "end_line",
    "text",
]
_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _fts_query(query: str) -> str:
    """Reduce an arbitrary query to space-separated tokens (defuses FTS operators)."""
    return " ".join(_FTS_TOKEN.findall(query))


class LanceStore:
    """LanceDB-backed chunk + file store with vector ANN and BM25 search."""

    def __init__(self, store_dir: Path, dim: int) -> None:
        import lancedb

        self.dim = dim
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._dir))
        self._lock = threading.RLock()
        self._chunks_buf: List[Dict[str, Any]] = []
        self._files_buf: List[Dict[str, Any]] = []
        self._next_id = 0
        self._ann_built = False
        # Symbol-index cache (callee graph expansion); invalidated by a write generation.
        self._gen = 0
        self._symbol_index: Optional[Dict[str, List[int]]] = None
        self._symbol_index_gen = -1
        if _CHUNKS in self._db.table_names():
            self._next_id = self._max_id() + 1
        self._refresh_ann_state()

    # --- schema ---

    def _chunks_schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                ("id", pa.int64()),
                ("path", pa.string()),
                ("symbol", pa.string()),
                ("kind", pa.string()),
                ("language", pa.string()),
                ("start_line", pa.int32()),
                ("end_line", pa.int32()),
                ("text", pa.string()),
                ("vector", pa.list_(pa.float32(), self.dim)),
            ]
        )

    def _files_schema(self) -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                ("path", pa.string()),
                ("content_hash", pa.string()),
                ("mtime", pa.float64()),
                ("size", pa.int64()),
                ("language", pa.string()),
                ("indexed_at", pa.float64()),
            ]
        )

    def _chunks_tbl(self) -> Any:
        if _CHUNKS not in self._db.table_names():
            return self._db.create_table(_CHUNKS, schema=self._chunks_schema())
        return self._db.open_table(_CHUNKS)

    def _files_tbl(self) -> Any:
        if _FILES not in self._db.table_names():
            return self._db.create_table(_FILES, schema=self._files_schema())
        return self._db.open_table(_FILES)

    def _max_id(self) -> int:
        tbl = self._db.open_table(_CHUNKS)
        n = tbl.count_rows()
        if n == 0:
            return -1
        rows = tbl.search().select(["id"]).limit(n).to_list()
        return max((int(r["id"]) for r in rows), default=-1)

    # --- provenance / lifecycle ---

    def bootstrap(self, embed_dim: int, embed_model: str) -> bool:
        """Record the embedding model/dim; clear the store if they changed.

        Returns True when a rebuild is required (model/dim changed) — the caller just
        re-indexes into the now-empty tables (there is no separate index to rebuild).
        """
        meta_path = self._dir / _META_FILE
        prev: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt meta
                prev = {}
        changed = bool(prev) and (
            int(prev.get("embed_dim", -1)) != embed_dim
            or prev.get("embed_model") != embed_model
        )
        if changed:
            logger.warning(
                "Embedding model changed (%s/%s -> %s/%s); clearing index.",
                prev.get("embed_model"),
                prev.get("embed_dim"),
                embed_model,
                embed_dim,
            )
            with self._lock:
                for name in (_CHUNKS, _FILES):
                    if name in self._db.table_names():
                        self._db.drop_table(name)
                self._chunks_buf.clear()
                self._files_buf.clear()
                self._next_id = 0
                self._ann_built = False
                self._gen += 1
        meta_path.write_text(
            json.dumps(
                {
                    "embed_model": embed_model,
                    "embed_dim": embed_dim,
                    "schema_version": _SCHEMA_VERSION,
                }
            ),
            encoding="utf-8",
        )
        return changed

    def close(self) -> None:
        with self._lock:
            self._chunks_buf.clear()
            self._files_buf.clear()

    def clear(self) -> None:
        """Drop all data (used by a full rebuild). Keeps the recorded provenance meta."""
        with self._lock:
            for name in (_CHUNKS, _FILES):
                if name in self._db.table_names():
                    self._db.drop_table(name)
            self._chunks_buf.clear()
            self._files_buf.clear()
            self._next_id = 0
            self._ann_built = False
            self._gen += 1

    # --- buffered writes ---

    def _flush(self) -> None:
        if self._chunks_buf:
            self._chunks_tbl().add(self._chunks_buf)
            self._chunks_buf = []
        if self._files_buf:
            self._files_tbl().add(self._files_buf)
            self._files_buf = []

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _delete_path_rows(self, rel: str) -> int:
        """Delete a file's chunk + file rows from the committed tables. Returns chunks gone."""
        names = self._db.table_names()
        removed = 0
        pred = f"path = '{rel.replace(chr(39), chr(39) * 2)}'"
        if _CHUNKS in names:
            ctbl = self._db.open_table(_CHUNKS)
            removed = len(
                ctbl.search().where(pred).select(["id"]).limit(10**9).to_list()
            )
            if removed:
                ctbl.delete(pred)
        if _FILES in names:
            self._db.open_table(_FILES).delete(pred)
        return removed

    def write_file(
        self,
        rel: str,
        language: str,
        content_hash: str,
        mtime: float,
        size: int,
        chunks: Sequence[Chunk],
        vectors: Optional[np.ndarray],
        *,
        replace: bool,
    ) -> Tuple[int, int]:
        """Index one file: (replace its old rows, if any) then buffer its new rows.

        Returns ``(chunks_added, chunks_removed)``. New files take the fully-batched fast
        path (no flush); replacing a changed file flushes + deletes its old rows first, so
        the delete-before-add invariant holds on the single writer thread.
        """
        import time

        with self._lock:
            removed = 0
            if replace:
                self._flush()
                removed = self._delete_path_rows(rel)
            added = 0
            if chunks and vectors is not None:
                mat = np.ascontiguousarray(vectors, dtype="float32")
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                mat = mat / np.where(norms == 0.0, 1.0, norms)
                for chunk, vec in zip(chunks, mat, strict=False):
                    self._chunks_buf.append(
                        {
                            "id": self._next_id,
                            "path": rel,
                            "symbol": chunk.symbol or "",
                            "kind": chunk.kind,
                            "language": chunk.language,
                            "start_line": int(chunk.start_line),
                            "end_line": int(chunk.end_line),
                            "text": chunk.text,
                            "vector": vec.tolist(),
                        }
                    )
                    self._next_id += 1
                    added += 1
            self._files_buf.append(
                {
                    "path": rel,
                    "content_hash": content_hash,
                    "mtime": float(mtime),
                    "size": int(size),
                    "language": language,
                    "indexed_at": time.time(),
                }
            )
            self._gen += 1
            if len(self._chunks_buf) >= _FLUSH_ROWS:
                self._flush()
            return added, removed

    def delete_file(self, rel: str) -> int:
        with self._lock:
            self._flush()
            removed = self._delete_path_rows(rel)
            if removed:
                self._gen += 1
            return removed

    def optimize(self) -> None:
        """Flush, compact, (re)build the BM25 index, and build the vector ANN index at scale."""
        with self._lock:
            self._flush()
            if _CHUNKS not in self._db.table_names():
                return
            tbl = self._db.open_table(_CHUNKS)
            try:
                tbl.optimize()
                tbl.cleanup_old_versions()
            except Exception:  # pragma: no cover - compaction is best-effort
                logger.exception("LanceDB optimize failed (continuing).")
            self._build_search_indexes(tbl)

    def maybe_reindex(self) -> bool:
        """Keep the ANN/FTS indexes fresh on an incremental pass, without a full optimize.

        Incremental writes append rows to an *unindexed tail* that every vector query
        brute-forces; left unbounded that tail is what turns sub-50ms retrieval into
        hundreds of ms. This rebuilds the search indexes only when the tail has grown past
        ``_ANN_REINDEX_TAIL`` (or no ANN index exists yet at scale) — otherwise it is a
        cheap no-op (just a flush). Returns True iff it rebuilt. Skips the compaction that
        ``optimize`` does, so it stays light enough for a watcher to call after each edit.
        """
        with self._lock:
            self._flush()
            if _CHUNKS not in self._db.table_names():
                return False
            tbl = self._db.open_table(_CHUNKS)
            n = tbl.count_rows()
            if n < _ANN_MIN_ROWS:
                self._ann_built = False
                return False
            state = self._vector_index_stats(tbl)
            # No index yet => the whole table is the (brute-forced) tail.
            unindexed = state[1] if state is not None else n
            if state is not None and unindexed <= _ANN_REINDEX_TAIL:
                self._ann_built = True
                return False
            self._build_search_indexes(tbl)
            return True

    def _build_search_indexes(self, tbl: Any) -> None:
        """(Re)build the FTS, scalar (id/path), and vector ANN indexes for ``tbl``.

        Each build is independent and best-effort: a failure is logged loudly and leaves
        that lookup on LanceDB's brute-force/scan fallback rather than aborting the others.
        ``_ann_built`` tracks the *real* state so a silent fallback is observable via
        ``index_kind`` instead of masquerading as an ANN index.
        """
        try:
            tbl.create_fts_index("text", replace=True)
        except Exception:  # pragma: no cover
            logger.exception("LanceDB FTS index build failed (lexical scan fallback).")
        # Scalar indexes make hydrate's ``id IN (...)`` and path deletes index lookups
        # instead of full-table scans (which grow with the corpus).
        for col in ("id", "path"):
            try:
                tbl.create_scalar_index(col, replace=True)
            except Exception:  # pragma: no cover
                logger.exception(
                    "LanceDB scalar index on %s failed (scan fallback).", col
                )
        n = tbl.count_rows()
        if n < _ANN_MIN_ROWS:
            # Brute-force is exact and fast at this scale; no ANN index to maintain.
            self._ann_built = False
            return
        try:
            nlist = max(1, min(int(4 * math.sqrt(n)), n // 39))
            tbl.create_index(
                metric="cosine",
                vector_column_name="vector",
                num_partitions=nlist,
                replace=True,
            )
            self._ann_built = True
        except Exception:  # pragma: no cover - falls back to brute-force search
            logger.exception(
                "LanceDB vector index build failed (brute-force fallback)."
            )
            self._ann_built = False

    def _vector_index_stats(self, tbl: Any) -> Optional[Tuple[int, int]]:
        """``(indexed_rows, unindexed_rows)`` for the vector ANN index, or None if absent.

        The unindexed count is the brute-forced tail; it drives the reindex decision.
        Robust to lancedb version differences in the index/stats object shape.
        """
        try:
            indices = tbl.list_indices()
        except Exception:  # pragma: no cover - older/newer API shape
            return None
        name: Optional[str] = None
        for idx in indices:
            cols = (
                getattr(idx, "columns", None)
                or getattr(idx, "column_names", None)
                or []
            )
            if "vector" in cols:
                name = getattr(idx, "name", None) or getattr(idx, "index_name", None)
                break
        if name is None:
            return None
        try:
            stats = tbl.index_stats(name)
        except Exception:  # pragma: no cover
            return None
        indexed = int(getattr(stats, "num_indexed_rows", 0) or 0)
        unindexed = int(getattr(stats, "num_unindexed_rows", 0) or 0)
        return indexed, unindexed

    def _refresh_ann_state(self) -> None:
        """Set ``_ann_built`` from what is actually on disk (called on open)."""
        if _CHUNKS not in self._db.table_names():
            self._ann_built = False
            return
        self._ann_built = (
            self._vector_index_stats(self._db.open_table(_CHUNKS)) is not None
        )

    @property
    def index_kind(self) -> str:
        return "lancedb-ann" if self._ann_built else "lancedb"

    # --- file metadata / change detection ---

    def get_file_meta(self, rel: str) -> Optional[Dict[str, Any]]:
        self.flush()
        if _FILES not in self._db.table_names():
            return None
        pred = f"path = '{rel.replace(chr(39), chr(39) * 2)}'"
        rows = self._db.open_table(_FILES).search().where(pred).limit(1).to_list()
        return rows[0] if rows else None

    def all_file_metas(self) -> Dict[str, Dict[str, Any]]:
        """Every file's change-detection metadata, in one scan (indexer preload)."""
        self.flush()
        if _FILES not in self._db.table_names():
            return {}
        tbl = self._db.open_table(_FILES)
        rows = (
            tbl.search()
            .select(["path", "content_hash", "mtime", "size"])
            .limit(max(1, tbl.count_rows()))
            .to_list()
        )
        return {r["path"]: r for r in rows}

    def all_file_paths(self) -> List[str]:
        return list(self.all_file_metas().keys())

    # --- retrieval ---

    def vector_search(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if _CHUNKS not in self._db.table_names():
            return []
        q = np.asarray(qvec, dtype="float32").reshape(-1)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        tbl = self._db.open_table(_CHUNKS)
        if tbl.count_rows() == 0:
            return []
        rows = tbl.search(q.tolist()).metric("cosine").select(["id"]).limit(k).to_list()
        return [(int(r["id"]), 1.0 - float(r["_distance"])) for r in rows]

    def lexical_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if _CHUNKS not in self._db.table_names():
            return []
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = (
                self._db.open_table(_CHUNKS)
                .search(match, query_type="fts")
                .select(["id"])
                .limit(k)
                .to_list()
            )
        except Exception:  # pragma: no cover - FTS index not built yet / query rejected
            return []
        return [(int(r["id"]), float(r["_score"])) for r in rows]

    def chunk_ids_for_path(self, rel: str) -> List[int]:
        """The chunk ids belonging to one file (for inspection/tests)."""
        self.flush()
        if _CHUNKS not in self._db.table_names():
            return []
        pred = f"path = '{rel.replace(chr(39), chr(39) * 2)}'"
        tbl = self._db.open_table(_CHUNKS)
        rows = (
            tbl.search()
            .where(pred)
            .select(["id"])
            .limit(max(1, tbl.count_rows()))
            .to_list()
        )
        return [int(r["id"]) for r in rows]

    def hydrate(self, ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not ids or _CHUNKS not in self._db.table_names():
            return {}
        csv = ",".join(str(int(i)) for i in ids)
        rows = (
            self._db.open_table(_CHUNKS)
            .search()
            .where(f"id IN ({csv})")
            .select(_HYDRATE_COLS)
            .limit(len(ids))
            .to_list()
        )
        return {int(r["id"]): r for r in rows}

    def symbol_index(self) -> Dict[str, List[int]]:
        """Map each symbol's bare name -> chunk ids defining it (cached, gen-invalidated)."""
        with self._lock:
            if self._symbol_index is not None and self._symbol_index_gen == self._gen:
                return self._symbol_index
            gen = self._gen
            self._flush()
        index: Dict[str, List[int]] = {}
        if _CHUNKS in self._db.table_names():
            tbl = self._db.open_table(_CHUNKS)
            rows = (
                tbl.search()
                .where("symbol != ''")
                .select(["id", "symbol"])
                .limit(max(1, tbl.count_rows()))
                .to_list()
            )
            for r in rows:
                bare = str(r["symbol"]).rsplit(".", 1)[-1].strip()
                if len(bare) >= 3:
                    index.setdefault(bare, []).append(int(r["id"]))
        with self._lock:
            self._symbol_index = index
            self._symbol_index_gen = gen
        return index

    # --- stats / UI ---

    def total_chunks(self) -> int:
        self.flush()
        if _CHUNKS not in self._db.table_names():
            return 0
        return int(self._db.open_table(_CHUNKS).count_rows())

    def stats(self) -> IndexStats:
        self.flush()
        names = self._db.table_names()
        files = int(self._db.open_table(_FILES).count_rows()) if _FILES in names else 0
        chunks = (
            int(self._db.open_table(_CHUNKS).count_rows()) if _CHUNKS in names else 0
        )
        return IndexStats(total_files=files, total_chunks=chunks)

    def _distinct(self, column: str) -> List[str]:
        self.flush()
        if _CHUNKS not in self._db.table_names():
            return []
        tbl = self._db.open_table(_CHUNKS)
        rows = tbl.search().select([column]).limit(max(1, tbl.count_rows())).to_list()
        return sorted({r[column] for r in rows if r.get(column)})

    def distinct_languages(self) -> List[str]:
        return self._distinct("language")

    def distinct_kinds(self) -> List[str]:
        return self._distinct("kind")

    # --- convenience hybrid search (used by the bake-off scripts; engine uses HybridSearcher) ---

    def search(
        self,
        query: str,
        provider: Any,
        top_k: int = 8,
        *,
        fetch_k: int = 50,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> List[SearchHit]:
        if not query.strip():
            return []
        fetch = max(fetch_k, top_k)
        dense = self.vector_search(provider.embed_query(query), fetch)
        lexical = self.lexical_search(query, fetch)
        similarity = {cid: max(0.0, min(1.0, s)) for cid, s in dense}
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in dense], [cid for cid, _ in lexical]],
            k=rrf_k,
            weights=[dense_weight, lexical_weight],
        )[:top_k]
        if not fused:
            return []
        rows = self.hydrate([cid for cid, _ in fused])
        hits: List[SearchHit] = []
        for cid, score in fused:
            r = rows.get(cid)
            if r is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    path=r["path"],
                    symbol=r["symbol"] or None,
                    kind=r["kind"],
                    language=r["language"],
                    start_line=int(r["start_line"]),
                    end_line=int(r["end_line"]),
                    text=r["text"],
                    score=float(score),
                    similarity=float(similarity.get(cid, 0.0)),
                )
            )
        return hits
