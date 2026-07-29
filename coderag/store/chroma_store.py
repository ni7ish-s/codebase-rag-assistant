"""ChromaDB-backed chunk store, with an in-memory BM25 index for lexical search.

Drop-in replacement for ``ChromaStore``. ChromaDB natively handles vector storage +
ANN search (HNSW) and chunk metadata, but has no built-in BM25/FTS, so lexical search
is served by a ``rank_bm25.BM25Okapi`` index kept in memory and rebuilt lazily whenever
the store's write-generation changes (same caching pattern as the symbol index).

File-level metadata (content hash / mtime / size, used for incremental re-indexing) is
kept in a small JSON sidecar (``files_meta.json``) rather than a Chroma collection,
since Chroma is built for embeddings, not this kind of small structured lookup table.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from coderag.retrieval.fusion import reciprocal_rank_fusion
from coderag.types import Chunk, IndexStats, SearchHit

logger = logging.getLogger(__name__)

_COLLECTION = "chunks"
_META_FILE = "meta.json"
_FILES_META_FILE = "files_meta.json"
_SCHEMA_VERSION = 1
_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class ChromaStore:
    """ChromaDB-backed chunk + file store with vector ANN and BM25 search."""

    def __init__(self, store_dir: Path, dim: int) -> None:
        import chromadb

        self.dim = dim
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._dir))
        self._lock = threading.RLock()
        self._next_id = 0
        self._gen = 0

        self._bm25 = None
        self._bm25_ids: List[int] = []
        self._bm25_gen = -1

        self._symbol_index: Optional[Dict[str, List[int]]] = None
        self._symbol_index_gen = -1

        self._files_meta_path = self._dir / _FILES_META_FILE
        self._files_meta = self._load_files_meta()

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        n = self._collection.count()
        if n:
            ids = self._collection.get(include=[]).get("ids", [])
            self._next_id = (max((int(i) for i in ids), default=-1) + 1) if ids else 0

    # --- provenance / lifecycle ---

    def bootstrap(self, embed_dim: int, embed_model: str) -> bool:
        """Record the embedding model/dim; clear the store if they changed."""
        meta_path = self._dir / _META_FILE
        prev: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                prev = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
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
            self.clear()
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
        pass

    def clear(self) -> None:
        """Drop all data (used by a full rebuild). Keeps the recorded provenance meta."""
        with self._lock:
            try:
                self._client.delete_collection(_COLLECTION)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            self._files_meta = {}
            self._save_files_meta()
            self._next_id = 0
            self._gen += 1
            self._bm25 = None
            self._symbol_index = None

    def flush(self) -> None:
        pass

    # --- files metadata sidecar ---

    def _load_files_meta(self) -> Dict[str, Dict[str, Any]]:
        if self._files_meta_path.exists():
            try:
                return json.loads(self._files_meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_files_meta(self) -> None:
        self._files_meta_path.write_text(
            json.dumps(self._files_meta), encoding="utf-8"
        )

    # --- writes ---

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
        import time

        with self._lock:
            removed = 0
            if replace:
                removed = self._delete_path_chunks(rel)

            added = 0
            if chunks and vectors is not None:
                mat = np.ascontiguousarray(vectors, dtype="float32")
                ids: List[str] = []
                embeddings: List[List[float]] = []
                documents: List[str] = []
                metadatas: List[Dict[str, Any]] = []
                for chunk, vec in zip(chunks, mat, strict=False):
                    cid = self._next_id
                    self._next_id += 1
                    ids.append(str(cid))
                    embeddings.append(vec.tolist())
                    documents.append(chunk.text)
                    metadatas.append(
                        {
                            "path": rel,
                            "symbol": chunk.symbol or "",
                            "kind": chunk.kind,
                            "language": chunk.language,
                            "start_line": int(chunk.start_line),
                            "end_line": int(chunk.end_line),
                        }
                    )
                    added += 1
                if ids:
                    self._collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )

            self._files_meta[rel] = {
                "path": rel,
                "content_hash": content_hash,
                "mtime": float(mtime),
                "size": int(size),
                "language": language,
                "indexed_at": time.time(),
            }
            self._save_files_meta()
            self._gen += 1
            return added, removed

    def _delete_path_chunks(self, rel: str) -> int:
        existing = self._collection.get(where={"path": rel}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def delete_file(self, rel: str) -> int:
        with self._lock:
            removed = self._delete_path_chunks(rel)
            self._files_meta.pop(rel, None)
            self._save_files_meta()
            if removed:
                self._gen += 1
            return removed

    def optimize(self) -> None:
        pass

    def maybe_reindex(self) -> bool:
        return False

    @property
    def index_kind(self) -> str:
        return "chroma-hnsw"

    # --- file metadata / change detection ---

    def get_file_meta(self, rel: str) -> Optional[Dict[str, Any]]:
        return self._files_meta.get(rel)

    def all_file_metas(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._files_meta)

    def all_file_paths(self) -> List[str]:
        return list(self._files_meta.keys())

    # --- retrieval ---

    def vector_search(self, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self._collection.count() == 0:
            return []
        q = np.asarray(qvec, dtype="float32").reshape(-1).tolist()
        res = self._collection.query(
            query_embeddings=[q],
            n_results=min(k, self._collection.count()),
            include=["distances"],
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [(int(i), 1.0 - float(d)) for i, d in zip(ids, dists)]

    def _rebuild_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        if self._collection.count() == 0:
            self._bm25 = None
            self._bm25_ids = []
            self._bm25_gen = self._gen
            return
        got = self._collection.get(include=["documents"])
        ids = [int(i) for i in got.get("ids", [])]
        docs = got.get("documents", [])
        tokenized = [_tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._bm25_ids = ids
        self._bm25_gen = self._gen

    def lexical_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        with self._lock:
            if self._bm25 is None or self._bm25_gen != self._gen:
                self._rebuild_bm25()
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:k]
        return [
            (self._bm25_ids[i], float(scores[i])) for i in order if scores[i] > 0
        ]

    def chunk_ids_for_path(self, rel: str) -> List[int]:
        got = self._collection.get(where={"path": rel}, include=[])
        return [int(i) for i in got.get("ids", [])]

    def hydrate(self, ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        str_ids = [str(int(i)) for i in ids]
        got = self._collection.get(ids=str_ids, include=["documents", "metadatas"])
        out: Dict[int, Dict[str, Any]] = {}
        for i, doc, meta in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", [])):
            out[int(i)] = {
                "id": int(i),
                "path": meta["path"],
                "symbol": meta.get("symbol", ""),
                "kind": meta["kind"],
                "language": meta["language"],
                "start_line": int(meta["start_line"]),
                "end_line": int(meta["end_line"]),
                "text": doc,
            }
        return out

    def symbol_index(self) -> Dict[str, List[int]]:
        with self._lock:
            if self._symbol_index is not None and self._symbol_index_gen == self._gen:
                return self._symbol_index
            gen = self._gen
        index: Dict[str, List[int]] = {}
        if self._collection.count():
            got = self._collection.get(include=["metadatas"])
            for i, meta in zip(got.get("ids", []), got.get("metadatas", [])):
                symbol = meta.get("symbol") or ""
                if not symbol:
                    continue
                bare = symbol.rsplit(".", 1)[-1].strip()
                if len(bare) >= 3:
                    index.setdefault(bare, []).append(int(i))
        with self._lock:
            self._symbol_index = index
            self._symbol_index_gen = gen
        return index

    # --- stats / UI ---

    def total_chunks(self) -> int:
        return self._collection.count()

    def stats(self) -> IndexStats:
        return IndexStats(
            total_files=len(self._files_meta), total_chunks=self._collection.count()
        )

    def _distinct(self, column: str) -> List[str]:
        if self._collection.count() == 0:
            return []
        got = self._collection.get(include=["metadatas"])
        return sorted({m[column] for m in got.get("metadatas", []) if m.get(column)})

    def distinct_languages(self) -> List[str]:
        return self._distinct("language")

    def distinct_kinds(self) -> List[str]:
        return self._distinct("kind")

    # --- convenience hybrid search (mirrors ChromaStore.search) ---

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
