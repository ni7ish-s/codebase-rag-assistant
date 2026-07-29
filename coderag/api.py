"""The public CodeRAG facade — the one object every surface (CLI, HTTP, UI) routes through.

Holds the wired-together engine: embedding provider, the ChromaDB store (chunk metadata +
text/BM25 + vectors/ANN in one place), the indexer, and the hybrid searcher. Collaborators
are built lazily so constructing a ``CodeRAG`` is cheap and importing this module pulls in no
heavy dependencies.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from coderag._lines import split_lines
from coderag.config import Config
from coderag.types import IndexProgress, IndexStats, SearchHit

if TYPE_CHECKING:  # avoid import-time cost / cycles
    from coderag.embeddings import EmbeddingProvider
    from coderag.indexer import Indexer
    from coderag.retrieval.search import HybridSearcher
    from coderag.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class CodeRAG:
    """High-level entry point for indexing and searching a codebase."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.from_env()
        self._provider: Optional["EmbeddingProvider"] = None
        self._store: Optional["ChromaStore"] = None
        self._indexer: Optional["Indexer"] = None
        self._searcher: Optional["HybridSearcher"] = None
        # Serializes all indexing/deletion so concurrent writers (the CLI, the HTTP
        # surface, the MCP server's background index, and the live watcher) can't
        # interleave a file's delete-before-add sequence. Reads (search) are unaffected.
        self._index_lock = threading.Lock()
        # Guards the lazy construction of the collaborators below. The MCP server now serves
        # the protocol before warm-up finishes, so a query can land while the background
        # bootstrap is still building the store/provider — without this lock two threads could
        # each construct a second (conflicting) ChromaStore. Reentrant because the properties
        # depend on each other (e.g. ``store`` reads ``provider`` while holding the lock).
        self._build_lock = threading.RLock()

    # --- lazily constructed collaborators ---

    @property
    def provider(self) -> "EmbeddingProvider":
        if self._provider is None:
            from coderag.embeddings import get_provider

            with self._build_lock:
                if self._provider is None:
                    self._provider = get_provider(self.config)
        return self._provider

    @property
    def store(self) -> "ChromaStore":
        if self._store is None:
            from coderag.store.chroma_store import ChromaStore

            with self._build_lock:
                if self._store is None:
                    self.config.store_dir.mkdir(parents=True, exist_ok=True)
                    store = ChromaStore(self.config.store_dir, self.provider.dim)
                    # Clears the store when the embedding model/dim changed; a re-index then
                    # repopulates the now-empty tables (no separate cache to rebuild).
                    store.bootstrap(self.provider.dim, self.provider.model_id)
                    self._store = store
        return self._store

    @property
    def indexer(self) -> "Indexer":
        if self._indexer is None:
            from coderag.indexer import Indexer

            with self._build_lock:
                if self._indexer is None:
                    self._indexer = Indexer(self.config, self.provider, self.store)
        return self._indexer

    @property
    def searcher(self) -> "HybridSearcher":
        if self._searcher is None:
            from coderag.retrieval.rerank import get_reranker
            from coderag.retrieval.search import HybridSearcher

            with self._build_lock:
                if self._searcher is None:
                    self._searcher = HybridSearcher(
                        self.config,
                        self.provider,
                        self.store,
                        reranker=get_reranker(self.config),
                    )
        return self._searcher

    # --- public operations ---

    def index(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        full: bool = False,
        live: Optional[IndexProgress] = None,
    ) -> IndexStats:
        """Incrementally index ``path`` (defaults to the configured watched dir).

        Only files whose content hash changed are re-embedded. Pass ``full=True`` to
        force a clean rebuild. Pass ``live`` (an :class:`IndexProgress`) to receive live,
        pollable progress — the MCP server uses this so ``index_status`` reflects the run.
        """
        target = Path(path).expanduser() if path else self.config.watched_dir
        with self._index_lock:
            return self.indexer.index(target, full=full, live=live)

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchHit]:
        """Hybrid (dense + lexical) search over the indexed codebase."""
        return self.searcher.search(query, top_k or self.config.top_k)

    def search_files(self, pattern: str, **kwargs: Any) -> dict:
        """Exact regex/glob search over the workspace (the complement to ``search``).

        Thin pass-through to :func:`coderag.fs_search.search_files`, wired to the
        configured ``watched_dir`` and ``ignore_globs`` so it sees exactly the same
        files the indexer does. See that function for the keyword arguments.
        """
        from coderag.fs_search import search_files

        return search_files(
            self.config.watched_dir,
            pattern,
            ignore_globs=self.config.ignore_globs,
            use_gitignore=self.config.use_gitignore,
            **kwargs,
        )

    def suggest_paths(self, path: Union[str, Path], n: int = 3) -> List[str]:
        """Indexed paths whose name is closest to ``path`` — for "did you mean?" hints."""
        import difflib

        name = Path(str(path)).name
        candidates = self.store.all_file_paths()
        # Match on basename first (agents often pass a bare filename), then full path.
        by_name = {c: Path(c).name for c in candidates}
        close = difflib.get_close_matches(name, list(by_name.values()), n=n, cutoff=0.5)
        hits = [c for c, base in by_name.items() if base in close]
        if not hits:
            hits = difflib.get_close_matches(str(path), candidates, n=n, cutoff=0.4)
        return hits[:n]

    def get_file(
        self,
        path: Union[str, Path],
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Return the contents of an indexed file, optionally a 1-based line range.

        Only files that are actually in the index can be read — this is a defense in
        depth so the endpoint can't be used to read arbitrary files (e.g. ``.env`` or
        ``.git`` contents) that merely happen to sit under the watched root.
        """
        root = self.config.watched_dir.resolve()
        full = (root / Path(path)).resolve()
        if root not in full.parents and full != root:
            raise ValueError(f"Path escapes the indexed root: {path}")
        rel = full.relative_to(root).as_posix()
        if self.store.get_file_meta(rel) is None:
            raise FileNotFoundError(f"Not an indexed file: {path}")
        # Decode raw bytes exactly as the indexer does — no universal-newline
        # translation — so line numbers line up with the chunker (Path.read_text
        # would collapse \r and \r\n to \n and desync the numbering).
        text = full.read_bytes().decode("utf-8", errors="replace")
        if start_line is None and end_line is None:
            return text
        # Same line-splitting as the chunker, so a returned range matches the line
        # numbers a SearchHit reports (str.splitlines() would desync on CRLF/\f/…).
        lines = split_lines(text)
        lo = max(0, (start_line or 1) - 1)
        hi = min(len(lines), end_line or len(lines))
        return "\n".join(lines[lo:hi])

    def delete_path(self, path: Union[str, Path]) -> int:
        """Forget a file that was removed from disk. Returns chunks removed."""
        root = self.config.watched_dir.resolve()
        try:
            rel = Path(path).resolve().relative_to(root).as_posix()
        except ValueError:
            return 0
        with self._index_lock:
            return self.store.delete_file(rel)

    def warm(self) -> None:
        """Eagerly load the provider, store, and embedding model — and the search path.

        Done at server startup so the first query — and the demo UI's search-speed
        badge — reflect warm performance, not the one-off lazy load. A real search is
        run (not just an embed) because the store's vector/FTS/scalar indexes and
        ChromaDB's query path are loaded lazily on first use; warming only the model
        leaves that cold-load to land on the first user query, where it shows up as a
        large ``store_ms``. Best-effort: warm-up failures must not block startup.
        """
        self.status()  # builds provider/store
        self.provider.embed_query("warm up")  # loads the model + JITs the query path
        try:
            # Exercise the full retrieval path (vector + lexical + hydrate) so the
            # store's indexes are resident before the first real query.
            self.search("warm up", top_k=1)
        except Exception:  # pragma: no cover - warm-up is best-effort
            logger.exception("Search warm-up failed (continuing).")

    def status(self) -> dict:
        """Index statistics and provenance."""
        stats = self.store.stats()
        return {
            "provider": self.config.provider,
            "model": self.provider.model_id,
            "embedding_dim": self.provider.dim,
            "llm_provider": self.config.llm_provider,
            "chat_model": (
                self.config.anthropic_model
                if self.config.llm_provider == "anthropic"
                else self.config.chat_model
            ),
            "llm_base_url": self.config.openai_base_url or "",
            "index_type": self.store.index_kind,
            "rerank": self.config.rerank,
            "rerank_model": self.config.rerank_model if self.config.rerank else "",
            "adaptive_fusion": self.config.adaptive_fusion,
            "graph_expansion": self.config.graph_expansion,
            "store_dir": str(self.config.store_dir),
            "watched_dir": str(self.config.watched_dir),
            "total_files": stats.total_files,
            "total_chunks": stats.total_chunks,
        }

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
