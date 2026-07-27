"""Incremental indexing orchestration.

Ties chunking -> embedding -> the LanceDB store together with content-hash change detection.
The critical correctness property: a changed file's *old* chunks are removed from the store
**before** the new ones are added (``write_file(..., replace=True)``), so re-saving a file
never accumulates duplicate or stale rows.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from coderag._ignore import ignore_dir_names, is_ignored, walk_files
from coderag.chunking import chunk_file
from coderag.chunking.languages import detect_language
from coderag.config import Config
from coderag.embeddings import EmbeddingProvider
from coderag.types import Chunk, IndexProgress, IndexStats

if TYPE_CHECKING:
    from coderag.store.lance_store import LanceStore

logger = logging.getLogger(__name__)

# During a long initial index, commit buffered rows at least this often so dense search can
# return partial results before the steady-state 8192-chunk flush boundary. Only applied when
# a live progress object is supplied (the MCP background index), so the CLI/watcher keep their
# single-flush-at-end batching.
_PARTIAL_FLUSH_SECS = 5.0


class _ProgressReporter:
    """Live, human-facing indexing progress, written to stderr (stdout stays clean).

    A large index is otherwise a silent wait — the very problem behind an agent sitting at
    "Working… 10 min" while an over-broad root is crawled. This narrates *both* phases: the
    discovery walk (which hashes every candidate before a single chunk is embedded) and the
    embedding pass. On a TTY it redraws one line in place; otherwise (agent terminals,
    captured logs) it prints throttled newline updates so output stays readable. It is a
    no-op unless ``enabled`` — the library facade and the MCP background index pass
    ``progress=False`` and stay quiet.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._next = 0.0  # monotonic time of the next allowed (unforced) update

    def update(self, msg: str, *, force: bool = False) -> None:
        """Show ``msg``, throttled so per-file calls don't flood the terminal/logs."""
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now < self._next:
            return
        self._next = now + (0.1 if self._tty else 2.0)
        sys.stderr.write(f"\r\x1b[2K{msg}" if self._tty else msg + "\n")
        sys.stderr.flush()

    def done(self, msg: str) -> None:
        """Emit a final line and stop redrawing (clears the in-place line on a TTY)."""
        if not self.enabled:
            return
        sys.stderr.write(f"\r\x1b[2K{msg}\n" if self._tty else msg + "\n")
        sys.stderr.flush()


@dataclass(slots=True)
class _Work:
    rel: str
    language: str
    text: str
    content_hash: str
    mtime: float
    size: int
    existed: bool  # whether the file already had rows (→ replace, delete-before-add)


class Indexer:
    def __init__(
        self,
        config: Config,
        provider: EmbeddingProvider,
        store: "LanceStore",
    ) -> None:
        self.config = config
        self.provider = provider
        self.store = store
        self._ignore_dirs = ignore_dir_names(config.ignore_globs)

    # --- public ---

    def index(
        self,
        target: Optional[Path] = None,
        *,
        full: bool = False,
        progress: bool = False,
        live: Optional[IndexProgress] = None,
    ) -> IndexStats:
        """Index ``target`` incrementally.

        ``progress`` enables the human-facing stderr narration; ``live`` is an optional
        machine-readable :class:`IndexProgress` the caller can poll concurrently (used by the
        MCP server's background index so ``index_status`` reflects live state). Both default
        off, so existing callers (CLI/watcher/tests) are unaffected.
        """
        root = self.config.watched_dir.resolve()
        target = (target or self.config.watched_dir).resolve()
        prune = target == root  # only a full-root pass removes vanished files
        rep = _ProgressReporter(progress)
        if live is not None:
            live.begin("scanning")

        stats = IndexStats()
        if full:
            self.store.clear()

        # 1. Discover candidates and detect what actually changed (cheap stat/hash check).
        #    Preload all file metadata once (one scan) so discovery does no per-file query.
        metas = self.store.all_file_metas()
        rep.update(f"Scanning {target} for files to index…", force=True)
        walked: set[str] = set()
        work: List[_Work] = []
        for abs_path, rel, language in self._walk(target, root):
            walked.add(rel)
            item = self._maybe_work(abs_path, rel, language, metas)
            if item is None:
                stats.files_skipped += 1
            else:
                work.append(item)
            if live is not None:
                live.saw_file(len(walked), len(work))
            rep.update(
                f"Scanning {target} — {len(walked)} file(s) seen, "
                f"{len(work)} to index, {stats.files_skipped} unchanged/skipped…"
            )
        if work:
            rep.update(
                f"Embedding {len(work)} changed file(s) "
                f"({stats.files_skipped} unchanged/skipped)…",
                force=True,
            )
        else:
            rep.update(
                f"Up to date — {stats.files_skipped} file(s) unchanged.", force=True
            )

        # 2. (Re)index changed files. Chunking + embedding (the CPU/network cost) may run
        #    in parallel across files (config.index_workers); the store writes stay on this
        #    single thread to preserve the delete-before-add invariant and single writer.
        if live is not None and work:
            live.set_state("indexing")
        last_flush = time.monotonic()
        for item, added, removed in self._embed_and_write(work, reporter=rep):
            stats.chunks_added += added
            stats.chunks_removed += removed
            stats.files_indexed += 1
            if live is not None:
                live.wrote_file(item.rel, added)
                # Commit periodically so dense search picks up partials during a long initial
                # index, instead of waiting for the 8192-chunk boundary or the final persist.
                if time.monotonic() - last_flush > _PARTIAL_FLUSH_SECS:
                    self.store.flush()
                    last_flush = time.monotonic()

        # 3. Prune files that disappeared from disk (full-root passes only).
        if prune:
            for rel in set(self.store.all_file_paths()) - walked:
                removed = self.store.delete_file(rel)
                stats.files_removed += 1
                stats.chunks_removed += removed

        # 4. Persist. A full pass that changed something rebuilds the FTS/vector indexes
        #    and compacts. An incremental/single-file pass skips the compaction but still
        #    asks the store to refresh the ANN/FTS indexes when the unindexed tail has
        #    grown enough to drag query latency down (``maybe_reindex`` is a cheap no-op
        #    otherwise) — so a watcher edit never triggers a full rebuild, but the tail of
        #    brute-forced rows also can't grow unbounded and silently degrade retrieval.
        changed = stats.files_indexed > 0 or stats.files_removed > 0
        if prune and changed:
            if live is not None:
                live.set_state("optimizing")
            self.store.optimize()
        elif changed:
            self.store.maybe_reindex()
        else:
            self.store.flush()

        final = self.store.stats()
        stats.total_files = final.total_files
        stats.total_chunks = final.total_chunks
        rep.done(
            f"✓ Indexed {stats.files_indexed} file(s) — "
            f"{stats.total_files} total / {stats.total_chunks} chunks."
        )
        if live is not None:
            live.finish("ready")
        return stats

    # --- internals ---

    def _maybe_work(
        self,
        abs_path: Path,
        rel: str,
        language: str,
        metas: Dict[str, Dict[str, Any]],
    ) -> Optional[_Work]:
        existing = metas.get(rel)
        try:
            st = abs_path.stat()
        except OSError as exc:
            logger.warning("Cannot stat %s: %s", abs_path, exc)
            return None
        # Cheap fast-path: if size and mtime are unchanged, skip the read+hash entirely.
        # The hash stays the authority on "did content change" — this only avoids the read
        # for the common untouched case (the dominant cost of re-indexing a large tree).
        if (
            existing is not None
            and existing.get("size") is not None
            and int(existing["size"]) == st.st_size
            and abs(float(existing.get("mtime") or 0.0) - st.st_mtime) < 1e-6
        ):
            return None
        try:
            data = abs_path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", abs_path, exc)
            return None
        if len(data) > self.config.max_file_bytes or not data.strip():
            return None
        if b"\x00" in data[:8192]:
            return None  # binary file (NUL byte in the head) — never index as text
        content_hash = hashlib.sha256(data).hexdigest()
        if existing is not None and existing.get("content_hash") == content_hash:
            return None  # content unchanged (e.g. touched) -> no embedding cost
        text = data.decode("utf-8", errors="replace")
        return _Work(
            rel,
            language,
            text,
            content_hash,
            st.st_mtime,
            st.st_size,
            existing is not None,
        )

    def _embed_and_write(
        self, work: List[_Work], *, reporter: _ProgressReporter
    ) -> Iterator[Tuple[_Work, int, int]]:
        """Chunk+embed each file (optionally across worker threads) and apply the writes.

        Embedding is the expensive, parallelizable step and touches no shared mutable
        state, so it runs in a thread pool when ``index_workers > 1``. The store writes are
        drained here on the single calling thread, so the no-duplicate (delete-before-add)
        invariant and the single-writer store are preserved.

        Yields ``(item, chunks_added, chunks_removed)`` per file — the ``_Work`` item is
        surfaced so the caller can report the current path (the worker pool completes out of
        order, so positional zipping back to ``work`` is not possible).
        """
        if not work:
            return
        workers = max(1, self.config.index_workers)
        total = len(work)
        done = 0
        if workers > 1 and len(work) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._prepare, item): item for item in work}
                for fut in as_completed(futures):
                    item = futures[fut]
                    chunks, vectors = fut.result()
                    added, removed = self._write(item, chunks, vectors)
                    yield item, added, removed
                    done += 1
                    reporter.update(f"Embedding {done}/{total} file(s)…")
        else:
            for item in work:
                chunks, vectors = self._prepare(item)
                added, removed = self._write(item, chunks, vectors)
                yield item, added, removed
                done += 1
                reporter.update(f"Embedding {done}/{total} file(s)…")

    def _prepare(self, item: _Work) -> Tuple[List[Chunk], Optional[np.ndarray]]:
        """Chunk and embed a file. Pure with respect to the store, so it is safe to run in
        a worker thread; the resulting writes are applied by :meth:`_write`."""
        chunks = chunk_file(item.text, item.language, self.config)
        if not chunks:
            return [], None
        vectors = self.provider.embed_documents([c.text for c in chunks])
        return chunks, vectors

    def _write(
        self, item: _Work, chunks: List[Chunk], vectors: Optional[np.ndarray]
    ) -> Tuple[int, int]:
        """Apply a prepared file to the store (delete-before-add for a replacement).

        Must run single-threaded — it is the only writer.
        """
        return self.store.write_file(
            item.rel,
            item.language,
            item.content_hash,
            item.mtime,
            item.size,
            chunks,
            vectors,
            replace=item.existed,
        )

    def _walk(self, target: Path, root: Path) -> Iterator[Tuple[Path, str, str]]:
        if target.is_file():
            rel = self._rel(target, root)
            language = detect_language(target, all_text=self.config.index_all_text)
            if rel and language and not self._ignored(rel):
                yield target, rel, language
            return

        # walk_files owns dir-pruning + ignore-glob + .gitignore matching, shared with
        # fs_search so semantic and exact search see exactly the same files.
        for abs_path, rel in walk_files(
            target,
            self.config.ignore_globs,
            root=root,
            use_gitignore=self.config.use_gitignore,
        ):
            language = detect_language(
                abs_path.name, all_text=self.config.index_all_text
            )
            if language:
                yield abs_path, rel, language

    @staticmethod
    def _rel(abs_path: Path, root: Path) -> Optional[str]:
        try:
            return abs_path.resolve().relative_to(root).as_posix()
        except ValueError:
            return None

    def _ignored(self, rel: str) -> bool:
        return is_ignored(rel, self.config.ignore_globs, self._ignore_dirs)
