"""Shared data types used across chunking, storage, retrieval, and the public API."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # Literal lives in typing on 3.8+, but keep the import defensive.
    from typing import Literal

    IndexState = Literal[
        "idle", "scanning", "indexing", "optimizing", "ready", "failed"
    ]
except ImportError:  # pragma: no cover - very old runtimes
    IndexState = str  # type: ignore[misc]


@dataclass(slots=True)
class Chunk:
    """A unit of indexed code — usually a function/class/method, or a line window."""

    text: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    language: str
    symbol: Optional[str] = None  # qualified name, e.g. "ClassName.method"
    kind: str = "window"  # "function" | "class" | "method" | "window"

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass(slots=True)
class SearchHit:
    """A retrieval result, hydrated from the store."""

    chunk_id: int
    path: str
    symbol: Optional[str]
    kind: str
    language: str
    start_line: int
    end_line: int
    text: str
    score: float  # fused (RRF) score — relative ranking signal
    similarity: float  # raw cosine similarity in [0, 1] for display

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "score": self.score,
            "similarity": self.similarity,
            "location": self.location,
        }


@dataclass
class IndexStats:
    """Summary of an indexing run or the current index state."""

    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    total_files: int = 0
    total_chunks: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "files_removed": self.files_removed,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
        }


class IndexProgress:
    """Thread-safe live snapshot of an in-flight index run.

    Unlike :class:`IndexStats` (a plain value object returned once a run finishes), this is a
    *mutable, shared* object: the indexer thread updates it as it works while a different
    thread — the MCP ``index_status`` tool — polls it. That is what makes a long index
    legible: ``total_files``/``total_chunks`` read from the store stay 0 until rows are
    committed, but ``files_discovered`` / ``files_indexed`` here tick up live, so an agent can
    tell the difference between "hung" and "scanning a big tree".

    A single lock guards every field so ``snapshot()`` returns a consistent set rather than a
    torn mix of old and new values. Writer helpers are only called from the index thread;
    ``snapshot()`` is safe to call from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: IndexState = "idle"
        self.files_discovered = 0  # candidates walked so far (phase 1)
        self.files_to_index = 0  # changed files queued for embedding
        self.files_indexed = 0  # files embedded + written so far (phase 2)
        self.chunks_added = 0  # chunks written so far
        self.current_path: Optional[str] = None
        self.started_at: Optional[float] = None  # time.time() epoch
        self.finished_at: Optional[float] = None
        self.last_error: Optional[str] = None

    # --- writer-side helpers (called only from the index thread) ---

    def begin(self, state: IndexState = "scanning") -> None:
        with self._lock:
            self.state = state
            self.started_at = time.time()
            self.finished_at = None
            self.last_error = None
            self.files_discovered = self.files_to_index = 0
            self.files_indexed = self.chunks_added = 0
            self.current_path = None

    def set_state(self, state: IndexState) -> None:
        with self._lock:
            self.state = state

    def saw_file(self, n_discovered: int, n_to_index: int) -> None:
        with self._lock:
            self.files_discovered = n_discovered
            self.files_to_index = n_to_index

    def wrote_file(self, path: str, chunks_added: int) -> None:
        with self._lock:
            self.files_indexed += 1
            self.chunks_added += chunks_added
            self.current_path = path

    def finish(self, state: IndexState, error: Optional[str] = None) -> None:
        with self._lock:
            self.state = state
            self.finished_at = time.time()
            self.last_error = error
            self.current_path = None

    # --- reader-side ---

    def snapshot(self) -> Dict[str, Any]:
        """A consistent, JSON-friendly view of the current progress."""
        with self._lock:
            elapsed = None
            if self.started_at is not None:
                end = self.finished_at or time.time()
                elapsed = round(end - self.started_at, 2)
            return {
                "state": self.state,
                "files_discovered": self.files_discovered,
                "files_to_index": self.files_to_index,
                "files_indexed": self.files_indexed,
                "chunks": self.chunks_added,
                "current_path": self.current_path,
                "started_at": self.started_at,
                "elapsed": elapsed,
                "last_error": self.last_error,
            }
