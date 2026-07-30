"""Debounced filesystem watcher that keeps the index live as files change.

Replaces the old ``monitor.py``. Two important differences: events are *debounced* (editors
fire several writes per save) and each flushed path is re-hashed by the indexer, so an
unchanged file costs nothing and a changed file is updated without duplicating vectors.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from reporag.chunking.languages import detect_language

if TYPE_CHECKING:
    from reporag.api import RepoRAG

logger = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    def __init__(
        self, pending: Set[str], lock: threading.Lock, all_text: bool = False
    ) -> None:
        self._pending = pending
        self._lock = lock
        self._all_text = all_text

    def _note(self, path: str) -> None:
        if path and detect_language(path, all_text=self._all_text):
            with self._lock:
                self._pending.add(path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note(str(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note(str(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note(str(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._note(str(event.src_path))
            self._note(str(getattr(event, "dest_path", "")))


def watch(
    cr: "RepoRAG",
    debounce: float = 0.5,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Keep ``cr``'s index in sync with its watched directory.

    Blocks until Ctrl-C, or until ``stop_event`` is set — which lets the watcher run on a
    background thread (e.g. inside the MCP server) and be shut down cleanly.
    """
    root = cr.config.watched_dir
    pending: Set[str] = set()
    lock = threading.Lock()
    handler = _Handler(pending, lock, all_text=cr.config.index_all_text)
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    logger.info("Watching %s for changes (Ctrl-C to stop)...", root)

    try:
        while stop_event is None or not stop_event.is_set():
            time.sleep(debounce)
            with lock:
                batch = set(pending)
                pending.clear()
            for raw in batch:
                _apply(cr, raw)
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
    finally:
        observer.stop()
        observer.join()


def _apply(cr: "RepoRAG", raw: str) -> None:
    path = Path(raw)
    try:
        if path.exists():
            stats = cr.index(path)
            if stats.files_indexed:
                logger.info("Reindexed %s (+%d chunks)", raw, stats.chunks_added)
        else:
            removed = cr.delete_path(path)
            if removed:
                logger.info("Removed %s (-%d chunks)", raw, removed)
    except Exception as exc:  # pragma: no cover - defensive, keep the loop alive
        logger.error("Failed to process %s: %s", raw, exc)
