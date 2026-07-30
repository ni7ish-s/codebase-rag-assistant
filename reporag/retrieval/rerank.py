"""Optional second-stage reranking for two-stage retrieve-then-rerank search.

First-stage hybrid retrieval (dense + BM25 + RRF) is tuned for *recall* — get the right
chunks into a candidate pool cheaply. A cross-encoder reranker then scores each candidate
*jointly* with the query (not via independent embeddings), which is far more precise at the
top of the list.

It's **opt-in** (``config.rerank``). Backed by sentence-transformers' ``CrossEncoder``
(PyTorch), matching the same backend used for the embedding provider.
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, runtime_checkable

from reporag.config import Config

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@runtime_checkable
class Reranker(Protocol):
    """Scores how well each document answers the query (higher = more relevant)."""

    @property
    def model_id(self) -> str:
        """Identifier of the underlying rerank model (used in status/logging)."""

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        """Return one relevance score per document, aligned to input order."""


class CrossEncoderReranker:
    """Local cross-encoder reranker backed by sentence-transformers' ``CrossEncoder``."""

    name = "cross-encoder"

    def __init__(
        self, model: str = DEFAULT_RERANK_MODEL, cache_dir: Optional[Path] = None
    ) -> None:
        self._model_name = model
        self._cache_dir = str(cache_dir) if cache_dir else None

    @cached_property
    def _encoder(self) -> Any:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker %s ...", self._model_name)
        return CrossEncoder(self._model_name, cache_folder=self._cache_dir)

    @property
    def model_id(self) -> str:
        return self._model_name

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._encoder.predict(pairs)
        return [float(s) for s in scores]


def get_reranker(config: Config) -> Optional[Reranker]:
    """Build the reranker if ``config.rerank`` is on, else ``None`` (reranking disabled)."""
    if not config.rerank:
        return None
    return CrossEncoderReranker(config.rerank_model, cache_dir=config.cache_dir)


__all__ = [
    "CrossEncoderReranker",
    "DEFAULT_RERANK_MODEL",
    "Reranker",
    "get_reranker",
]