"""Local embedding provider backed by sentence-transformers (PyTorch).

Mirrors the FastEmbedProvider interface so it's a drop-in swap: same
``name`` / ``model_id`` / ``dim`` / ``embed_documents`` / ``embed_query``
contract, just backed by sentence-transformers instead of ONNX/fastembed.

The model is loaded lazily on first use so commands like ``reporag --help``
or ``status`` stay fast and never trigger a model download.
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformersProvider:
    name = "sentence-transformers"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
        *,
        device: str = "auto",
        batch_size: int = 64,
    ):
        self._model_name = model
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._device = None if device == "auto" else device
        self._batch_size = max(1, batch_size)

    @cached_property
    def _model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading sentence-transformers model %s (device=%s)…",
            self._model_name,
            self._device or "auto",
        )
        return SentenceTransformer(
            self._model_name,
            cache_folder=self._cache_dir,
            device=self._device,
        )

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype="float32")