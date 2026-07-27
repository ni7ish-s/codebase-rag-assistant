"""Local-first embedding provider backed by fastembed (ONNX, no torch, no API key).

This is the default backend. The model is loaded lazily on first use so that
``coderag --help``, ``status``, and any code path that doesn't actually embed stays fast
and never triggers a model download.
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedProvider:
    name = "fastembed"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
        *,
        device: str = "auto",
        threads: int = 0,
        batch_size: int = 64,
    ):
        self._model_name = model
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._device = device
        self._threads = threads
        self._batch_size = max(1, batch_size)
        self._dim = self._lookup_dim(model)

    @staticmethod
    def _lookup_dim(model: str) -> Optional[int]:
        try:
            from fastembed import TextEmbedding

            for entry in TextEmbedding.list_supported_models():
                if entry.get("model") == model:
                    return int(entry["dim"])
        except Exception:  # pragma: no cover - metadata lookup best-effort
            pass
        return None

    def _providers(self) -> Optional[list[str]]:
        """ONNX execution providers for the chosen device, or None for the library default.

        CUDA is listed with a CPU fallback so onnxruntime degrades gracefully at runtime;
        ``auto`` only requests CUDA when onnxruntime actually exposes it.
        """
        if self._device == "cpu":
            return ["CPUExecutionProvider"]
        if self._device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:  # auto: prefer a GPU only if one is really available
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:  # pragma: no cover - onnxruntime optional / probe best-effort
            pass
        return None

    @cached_property
    def _model(self) -> Any:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"cache_dir": self._cache_dir}
        providers = self._providers()
        if providers is not None:
            kwargs["providers"] = providers
        if self._threads > 0:
            kwargs["threads"] = self._threads
        logger.info(
            "Loading fastembed model %s (device=%s)…", self._model_name, self._device
        )
        try:
            return TextEmbedding(self._model_name, **kwargs)
        except (
            Exception
        ) as exc:  # pragma: no cover - GPU init can fail on broken drivers
            if providers and providers[0] != "CPUExecutionProvider":
                logger.warning(
                    "GPU embedding init failed (%s); falling back to CPU.", exc
                )
                kwargs.pop("providers", None)
                return TextEmbedding(self._model_name, **kwargs)
            raise

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Fall back to probing the loaded model.
            self._dim = int(self._model.embedding_size)
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = list(self._model.passage_embed(list(texts), batch_size=self._batch_size))
        return np.vstack(vecs).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.query_embed([text])))
        return np.asarray(vec, dtype="float32")
