"""Opt-in OpenAI (and OpenAI-compatible) embedding provider.

Unlike the old RepoRAG, this embeds each chunk independently (no file-level averaging) and
batches requests for throughput. The client and dimension are resolved lazily.

Setting ``base_url`` points the same client at a self-hosted / local OpenAI-compatible
server (Ollama, vLLM, LM Studio, LocalAI, text-embeddings-inference, …); such servers
usually don't require an API key, so one is optional when ``base_url`` is given.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, List, Optional, Sequence

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Known output dimensions; anything else is probed on first use.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_BATCH = 128


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._dim = _KNOWN_DIMS.get(model)

    @cached_property
    def _client(self) -> Any:
        from openai import OpenAI

        if not self._api_key and not self._base_url:
            raise RuntimeError(
                "OpenAI provider requires an API key (set OPENAI_API_KEY), or a "
                "self-hosted endpoint (set OPENAI_BASE_URL)."
            )
        # Local OpenAI-compatible servers often ignore the key but the SDK still
        # requires a non-empty string, so pass a harmless placeholder when absent.
        return OpenAI(api_key=self._api_key or "not-needed", base_url=self._base_url)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.embed_query("probe").shape[0]
        return self._dim

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _embed_batch(self, inputs: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self._model, input=inputs, timeout=30
        )
        return np.array([d.embedding for d in resp.data], dtype="float32")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim if self._dim else 1), dtype="float32")
        out: List[np.ndarray] = []
        for i in range(0, len(texts), _BATCH):
            out.append(self._embed_batch(list(texts[i : i + _BATCH])))
        return np.vstack(out).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]
