"""Tests for the recommended-embedding-model registry (offline, no downloads)."""

from __future__ import annotations

from coderag.embeddings.models import (
    RECOMMENDED,
    RECOMMENDED_RERANKERS,
    format_models,
)


def test_registry_is_nonempty_and_well_formed():
    assert RECOMMENDED
    for m in RECOMMENDED:
        assert m.name and "/" in m.name  # looks like a HF model id
        assert m.dim > 0
        assert m.size_gb > 0
        assert m.note


def test_default_model_is_listed():
    # The current default must appear so users can see its trade-off.
    assert any(m.name == "BAAI/bge-small-en-v1.5" for m in RECOMMENDED)


def test_reranker_registry_well_formed():
    assert RECOMMENDED_RERANKERS
    for r in RECOMMENDED_RERANKERS:
        assert r.name and "/" in r.name
        assert r.size_gb > 0 and r.note
    # The default reranker model must be listed.
    assert any(
        r.name == "Xenova/ms-marco-MiniLM-L-12-v2" for r in RECOMMENDED_RERANKERS
    )


def test_format_models_renders_table():
    out = format_models()
    assert "model" in out and "code?" in out
    assert "jina-embeddings-v2-base-code" in out
    assert "Rerankers" in out and "bge-reranker-base" in out
