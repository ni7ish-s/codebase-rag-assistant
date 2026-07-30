"""Device/provider selection for the fastembed backend (no model load, no network)."""

from __future__ import annotations

from reporag.embeddings.fastembed_provider import FastEmbedProvider


def _provider(device: str) -> FastEmbedProvider:
    return FastEmbedProvider("BAAI/bge-small-en-v1.5", device=device)


def test_cpu_device_forces_cpu_provider():
    assert _provider("cpu")._providers() == ["CPUExecutionProvider"]


def test_cuda_device_lists_cuda_then_cpu_fallback():
    # CPU is listed second so onnxruntime degrades gracefully if CUDA init fails.
    assert _provider("cuda")._providers() == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_auto_uses_cpu_when_no_gpu(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    assert _provider("auto")._providers() is None  # library CPU default


def test_auto_uses_gpu_when_available(monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert _provider("auto")._providers() == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
