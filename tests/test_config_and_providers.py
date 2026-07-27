"""P0 scaffolding tests: Config behaviour and the embedding provider abstraction."""

from __future__ import annotations

import numpy as np

from coderag.config import Config
from coderag.embeddings import EmbeddingProvider, get_provider


def test_config_defaults_and_derived_paths(tmp_path):
    cfg = Config(store_dir=tmp_path / ".coderag")
    assert cfg.provider == "fastembed"
    assert cfg.store_dir == tmp_path / ".coderag"


def test_store_dir_derives_from_watched_dir_not_cwd(tmp_path, monkeypatch):
    """The store lives under the *watched* dir, not the shell's cwd.

    Regression: a cwd-relative default silently pointed `status`/`search` at a different
    (empty) ``.coderag`` whenever a command ran from a directory other than the indexed one.
    """
    watched = tmp_path / "project"
    watched.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # stand somewhere other than the watched dir

    cfg = Config(watched_dir=watched)
    assert cfg.store_dir == watched / ".coderag"

    # Same result via from_env, and regardless of where the process is launched from.
    cfg_env = Config.from_env(watched_dir=watched)
    assert cfg_env.store_dir == watched / ".coderag"


def test_explicit_store_dir_overrides_derivation(tmp_path):
    watched = tmp_path / "project"
    explicit = tmp_path / "custom-store"
    cfg = Config(watched_dir=watched, store_dir=explicit)
    assert cfg.store_dir == explicit  # explicit wins; not <watched>/.coderag


def test_store_dir_from_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CODERAG_STORE_DIR", str(tmp_path / "env-store"))
    cfg = Config.from_env(watched_dir=tmp_path / "project")
    assert cfg.store_dir == tmp_path / "env-store"


def test_overriding_watched_dir_moves_derived_store(tmp_path):
    """Re-pointing watched_dir on an auto-derived config moves the store with it."""
    cfg = Config(watched_dir=tmp_path / "a")
    assert cfg.store_dir == tmp_path / "a" / ".coderag"
    moved = cfg.with_overrides(watched_dir=tmp_path / "b")
    assert moved.store_dir == tmp_path / "b" / ".coderag"
    # But an explicit store_dir is sticky across a watched_dir move.
    pinned = Config(watched_dir=tmp_path / "a", store_dir=tmp_path / "store")
    assert (
        pinned.with_overrides(watched_dir=tmp_path / "b").store_dir
        == tmp_path / "store"
    )


def test_config_is_immutable_and_copies():
    cfg = Config()
    updated = cfg.with_overrides(top_k=42)
    assert updated.top_k == 42
    assert cfg.top_k == 8  # original untouched


def test_from_env_reads_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CODERAG_PROVIDER", "fake")
    monkeypatch.setenv("CODERAG_TOP_K", "3")
    cfg = Config.from_env(store_dir=tmp_path)
    assert cfg.provider == "fake"
    assert cfg.top_k == 3
    assert cfg.store_dir == tmp_path  # explicit override wins


def test_from_env_ignores_bad_ints(monkeypatch):
    monkeypatch.setenv("CODERAG_TOP_K", "not-a-number")
    cfg = Config.from_env()
    assert cfg.top_k == 8  # falls back to default


def test_env_ignore_globs_append_to_defaults(monkeypatch):
    from coderag.config import DEFAULT_IGNORE_GLOBS

    monkeypatch.setenv("CODERAG_IGNORE_GLOBS", "secret/*, *.bin")
    cfg = Config.from_env()
    assert set(DEFAULT_IGNORE_GLOBS) <= set(cfg.ignore_globs)  # defaults kept
    assert "secret/*" in cfg.ignore_globs and "*.bin" in cfg.ignore_globs


def test_default_ignores_cover_dependency_and_cache_dirs():
    from coderag.config import DEFAULT_IGNORE_GLOBS

    for junk in ("site-packages/*", ".cache/*", "node_modules/*", "target/*"):
        assert junk in DEFAULT_IGNORE_GLOBS


def test_env_embed_device_and_threads(monkeypatch):
    monkeypatch.setenv("CODERAG_EMBED_DEVICE", "cuda")
    monkeypatch.setenv("CODERAG_EMBED_THREADS", "8")
    cfg = Config.from_env()
    assert cfg.embed_device == "cuda"
    assert cfg.embed_threads == 8


def test_secrets_are_kept_out_of_repr():
    cfg = Config(
        openai_api_key="sk-openai-secret",
        anthropic_api_key="sk-anthropic-secret",
        api_key="server-secret",
    )
    text = repr(cfg)
    assert "sk-openai-secret" not in text
    assert "sk-anthropic-secret" not in text
    assert "server-secret" not in text


def test_cors_origins_parsed_from_env(monkeypatch):
    monkeypatch.setenv("CODERAG_CORS_ORIGINS", "https://a.example, https://b.example")
    cfg = Config.from_env()
    assert cfg.cors_origins == ("https://a.example", "https://b.example")


def test_demo_mode_defaults_off_and_parses_from_env(monkeypatch):
    assert Config().demo_mode is False
    monkeypatch.setenv("CODERAG_DEMO_MODE", "true")
    monkeypatch.setenv("CODERAG_DEMO_MAX_ANSWERS", "3")
    monkeypatch.setenv("CODERAG_DEMO_COOLDOWN_SECONDS", "15")
    cfg = Config.from_env()
    assert cfg.demo_mode is True
    assert cfg.demo_max_answers == 3
    assert cfg.demo_cooldown_seconds == 15
    # Non-truthy strings keep it off.
    monkeypatch.setenv("CODERAG_DEMO_MODE", "off")
    assert Config.from_env().demo_mode is False


def test_fake_provider_conforms_to_protocol():
    provider = get_provider(Config(provider="fake"))
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dim == 16


def test_fake_provider_is_deterministic_and_normalized():
    provider = get_provider(Config(provider="fake"))
    a = provider.embed_documents(["def foo(): pass", "class Bar: ..."])
    b = provider.embed_documents(["def foo(): pass", "class Bar: ..."])
    assert a.shape == (2, provider.dim)
    assert a.dtype == np.dtype("float32")
    np.testing.assert_array_equal(a, b)  # deterministic
    norms = np.linalg.norm(a, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)  # unit vectors


def test_fake_provider_query_matches_identical_document():
    provider = get_provider(Config(provider="fake"))
    q = provider.embed_query("hello world")
    d = provider.embed_documents(["hello world"])[0]
    np.testing.assert_allclose(q, d)


def test_empty_documents_returns_empty_array():
    provider = get_provider(Config(provider="fake"))
    out = provider.embed_documents([])
    assert out.shape == (0, provider.dim)


def test_unknown_provider_raises():
    import pytest

    with pytest.raises(ValueError):
        get_provider(Config(provider="bogus"))
