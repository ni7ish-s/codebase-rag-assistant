"""Tests for the multi-platform LLM answer surface.

These never hit the network: the OpenAI / Anthropic SDKs are stubbed, so we only verify
backend selection, credential handling, and token streaming.
"""

from __future__ import annotations

import sys
import types

import pytest

from reporag import llm
from reporag.config import Config
from reporag.types import SearchHit


def _hit() -> SearchHit:
    return SearchHit(
        chunk_id=1,
        path="coderag/api.py",
        symbol="CodeRAG.search",
        kind="method",
        language="python",
        start_line=10,
        end_line=20,
        text="def search(self, q): ...",
        score=1.0,
        similarity=0.9,
    )


class _StubCodeRAG:
    """Minimal stand-in for CodeRAG: just config + search()."""

    def __init__(self, config: Config, hits):
        self.config = config
        self._hits = hits

    def search(self, query, top_k):
        return self._hits


# --- config wiring ---


def test_from_env_reads_model_platform_settings(monkeypatch):
    monkeypatch.setenv("CODERAG_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("CODERAG_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CODERAG_ANSWER_MAX_TOKENS", "2048")
    cfg = Config.from_env()
    assert cfg.llm_provider == "anthropic"
    assert cfg.anthropic_api_key == "sk-ant-test"
    assert cfg.anthropic_model == "claude-sonnet-4-6"
    assert cfg.openai_base_url == "http://localhost:11434/v1"
    assert cfg.answer_max_tokens == 2048


def test_llm_provider_defaults_to_openai():
    assert Config().llm_provider == "openai"
    assert Config().anthropic_model == "claude-opus-4-8"


# --- dispatch and credential handling ---


def test_no_hits_yields_message():
    cr = _StubCodeRAG(Config(), hits=[])
    assert "".join(llm.stream_answer(cr, "q")) == (
        "No relevant code was found in the index for that query."
    )


def test_unknown_provider_raises():
    cr = _StubCodeRAG(Config(llm_provider="bogus"), hits=[_hit()])
    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        list(llm.stream_answer(cr, "q"))


def test_openai_without_key_or_base_url_raises():
    cr = _StubCodeRAG(Config(llm_provider="openai"), hits=[_hit()])
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        list(llm.stream_answer(cr, "q"))


def test_anthropic_without_key_raises():
    cr = _StubCodeRAG(Config(llm_provider="anthropic"), hits=[_hit()])
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        list(llm.stream_answer(cr, "q"))


# --- streaming, with stubbed SDKs ---


def test_openai_streams_tokens(monkeypatch):
    captured = {}

    def _delta(text):
        choice = types.SimpleNamespace(delta=types.SimpleNamespace(content=text))
        return types.SimpleNamespace(choices=[choice])

    class _Completions:
        def create(self, model, messages, temperature, stream):
            captured["model"] = model
            return iter([_delta("hel"), _delta("lo"), _delta(None)])

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

        chat = _Chat()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))

    cfg = Config(
        llm_provider="openai",
        openai_base_url="http://localhost:11434/v1",
        chat_model="llama3",
    )
    cr = _StubCodeRAG(cfg, hits=[_hit()])
    assert "".join(llm.stream_answer(cr, "what does search do?")) == "hello"
    assert captured["model"] == "llama3"
    assert captured["base_url"] == "http://localhost:11434/v1"
    # A local server needs no real key; a placeholder is supplied.
    assert captured["api_key"] == "not-needed"


def test_anthropic_streams_tokens(monkeypatch):
    captured = {}

    class _Stream:
        text_stream = iter(["Cite ", "`api.py:10`"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Messages:
        def stream(self, model, max_tokens, system, messages):
            captured.update(model=model, max_tokens=max_tokens, system=system)
            return _Stream()

    class _Anthropic:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = _Messages()

    monkeypatch.setitem(
        sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic)
    )

    cfg = Config(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test",
        anthropic_model="claude-opus-4-8",
        answer_max_tokens=512,
    )
    cr = _StubCodeRAG(cfg, hits=[_hit()])
    assert "".join(llm.stream_answer(cr, "how to cite?")) == "Cite `api.py:10`"
    assert captured["model"] == "claude-opus-4-8"
    assert captured["max_tokens"] == 512
    assert captured["api_key"] == "sk-ant-test"
    assert captured["system"] == llm.SYSTEM_PROMPT
