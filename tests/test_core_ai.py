from __future__ import annotations


import pytest

from study_utils.core import ai
from study_utils.core.ai import load_client


def test_load_client_requires_openai_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai, "OpenAI", None)
    with pytest.raises(RuntimeError) as exc:
        load_client()
    assert "openai" in str(exc.value).lower()


def test_load_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        load_client()
    assert "OPENAI_API_KEY" in str(exc.value)


def test_load_client_returns_stub_with_key(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = load_client()
    assert client is openai_factory.last
    assert openai_factory.last is not None
    assert (
        getattr(openai_factory.last, "init_kwargs", {}).get("api_key")
        == "test-key"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_client_local_true_uses_local_key(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-key")
    _client = load_client(local=True)
    assert (
        getattr(openai_factory.last, "init_kwargs", {}).get("api_key")
        == "local-key"
    )
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)


def test_load_client_local_true_precedes_openai_key(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    _client = load_client(local=True)
    assert (
        getattr(openai_factory.last, "init_kwargs", {}).get("api_key")
        == "local-key"
    )
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_client_local_true_no_key_raises(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        load_client(local=True)
    assert "LOCAL_LLM_API_KEY" in str(exc.value)


def test_load_client_api_base_passthrough(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _client = load_client(api_base="http://example.com/v1")
    assert (
        getattr(openai_factory.last, "init_kwargs", {}).get("api_base")
        == "http://example.com/v1"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_client_api_base_none_omitted(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _client = load_client(api_base=None)
    assert "api_base" not in getattr(openai_factory.last, "init_kwargs", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_client_backward_compat(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = load_client()
    assert client is openai_factory.last
    assert openai_factory.last is not None
    assert (
        getattr(openai_factory.last, "init_kwargs", {}).get("api_key")
        == "test-key"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
