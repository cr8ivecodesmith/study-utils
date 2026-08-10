from __future__ import annotations

from unittest.mock import patch

import pytest

from study_utils.core import ai
from study_utils.core.ai import load_client, load_llama_swap_upstream_client


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
        getattr(openai_factory.last, "init_kwargs", {}).get("base_url")
        == "http://example.com/v1"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_client_api_base_none_omitted(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    monkeypatch.setattr(ai, "OpenAI", ai.OpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _client = load_client(api_base=None)
    assert "base_url" not in getattr(openai_factory.last, "init_kwargs", {})
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


def test_upstream_client_default_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client()
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["url"] == "http://localhost:8080/upstream/default/chat/completions"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_upstream_client_custom_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client(model_id="llama-3.1-8b")
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        url = call_args.kwargs["url"]
        assert url == "http://localhost:8080/upstream/llama-3.1-8b/chat/completions"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_upstream_client_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client(endpoint="completions")
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["url"] == "http://localhost:8080/upstream/default/completions"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_upstream_client_method_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client(method="GET")
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["method"] == "GET"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_upstream_client_json_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    json_payload = {"messages": [{"role": "user", "content": "hello"}]}
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client(json=json_payload)
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["json"] == json_payload
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_upstream_client_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    custom_headers = {"X-Custom": "value"}
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client(headers=custom_headers)
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        auth_header = call_args.kwargs["headers"]["Authorization"]
        custom_header = call_args.kwargs["headers"]["X-Custom"]
        assert auth_header == "Bearer test-key"
        assert custom_header == "value"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_upstream_client_local_key_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "local-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client()
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        auth_header = call_args.kwargs["headers"]["Authorization"]
        assert auth_header == "Bearer local-key"
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)


def test_upstream_client_no_key_no_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("study_utils.core.ai.requests.request") as mock_request:
        load_llama_swap_upstream_client()
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args.kwargs["headers"] == {}
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_all_exports() -> None:
    assert "load_client" in ai.__all__
    assert "load_llama_swap_upstream_client" in ai.__all__
