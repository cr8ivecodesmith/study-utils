"""Shared AI helper utilities."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
import requests

try:  # Allow module import even when the OpenAI dependency is absent.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None  # type: ignore

__all__ = ["load_client", "load_llama_swap_upstream_client"]


def load_client(local: bool = False, api_base: str | None = None) -> Any:
    """Initialize an OpenAI client using environment-derived credentials.

    Args:
        local: When True, prefer LOCAL_LLM_API_KEY over OPENAI_API_KEY.
        api_base: Optional API base URL to pass to the OpenAI client.

    Returns:
        An OpenAI client instance.

    Raises:
        RuntimeError: If the openai package is not available or no API key
            is found in the environment.
    """
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required to create a client. "
            "Install it and retry."
        )
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if local:
        api_key = os.getenv("LOCAL_LLM_API_KEY") or api_key
    if not api_key:
        if local:
            raise RuntimeError(
                "LOCAL_LLM_API_KEY not found in environment. "
                "Set it or add to .env"
            )
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. Set it or add to .env"
        )
    init_kwargs: dict[str, Any] = {"api_key": api_key}
    if api_base is not None:
        init_kwargs["base_url"] = api_base
    return OpenAI(**init_kwargs)


def load_llama_swap_upstream_client(
    model_id: str = "default",
    endpoint: str = "chat/completions",
    api_base: str = "http://localhost:8080",
    method: str = "POST",
    json: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Call llama-swap's upstream endpoint via requests.

    Args:
        model_id: The model identifier for the upstream URL.
        endpoint: The API endpoint path (e.g. ``chat/completions``).
        api_base: The base URL of the llama-swap instance.
        method: HTTP method to use for the request.
        json: JSON payload to send in the request body.
        files: Files to send as multipart form data.
        headers: Additional headers to merge with the Authorization header.

    Returns:
        A ``requests.Response`` instance.
    """
    load_dotenv()
    url = f"{api_base}/upstream/{model_id}/{endpoint}"

    api_key = os.getenv("LOCAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    auth_headers: dict[str, str] = {}
    if api_key:
        auth_headers["Authorization"] = f"Bearer {api_key}"

    merged_headers = auth_headers.copy()
    if headers:
        merged_headers.update(headers)

    return requests.request(
        method=method,
        url=url,
        json=json,
        files=files,
        headers=merged_headers if merged_headers is not None else None,
    )
