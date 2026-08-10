"""Shared AI helper utilities."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

try:  # Allow module import even when the OpenAI dependency is absent.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None  # type: ignore

__all__ = ["load_client"]


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
        init_kwargs["api_base"] = api_base
    return OpenAI(**init_kwargs)
