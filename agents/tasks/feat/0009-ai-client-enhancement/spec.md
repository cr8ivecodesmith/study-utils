# AI Client Enhancement — Spec

## Summary

Extend `src/study_utils/core/ai.py` to support local LLMs via `LOCAL_LLM_API_KEY` and add an `api_base` parameter to `load_client()`. Introduce `load_llama_swap_upstream_client()` for llama-swap's upstream endpoint.

## Behavior

- Given `load_client()` is called with no arguments, when no `OPENAI_API_KEY` is set, then it raises `RuntimeError` as before (backward compatible).
- Given `load_client(local=True)` is called, when `LOCAL_LLM_API_KEY` is set (and `OPENAI_API_KEY` is not), then the client is created with `LOCAL_LLM_API_KEY`.
- Given both `OPENAI_API_KEY` and `LOCAL_LLM_API_KEY` are set, when `load_client(local=True)` is called, then `LOCAL_LLM_API_KEY` takes precedence.
- Given `load_client(local=True)` is called with neither key set, then it raises `RuntimeError` mentioning `LOCAL_LLM_API_KEY`.
- Given `load_client(api_base="http://localhost:8080/v1")` is called, then the client is created with the provided `api_base`.
- Given `load_client(local=True, api_base="http://localhost:8080/v1")` is called, then the client uses `LOCAL_LLM_API_KEY` and the provided `api_base`.
- Given `load_client(api_base=None)` is called, then no `api_base` is passed to `OpenAI()` (uses default endpoint).
- Given `load_llama_swap_upstream_client()` is called with no arguments, it returns a `requests.Response` with `api_base="http://localhost:8080"` (llama-swap base, no `/v1` suffix since upstream handles it).
- Given `load_llama_swap_upstream_client()` is called with no API key set, then the response has no `Authorization` header.
- Given `load_llama_swap_upstream_client(endpoint="completions")` is called, the URL is `{api_base}/upstream/{model_id}/completions`.
- Given `load_llama_swap_upstream_client(method="GET")` is called, it performs a GET request instead of POST.

## Design

### Changes to `load_client()`

Add two optional parameters: `local: bool = False` and `api_base: str | None = None`.

**Key resolution precedence:**
1. `LOCAL_LLM_API_KEY` (when `local=True` and set)
2. `OPENAI_API_KEY` (always available)
3. Raise `RuntimeError` if neither is set

**`api_base` behavior:**
- When `api_base` is `None`, it is not passed to `OpenAI()` (uses default endpoint).
- When `api_base` is a string, it is passed as `api_base=...` to `OpenAI()`.

### New function: `load_llama_swap_upstream_client()`

Uses `requests` to call llama-swap's upstream endpoint. Returns a `requests.Response`.

- Accepts `model_id: str = "default"`, `endpoint: str = "chat/completions"`, `api_base: str | None = None`, `method: str = "POST"`, `json: dict | None = None`, `headers: dict | None = None`
- URL format: `{api_base}/upstream/{model_id}/{endpoint}` (e.g., `http://localhost:8080/upstream/llama-3.1-8b/chat/completions`)
- Defaults `api_base` to `"http://localhost:8080"`
- The upstream routing is handled by llama-swap itself; the client just needs the correct base URL
- Respects `LOCAL_LLM_API_KEY` first, then falls back to `OPENAI_API_KEY` for the `Authorization: Bearer` header
- Returns a `requests.Response` with `.json()`, `.text`, `.status_code`, etc.

### Module `__all__`

Updated to: `["load_client", "load_llama_swap_upstream_client"]`

## Notes

- Zero new dependencies — `requests` is already available as a transitive dependency of `openai`.
- The `LOCAL_LLM_API_KEY` env var is optional; local LLMs (especially llama-swap) often run without authentication.
- Tests should cover: key precedence, `api_base` passthrough, `local=True` behavior for `load_client()`, upstream endpoint URL construction, and backward compatibility.

## History

### 2026-08-10 15:00
**Summary** — Draft spec for AI client enhancement
**Changes**
- Added `local` parameter to `load_client()` for `LOCAL_LLM_API_KEY` support.
- Added `api_base` parameter to `load_client()` for arbitrary OpenAI-compatible endpoints.
- Removed `load_llama_swap_client()` — merged into `load_client()`.
- Added `load_llama_swap_upstream_client()` for direct upstream endpoint access.
- Updated `load_llama_swap_upstream_client()` to use `requests` and return a `requests.Response`.
- Removed `local` parameter from `load_llama_swap_upstream_client()` since it's a local client already.
- Defined precedence rules and default values.
