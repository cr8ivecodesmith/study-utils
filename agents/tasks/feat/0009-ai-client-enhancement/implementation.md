# AI Client Enhancement — Implementation

## Understanding

- Extend `load_client()` in `src/study_utils/core/ai.py` to accept `local: bool` and `api_base: str | None`, merging the old `load_llama_swap_client()` logic back in.
- Add a new `load_llama_swap_upstream_client()` that uses `requests` to call llama-swap's upstream endpoint and returns a `requests.Response`.
- No new dependencies — `requests` is already a direct dep of the project.

### Assumptions / Open questions
- `LOCAL_LLM_API_KEY` is optional for upstream calls (llama-swap often runs without auth).
- The upstream `api_base` defaults to `"http://localhost:8080"` without a `/v1` suffix.
- `load_llama_swap_upstream_client()` does not need a `local` parameter since it's inherently local.

### Risks & mitigations
- **Breaking change to `load_client()` signature** — new params are optional with defaults, so existing callers are unaffected.
- **`requests` dependency** — already present as a direct dep in `pyproject.toml`.

## Resources

### Project docs
- `src/study_utils/core/ai.py` — target module for changes.
- `tests/test_core_ai.py` — existing tests to extend.
- `tests/fixtures/openai.py` — OpenAI stub factory for mocking.
- `agents/guides/engineering-guide.md` — seams, DI, testing focus.
- `agents/guides/styleguides.md` — Google docstrings, Ruff config, naming.

### External docs
- llama-swap upstream docs — URL format `{base}/upstream/{model_id}/{endpoint}`.

## Impact Analysis

### Affected behaviors & tests
| Behavior | Tests to add/modify |
|---|---|
| `load_client()` with `local=True` | New: key resolution with `LOCAL_LLM_API_KEY` |
| `load_client()` with `api_base` | New: `api_base` passthrough to `OpenAI()` |
| `load_client()` backward compat | Existing tests pass; no signature break |
| `load_llama_swap_upstream_client()` URL construction | New: model_id, endpoint, method, json, headers |
| `load_llama_swap_upstream_client()` auth header | New: LOCAL_LLM_API_KEY vs OPENAI_API_KEY |
| `load_llama_swap_upstream_client()` no API key | New: no Authorization header |
| `__all__` updated | New: verify export |

### Affected source files
- **Modify:** `src/study_utils/core/ai.py`
- **Modify:** `tests/test_core_ai.py`
- **No changes:** `tests/fixtures/openai.py` (stub factory already supports `init_kwargs`)

### Security considerations
- API keys read from environment only; never logged or exposed.
- `LOCAL_LLM_API_KEY` and `OPENAI_API_KEY` are treated equally as secrets.

## Solution Plan

### Architecture / pattern choices
- Follow the "pure core, dirty edges" principle: key resolution is pure logic; I/O is bounded to the `requests` call.
- Use factory-style fixtures in tests (consistent with existing `openai_factory`).

### Stepwise checklist
- [x] Add `local: bool = False` and `api_base: str | None = None` to `load_client()` signature.
- [x] Implement key resolution: `LOCAL_LLM_API_KEY` (when `local=True`) → `OPENAI_API_KEY` → `RuntimeError`.
- [x] Pass `api_base` to `OpenAI()` only when not `None`.
- [x] Add `load_llama_swap_upstream_client()` with parameters: `model_id`, `endpoint`, `api_base`, `method`, `json`, `headers`.
- [x] Build URL: `{api_base}/upstream/{model_id}/{endpoint}`.
- [x] Resolve auth header: `LOCAL_LLM_API_KEY` first, then `OPENAI_API_KEY`, then no header.
- [x] Call `requests.request()` and return `requests.Response`.
- [x] Update `__all__` to include `load_llama_swap_upstream_client`.
- [x] Add tests for all new behaviors.
- [x] Run `ruff` and `pytest`; fix any findings.

## Test Plan

### Unit cases
1. `test_load_client_local_true_uses_local_key` — `LOCAL_LLM_API_KEY` set, `OPENAI_API_KEY` absent, `local=True`.
2. `test_load_client_local_true_precedes_openai_key` — both keys set, `local=True` picks `LOCAL_LLM_API_KEY`.
3. `test_load_client_local_true_no_key_raises` — neither key set, `local=True` raises `RuntimeError` mentioning `LOCAL_LLM_API_KEY`.
4. `test_load_client_api_base_passthrough` — `api_base="http://example.com/v1"` passed to `OpenAI()`.
5. `test_load_client_api_base_none_omitted` — `api_base=None` not passed to `OpenAI()`.
6. `test_load_client_backward_compat` — no args, uses `OPENAI_API_KEY` as before.
7. `test_upstream_client_default_url` — default URL is `http://localhost:8080/upstream/default/chat/completions`.
8. `test_upstream_client_custom_model_id` — `model_id="llama-3.1-8b"` → `/upstream/llama-3.1-8b/`.
9. `test_upstream_client_custom_endpoint` — `endpoint="completions"` → `/chat/completions` path.
10. `test_upstream_client_method_get` — `method="GET"` sends GET request.
11. `test_upstream_client_json_payload` — `json={"messages": [...]}` sent in body.
12. `test_upstream_client_headers` — custom headers merged with `Authorization: Bearer`.
13. `test_upstream_client_local_key_auth` — `LOCAL_LLM_API_KEY` used for Authorization header.
14. `test_upstream_client_no_key_no_auth` — no key → no `Authorization` header.
15. `test_all_exports` — `ai.__all__` includes both functions.

### Contract
- `load_client()` returns an `OpenAI` instance (or stub) — existing tests validate.
- `load_llama_swap_upstream_client()` returns a `requests.Response` — verified by asserting `.status_code`, `.json()`, `.text`.

### Manual checks
- Run with real llama-swap instance (if available) to verify URL construction.

## Operability

- No new log statements required; existing `load_dotenv()` call handles env loading.
- Revert: removing the new params is safe since they have defaults — existing callers are unaffected.

## History

### 2026-08-10 15:30
**Summary** — Draft implementation for AI client enhancement
**Changes**
- Extended `load_client()` with `local` and `api_base` parameters.
- Added `load_llama_swap_upstream_client()` using `requests`.
- Defined 15 unit test cases covering all new behaviors.
- Zero new dependencies.

### 2026-08-10 19:08
**Summary** — Committed `load_client()` changes with tests
**Changes**
- Committed `load_client()` with `local` and `api_base` parameters.
- Added 6 unit tests for `load_client()` in `tests/test_core_ai.py`.
- Added `requests` to `pyproject.toml`.
- Remaining: implement `load_llama_swap_upstream_client()` and update `__all__`.

### 2026-08-10 19:30
**Summary** — Implemented `load_llama_swap_upstream_client()` and completed all remaining tasks
**Changes**
- Added `load_llama_swap_upstream_client()` function to `src/study_utils/core/ai.py`.
- Updated `__all__` to export the new function.
- Added 9 unit tests for the upstream client in `tests/test_core_ai.py`.
- Fixed `api_base` → `base_url` mismatch in existing tests.
- All 18 tests pass; ruff linting clean.
