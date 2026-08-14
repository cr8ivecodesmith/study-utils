# Docs Generator Local LLM Config — Implementation

## Understanding

The current `generate-document` feature hardcodes local LLM connection parameters (`use_local`, `api_base`) in a module-level dictionary and further hard-codes temperature and max_tokens. These values are not accessible via `documents.toml`. The RAG feature already solves this problem with its `[services.chat]` TOML section, typed dataclasses, and validation helpers.

This task expands the documents generator config to declare these same connection parameters in a first-class `[llm]` section, follows the RAG pattern (dataclass + TOML mapping), and wires them through `runner.py` into `load_client()`. Defaults remain identical to the current hard-coded values for backwards compatibility.

### Assumptions / Open Questions

1. **[llm] extraction before per-doc-type loop:** The existing `load_documents_config()` iterates raw TOML keys and skips entries without a `prompt` field (lines 89-91). The `[llm]` section has no `prompt`, so it would be silently skipped unless extracted first. We extract it explicitly before the loop.

2. **Return type change:** `load_documents_config()` currently returns `Dict[str, Dict[str, str]]`. A wrapper dataclass (`DocumentsConfig`) holding both `llm: LLMConfig` and `docs: Dict[str, Dict[str, str]]` is preferred for consistency with RAG's `RagConfig` approach. Only `runner.py` calls this function directly; `cli.py` delegates through it, so cascade changes are minimal.

3. **GPT-5 temperature override:** The spec says GPT-5 special handling takes precedence over the `[llm]` section default temperature. Implementation will set `temperature = llm_cfg.temperature` first, then apply the gpt-5 cap (`1.0`) and `max_completion_tokens = 8192` if `"gpt-5"` is in the model string.

4. **Validation strictness:** We use same validators as RAG's `ServiceAIConfig`: `_require_bool` for `use_local`, `_require_string` (with default) for `api_base` and `provider`, `_require_float_range` for `temperature`, and `_require_positive_int` for `max_tokens`. Missing keys resolve to defaults from `_DEFAULT_LLM_CONFIG`.

5. **No change to `load_client()`:** The function at `src/study_utils/core/ai.py:19` already accepts `local: bool` and `api_base: str | None`. We only pass new config-derived values.

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `[llm]` section silently skipped if not extracted before per-doc-type loop | Medium — config parsed but unused | Add explicit `llm_section = raw.pop("llm", {})` before the loop, with fallback to defaults |
| Return type change cascades to callers | Low — only `runner.py` calls `load_documents_config()` directly; `cli.py` delegates through it | Minimal diff in `runner.py`; all existing tests still pass since wrapper has direct accessors |
| Custom `documents.toml` with `[llm]` that has extra keys or comments | Low — strict validation could reject unexpected types | Type coercion via `_coerce_*` helpers; unknown keys ignored (TOML parsing is permissive) |

## Resources

### Project docs

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/generate_document/config.py` — Current config parsing. `load_documents_config()` returns flat dict. Entry point for adding `[llm]` dataclass, `_build_llm()`, and return wrapper type. **Primary implementation file.**

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/generate_document/runner.py` — Document generation orchestration. Hardcoded `_DEFAULT_LOCAL_LLM` dict (lines 13-16), temperature (line 96), and max_tokens (lines 99-103). **Secondary file** — replace hard-coded values with config-derived ones, wire into `load_client()`.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/generate_document/documents.toml` — TOML template. Gets new `[llm]` section at top.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/core/ai.py` — `load_client(local, api_base)` definition (line 19). **No changes needed.** Accepts both parameters already.

- `/home/ubuntu/Projects/matt/study-utils/tests/test_generate_document.py` — Existing test file with ~13 tests covering config parsing, runner behavior, CLI interaction. Tests will be extended, new dataclass, and test fixture `openai_factory`. **No new test files needed.**

### External references

- `src/study_utils/rag/config.py` — RAG's implementation patterns to follow:
  - `ServiceAIConfig` dataclass (lines 87-91): `use_local`, `api_base`, `provider`
  - `_build_*()` builder functions with validation helpers (`_require_bool`, `_require_string`, etc.)
  - `_DEFAULTS` dict with typed default values
  - `config_template()` string with TOML blocks

## Impact Analysis

### Affected behaviors & tests (unit)

| Current behavior | New behavior | Tests to add/modify |
|-----------------|-------------|---------------------|
| `load_documents_config()` returns `{keywords: {prompt, description, model}, ...}` — flat dict | Returns wrapper with `.llm` and `.docs`; `[llm]` extracted before loop | Add 3–4 tests for LLM config parsing (with section, without section, type errors) |
| `runner.py` uses `_DEFAULT_LOCAL_LLM` dict: hardcoded `USE_LOCAL=True`, `API_BASE="http://localhost:8080/v1"` | Reads `llm_cfg.use_local` and `llm_cfg.api_base` from config | Add 1 test verifying correct call to `load_client()` with config-derived values |
| Temperature is `0.2`; gpt-5 gets `1.0`; max_tokens is `4096` (gpt-5 uses `max_completion_tokens = 8192`) | Config values: `temperature=0.2`, `max_tokens=4096`; gpt-5 override preserved | Verify existing test `test_generate_document_writes_output_with_stubbed_client` still passes; add GPT-5 config test if needed |
| Existing `[keywords]`, `[reading_assignment]`, `[book]` sections unchanged | Same content, same parsing | All 13 existing tests pass without modification (backward compat) |

**Total new test cases: 4–5**

### Affected source files

**Modify:**
- `src/study_utils/generate_document/config.py` — Add `LLMConfig` dataclass, `_DEFAULT_LLM_CONFIG` constant, `_build_llm()` builder, update `load_documents_config()` return type and `[llm]` extraction.

- `src/study_utils/generate_document/runner.py` — Remove `_DEFAULT_LOCAL_LLM` dict, import new types, consume `llm_cfg` in `generate_document()`, pass values to `load_client()`, use config-derived temperature and max_tokens with gpt-5 override preserved.

**Modify (config template only):**
- `src/study_utils/generate_document/documents.toml` — Add `[llm]` section at top with all five keys and optional comments.

**No changes:**
- `src/study_utils/core/ai.py` — `load_client()` signature unchanged.
- `tests/test_generate_document.py` — Extended (new tests), no structural change to existing test file format or fixtures.

### Security considerations

- No new secrets or environment variables introduced. The feature reuses `LOCAL_LLM_API_KEY` and `OPENAI_API_KEY`.
- TOML config files keep existing permissions (`0o600`).
- `api_base` validated as a non-empty string when present, preventing accidental misdirection of API requests.

## Solution Plan

### Architecture / pattern choices

1. **Dataclass wrapper pattern:** Mirror RAG's `RagConfig` approach. A single `DocumentsConfig` dataclass with:
   - `llm: LLMConfig` — global local LLM connection params
   - `docs: Dict[str, Dict[str, str]]` — per-document-type entries (unchanged content)

2. **Builder function pattern:** Add `_build_llm(section: Mapping[str, Any]) -> LLMConfig` using the same validation helpers (`_require_bool`, `_require_string`, `_require_float_range`, `_require_positive_int`) that RAG uses in `config.py`. This enforces strict types and provides good error messages.

3. **Config extraction order:** In `load_documents_config()`:
   - Open TOML file, load raw dict
   - Extract `[llm]` section first (pop from raw), build LLMConfig
   - Iterate remaining keys for per-doc-type entries (same logic as before)
   - Return wrapper

4. **Backwards-compatible defaults:** `_DEFAULT_LLM_CONFIG` dataclass instance with values matching current hard-coded state:
   - `use_local = True` (was `True`)
   - `api_base = "http://localhost:8080/v1"` (same)
   - `provider = "local"`
   - `temperature = 0.2` (was hard-coded)
   - `max_tokens = 4096` (was hard-coded for non-gpt-5 models)

### Dependency injection & boundaries

- Caller boundary: `runner.py.generate_document()` reads config, creates the OpenAI client in one call to `load_client()`, then uses it for a single chat completion request. No shared state, no global toggles.
- Config parsing is isolated to `_build_llm()`. The per-doc-type loop remains unchanged — we only add extraction before the loop.

### Stepwise implementation checklist

- [ ] **Add LLMConfig dataclass** to config.py — with `use_local`, `api_base`, `provider`, `temperature`, `max_tokens`; all typed; frozen=True; default values matching current hard-coded state.

- [ ] **Add _DEFAULT_LLM_CONFIG constant** — module-level dataclass instance for fallback when `[llm]` is missing from TOML or not present during tests.

- [ ] **Add _build_llm() builder function** — reads raw TOML section, coerces types, validates ranges (temperature 0.0–2.0), returns LLMConfig. Uses same helpers as RAG config._require_bool, _require_float_range, etc.

- [ ] **Update load_documents_config()** — extract `[llm]` section before the per-doc-type loop; call `_build_llm(llm_section)` with fallback to `_DEFAULT_LLM_CONFIG`; return wrapped dataclass (or TypedDict) with both `llm` and `docs` fields.

- [ ] **Update runner.py imports** — remove `_DEFAULT_LOCAL_LLM` dict, import `DocumentsConfig` (or the return type from config), import `_build_llm` if needed for tests.

- [ ] **Wire LLM params into load_client()** — in generate_document(), pass `llm_cfg.use_local` and `llm_cfg.api_base` to `load_client()` instead of hardcoded dict lookups.

- [ ] **Wire temperature & max_tokens** — use `llm_cfg.temperature` as base; apply gpt-5 override (`1.0`, `max_completion_tokens=8192`); use `llm_cfg.max_tokens` for non-gpt-5 fallback instead of hard-coded 4096.

- [ ] **Add [llm] section to documents.toml** — add five keys at top with optional comment annotations matching RAG's style.

- [ ] **Write tests** — 4–5 targeted unit tests in existing test_generate_document.py:
  - Config parsing with `[llm]` present
  - Config parsing without `[llm]` (backwards compat)
  - Type validation errors raise descriptive errors
  - Runner passes correct params to `load_client()`

- [ ] **Verify** — run full test suite; expect all existing 13 generate_document tests to pass plus new ones.

## Test Plan

### Unit tests (in existing `tests/test_generate_document.py`)

1. **`test_load_documents_config_with_llm_section`** — Write a TOML with `[llm]` section (`use_local = false`, `api_base = "http://x:9000/v1"`, `provider = "local"`, `temperature = 0.5`, `max_tokens = 2048`). Call `load_documents_config()`. Verify `.llm` has correct values and `.docs` still contains `[keywords]`, `[reading_assignment]`, etc.

2. **`test_load_documents_config_without_llm_section_defaults`** — Write a minimal TOML with only per-doc-type entries (no `[llm]`). Call `load_documents_config()`. Verify `.llm` resolves to `_DEFAULT_LLM_CONFIG` values (`use_local=True`, `api_base="http://localhost:8080/v1"`, `provider="local"`, `temperature=0.2`, `max_tokens=4096`).

3. **`test_load_documents_config_llm_type_errors`** — Write a TOML with `[llm]` containing invalid types: `use_local = "yes"` (not bool), `temperature = 5.0` (out of range). Verify that the builder raises `ValueError` with descriptive message.

4. **`test_generate_document_passes_llm_params_to_client`** — Write a TOML with `use_local = false`, `api_base = "http://custom:1234/v1"`. Mock `load_client()` in runner.py. Call `generate_document()`. Assert that `load_client()` was invoked with `local=False, api_base="http://custom:1234/v1"` (not the hard-coded defaults).

5. **`test_generate_document_gpt5_override_with_llm_section`** *(extension of existing test)* — Write a TOML with `[llm]` setting `temperature = 0.7`. Write a config with `model = "gpt-5-turbo"`. Call `generate_document()`. Verify the client's `create()` call receives `model="gpt-5-turbo"`, `temperature=1.0` (gpt-5 override, not 0.7), and `max_completion_tokens=8192`.

### Contract / call site wiring verification

6. **Config wrapper is accessible to CLI** — The `generate_document_cli.config()` subcommand already uses `cli.py` which imports the same loader; verify that running `study generate-document config` with a [docs](file:///home/ubuntu/Projects/matt/study-utils/src/study_utils/rag/vector_store.py#L123-L142) file containing `[llm]` does not error out (inspected by existing test `test_load_documents_config_import_failure` pattern).

7. **Backwards compat** — Existing TOML files without `[llm]` should continue to work unchanged. Verified by Test #2 above and by ensuring the per-doc-type loop uses the same prompt/exit criteria as before.

## Operability

### Telemetry

No new logs or metrics required. The existing `load_client()` in `core/ai.py` prints connection details (base URL, which key was used) at DEBUG level. The `runner.py` error message `"AI returned empty content"` remains the same and continues to be valid.

### Revert steps

1. Remove the `[llm]` section from `documents.toml`.
2. In `config.py`, revert `load_documents_config()` to return `Dict[str, Dict[str, str]]` (or keep wrapper with optional `.llm`).
3. In `runner.py`, revert to `_DEFAULT_LOCAL_LLM` dict (comment it out but leave in file temporarily).
4. Callers that already pass config-derived values will continue working — either from TOML or falling back to dataclass defaults.

### Rollout

The `[llm]` section is entirely additive. Existing files not containing `[llm]` get sensible defaults from `_DEFAULT_LLM_CONFIG`. The builder pattern ensures typed validation — a user who writes `use_local = true` (string) will benefit from coercion, or strict validation depending on config loading path chosen.

## History

### 2026-08-14 draft
**Summary** — Implementation doc drafted after codebase exploration and spec review.
**Changes**
- Confirmed current state: `_DEFAULT_LOCAL_LLM` dict in runner.py; `load_documents_config()` returns flat dict; `[llm]` would be skipped if not extracted first.
- Chose dataclass wrapper approach (Option A) matching RAG's RagConfig pattern.
- Defined 4–5 new unit tests, all within existing test_generate_document.py.
- Identified single key risk: [llm] section must be extracted before per-doc-type loop so it isn't skipped by the prompt check at lines 89-91 of config.py.
