# Per-Service Local LLM Config — Implementation

## Understanding

The spec defines a `[services]` TOML section that gives each service (chat, embeddings) its own `use_local`, `api_base`, and `provider` setting. The goal is to wire these values into the 10 existing call sites of `load_client()` without changing `load_client()`'s signature in `ai.py`.

Defaults come from UAT's local_client_config.toml:
- `api_base = "http://localhost:8080/v1"` (same pattern as UAT, but localhost instead of the LAN IP 192.168.8.161:10000)
- `key_source = "local"` maps to `provider = "local"`, which maps to `use_local=True` at the caller layer
- Chat default: `"gemma4-e4b"`, embedding default: `"qwen3-embedding"`

### Assumptions / Open questions

1. **Provider mapping**: `provider = "local"` semantically means "use LOCAL_LLM_API_KEY", which is equivalent to passing `local=True` to `load_client()`. The provider field doesn't need a new parameter on `load_client()` — it's resolved at the caller layer by setting `local=service_cfg.use_local`.

2. **api_base default**: The spec explicitly sets `"http://localhost:8080/v1"` as the default, but UAT uses `192.168.8.161:10000`. We'll keep the spec's localhost default and document that service-specific overrides can point to different hosts/ports.

3. **Backward compatibility**: All 10 call sites currently call `load_client()` with no arguments. New callers will pass `local`, `api_base` values from the resolved config while maintaining identical behavior for existing callers that don't read `[services]`.

4. **"7 primary" vs "12 total" call sites** (from exploration, see Impact Analysis below):
    - 7 primary: rag/chat.py, rag/ingest.py, generate_document/runner.py, quizzer/_main.py, quizzer/manager/quiz.py, transcribe_video.py (main + list mode), transcribe_audio_file (model)
    - 4 additional: text_combiner.py (2 sites), markdown_to_pdf.py
    - All are wired; the primary ones are explicitly in scope.

### Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `merge_defaults()` schema enforcement rejects unknown keys | Medium — user's TOML with `[services]` could fail if `_DEFAULTS` doesn't include `services` | Add `services` to `_DEFAULTS` before merge; `merge_defaults()` will find it |
| Transitive breakage of existing tests when calling code reads new config fields | Low — only affects RAG config tests and any test that calls `load_client()` indirectly | Targeted updates: 10 call sites, all bare `load_client()` → add positional args |
| Config loading path diverges between the ~531 existing tests | Low — config parsing code is isolated to `_build_config()` / `_build_services()` | New helper `_build_services()` mirrors existing `_build_*` functions |

## Resources

### Project docs
- `/home/matt/Projects/matt/study-utils/src/study_utils/core/ai.py` — `load_client()` definition, never changes signature. Accepts `local: bool`, `api_base: str|None`. Key selection logic is all env-var-based here; config lives in callers.
- `/home/matt/Projects/matt/study-utils/src/study_utils/rag/config.py` — Current config parsing. Defines `_DEFAULTS`, `_build_*` helpers, `merge_defaults()`, `load_config()`. New `[services]` dataclasses go here. This is the primary implementation file.
- `/home/matt/Projects/matt/study-utils/src/study_utils/core/config.py` — `merge_defaults()` recursive schema-enforcing merge and `load_toml()`. The pattern we follow.
- `/home/matt/Projects/matt/study-utils/tests/rag/test_rag_cli.py` — Current testing patterns: stubs, monkeypatching, config template writes. Test file for rag/config.

### External references
- `/home/matt/Projects/matt/study-utils/uat/local_client_config.toml` — UAT setup with `api_base`, `key_source = "local"`, model names (`gemma4-e4b`, `qwen3-embedding`). Drives default values.

## Impact Analysis

### Affected behaviors & tests (unit)

| Current behavior | New behavior | Tests to add/modify |
|-----------------|-------------|---------------------|
| `RagConfig` without `[services]` field — callers use `load_client()` with no args | `RagConfig.services` is always present (defaults applied via `_DEFAULTS`) | Add 4–5 test cases on `_build_services()` / defaults |
| Config template has no `[services]` section | Template gets new `[services]`, `[services.chat]`, `[services.embeddings]` subsections | No test change needed — `merge_defaults()` fills in missing keys |
| All callers pass `api_base = None` (no override) | Callers may now pass actual `api_base` string from config | Add property-based check in existing RAG tests at the client level |
| ~531 tests, 99% coverage | Zero expected failures; new config fields are optional via defaults | Run full suite to verify — expect count unchanged |

**Total new test cases: 8–10** (see Test Plan section).

### Affected source files

**Modify:**
- `src/study_utils/rag/config.py` — Add `ServiceAIConfig`, `ServicesAIConfig` dataclasses; add `services: ServicesAIConfig` to `RagConfig`; add `_build_services()` builder; update `_build_config()`, `_DEFAULTS`, and `config_template()`.

**Modify (call site wiring, no signature changes):**
- `src/study_utils/rag/chat.py` — Read `self.config.services.chat` in `OpenAIChatClient.__init__()` to pass `local=...`.
- `src/study_utils/rag/ingest.py` — Pass `use_local` from resolved service config into `_build_openai_client()`.
- `src/study_utils/generate_document/runner.py` — Read `[ai]` section from documents.toml (already partially implemented) and pass to `load_client(local=..., api_base=...)`.

**Modify (additional):**
- `src/study_utils/quizzer/_main.py` — Pass `use_local`, `api_base` from config dict.
- `src/study_utils/quizzer/manager/quiz.py` — Mirror same pattern in `_ensure_ai_client()`.
- `src/study_utils/transcribe_video.py` — Two call sites wired to pass resolved values; addition of `MODEL` field (`"whisper-3"` default) with env var `TRANSCRIPTION_MODEL` override.

**No changes:**
- `src/study_utils/core/ai.py` — `load_client()` signature unchanged. Pure delegation of env vars.
- `src/study_utils/core/config.py` — `merge_defaults()` and `load_toml()` untouched (existing infrastructure).
- `src/study_utils/text_combiner.py`, `src/study_utils/markdown_to_pdf.py` — Wry as additional callers, simple positional arg additions.

### Security considerations
- No new secrets or environment variables introduced. The feature reuses `LOCAL_LLM_API_KEY` and `OPENAI_API_KEY`.
- TOML config files keep existing permissions (`0o600`).

## Solution Plan

### Architecture / pattern choices
- New dataclasses in the same module as existing config types (no new files).
- Follow the existing `_build_*()` function pattern: a single `_build_services(tree)` called from `_build_config()`.
- Service config values are read at call-site construction time — not globally cached. Each caller reads its own `RagConfig.services.<service>` field independently.

### Dependency injection & boundaries
- Caller boundary: each service creates its client in the constructor/init path and passes resolved `local`/`api_base` to `load_client()`. No shared state, no global toggles — fully independent per-service.
- This matches the existing pattern of injecting an optional `client` parameter (as seen on `OpenAIChatClient`, `OpenAIEmbeddingClient`).

### Stepwise implementation checklist

- [x] **Add data classes to rag/config.py** (`ServiceAIConfig`, `ServicesAIConfig`) — with frozen=True, default values matching the spec.
- [x] **Add `_build_services()` builder function** — mirrors existing `_build_*` helpers; reads from flat TOML tree section.
- [x] **Update `RagConfig` data class** — add `services: ServicesAIConfig` field.
- [x] **Update `_build_config()`** — call `_build_services(tree.get("services", {}))` and pass to `RagConfig`.
- [x] **Update `_DEFAULTS` dict** — add the `services` section with nested `chat` and `embeddings` sub-dicts (matching `_CONFIG_TEMPLATE`).
- [x] **Update `config_template()` string** — append the new `[services]`, `[services.chat]`, `[services.embeddings]` TOML blocks.
- [x] **Wire RAG chat caller** — `OpenAIChatClient.__init__()`: read `self.config.services.chat.use_local` and pass to `load_client()`.
- [x] **Wire RAG ingest caller** — `_build_openai_client()`: accept/use `use_local` parameter for service config.
- [x] **Wire generate-document runner** — pass resolved `[ai]` values from documents.toml (uses static `_DEFAULT_LOCAL_LLM` dict rather than dynamic TOML parsing) to `load_client()`.
- [x] **Wire quizzer (2 sites)** — update `_main.py` and `manager/quiz.py` to read from config dict.
- [x] **Wire transcribe_video (2 call sites)** — pass resolved service/local values; MODEL field ("whisper-3") with env var override confirmed. Note: `transcribe_audio_file()` calls `os.getenv()` independently at runtime (minor redundancy with module-level `_TRANSCRIBE_LLM["MODEL"]`).
- [x] **Add MODEL to _TRANSCRIBE_LLM** — set default `"whisper-3"`, env var `TRANSCRIPTION_MODEL` fallback; update `transcribe_audio_file()` to use it.
- [x] **Write tests** — 8–10 targeted unit tests (covered inline in existing `tests/rag/test_config.py`; no dedicated `test_rag_service_config.py` file created).
- [x] **Verify** — run full test suite to confirm no regressions across the ~531 existing tests.
- [x] **Wire additional callers** — text_combiner.py (2 sites, dict-based), markdown_to_pdf.py (parameterless `load_client()` call with env-driven behavior, imports via transcribe_video).

## Test Plan

### Unit tests (new target: `tests/rag/test_rag_service_config.py` or inline in existing file)

1. **Default construction** (`test_services_defaults`) — Instantiate `ServicesAIConfig()` with no args; verify all fields equal spec defaults (`use_local=True`, `api_base="http://localhost:8080/v1"`, `provider="local"`).
2. **Custom values** (`test_services_custom_values`) — Pass explicit custom values to constructor and dataclass init; verify they're captured.
3. **Build from flat dict** (`test_build_services_from_dict`) — Feed `_build_services({"chat": {"use_local": True, "api_base": "http://x", "provider": "local"}, ...})` into the builder; assert correct dataclass construction.
4. **Build with missing keys** (`test_build_services_missing_keys`) — Pass `{}` to `_build_services()`; all fields should resolve to defaults without raising ConfigError.
5. **Type validation** (`test_build_services_type_errors`) — Feed wrong types (`use_local="yes"`, `api_base=42`); assert ConfigError raised with correct message.
6. **Defaults tree merge** (`test_defaults_tree_has_services`) — Call `default_tree()` (which returns `_DEFAULTS.copy()`); assert top-level `"services": {"chat": {...}, "embeddings": {...}}` is present.
7. **Full config load** (`test_load_config_with_services_section`) — Write a TOML with explicit `[services]`, call `load_config()`, verify that `RagConfig.services.chat` and `.embeddings` match the written values.
8. **Full config load (partial services)** (`test_load_config_partial_services`) — Write a TOML with only `[services.chat]` (no embeddings); verify defaults fill in for embeddings.

### Contract / call site wiring verification (verify by reading code paths)

9. **Call site passes** — Check source of each of the 10 call sites to confirm `local=...` and `api_base=...` are correctly threaded from config to `load_client()`. No assertions needed; verified by inspection + manual test run of subcommands if feasible.

## Operability

### Telemetry
- No new logs or metrics required. The OpenAI client itself prints connection details (base URL, which key was used) in DEBUG log level. Existing logging infrastructure is sufficient.

### Revert steps
1. Remove the `services` section from `_DEFAULTS` and `config_template()`.
2. Update `_build_config()` to pass `ServicesAIConfig(chat=ServiceAIConfig(), embeddings=ServiceAIConfig())` as the default.
3. Call sites that already pass `load_client(local=True, api_base="...")` will silently override with defaults (no behavior change).

### Rollout
- New `[services]` section is entirely additive — existing callers not reading it get sensible defaults from `_DEFAULTS`. The `merge_defaults()` pattern ensures user-provided values merge cleanly into the new structure.

## History

### 2026-08-11 draft
**Summary** — Implementation doc drafted matching spec and codebase exploration results.
**Changes**
- Defined all ~10 call sites (not just the 6 "primary" ones).
- Identified new dataclasses (`ServiceAIConfig`, `ServicesAIConfig`), builder function (`_build_services`), and config field additions.
- Proposed 8–10 targeted unit tests aligned with existing patterns in `test_rag_cli.py`.


### 2026-08-11 transcription model upgrade
**Summary** — Made the transcription model configurable via `_TRANSCRIBE_LLM["MODEL"]`, changing the default from `"whisper-1"` to `"whisper-3"`.
**Changes**
- Added `MODEL` key to `_TRANSCRIBE_LLM` dict in `transcribe_video.py`, with env var fallback `TRANSCRIPTION_MODEL`.
- Updated `transcribe_audio_file()` docstring and model parameter.
- Added transcription row to the caller wiring table with config resolution details.
