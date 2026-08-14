# Docs Generator Local LLM Connection Config — Spec

## Summary

Expand the `generate-document` feature config (`documents.toml`) to declare local LLM
connection parameters (`use_local`, `api_base`, `provider`, `temperature`, `max_tokens`)
as a first-class `[llm]` section, mirroring the approach already used by the RAG feature.
Replace the hardcoded defaults in `runner.py` with config-derived values so that both
features share the same local LLM connection model without duplicating magic constants.

## Goals

- Users can configure which local LLM endpoint to connect from, directly in `documents.toml`,
  without needing environment variables or code changes.
- All connection-related values (`use_local`, `api_base`, `provider`, `temperature`,
  `max_tokens`) flow from config into `runner.py` at runtime.
- The change is backwards-compatible: existing `documents.toml` files that lack an `[llm]`
  section continue to work with the same behaviour as before.
- The implementation follows the RAG feature's pattern (dataclass + TOML mapping) for
  consistency across features.

## Non-Goals

- No changes to how OpenAI API keys resolved — `use_local` still controls whether
  `LOCAL_LLM_API_KEY` is preferred over `OPENAI_API_KEY`.
- No new dataclasses or config sections added beyond what is needed for local LLM params.
- No support for per-document-type temperature/max_tokens overrides (this can be a follow-up).
- RAG's own config (`rag/config.py`) is not modified; the spec covers only the documents
  generator feature.

## Behavior (BDD-ish)

- Given a `documents.toml` file with an `[llm]` section, **then** the CLI uses all
  declared parameters (`use_local`, `api_base`, `provider`, `temperature`, `max_tokens`)
  when making LLM calls for any document type.

- Given a `documents.toml` file without an `[llm]` section, **then** the fallback values
  match the previously hard-coded defaults:
  `use_local=true`, `api_base="http://localhost:8080/v1"`, `provider="local"`,
  `temperature=0.2`, `max_tokens=4096`.

- Given a document type with `model = "gpt-5"` and an `[llm]` section that sets
  `temperature = 0.3`, **then** the GPT-5 override still applies (temperature is set to
  `1.0` and `max_completion_tokens = 8192`) — GPT-5 special handling takes precedence
  over the section's default temperature.

- Given a user that sets `use_local = false` in `[llm]`, **then** the OpenAI client is
  instantiated without the `local=True` flag, and calls go to the configured base URL or
  the default OpenAI endpoint.

## Constraints & Dependencies

- **Python version:** Python 3.11+ (already required for `tomllib`).
- **TOML library:** Uses the same `tomllib` / `tomli` fallback pattern as existing code.
- **`load_client()`:** The function in `core/ai.py` already accepts `local` and `api_base`
  parameters — no changes needed there.
- **Backwards compatibility:** All keys are optional with sensible defaults; missing `[llm]`
  section produces identical behaviour to the pre-change hard-coded values.
- **Config loading:** `load_documents_config()` currently returns `Dict[str, Dict[str, str]]`.
  The updated return type should be adjusted (e.g., a wrapper dataclass or TypedDict) so
  callers can access both the global `[llm]` config and per-doc-type entries.

## Security & Privacy

- No secrets are read from the TOML file directly — only connection metadata
  (`use_local`, `api_base`, `provider`). Secret values (`OPENAI_API_KEY`, `LOCAL_LLM_API_KEY`)
  flow through environment variables as before.
- The `api_base` is validated as a non-empty string when present, preventing accidental
  misdirection of API requests.

## Telemetry & Operability

- Existing error messages in `runner.py` (e.g., "AI returned empty content") remain valid.
- The config subcommand (`cll.py`) should reflect `[llm]` keys in its output so users can
  inspect their configuration.
- Logging level / verbose flags are unaffected, since they belong to the RAG feature's
  broader logging configuration.

## Rollout / Revert

- No runtime migration required: TOML parsing gracefully ignores unknown sections for legacy
  configs or, equivalently, simply reads a newly-present section when present.
- To revert: remove the `[llm]` section and restore the inline defaults; both `config.py`
  dataclass fallbacks and `runner.py` defaults ensure nothing breaks during the intermediate state.

## Definition of Done

- [ ] Behavior verified — all scenarios in "Behavior (BDD-ish)" hold.
- [ ] `documents.toml` template updated with `[llm]` section.
- [ ] `config.py` parses `[llm]` into a dataclass; `load_documents_config()` returns enriched config.
- [ ] `runner.py` consumes config-derived values and passes all connection params to `load_client()`.
- [ ] Tests added or updated (unit tests for config parsing and runner behaviour with/without `[llm]`).
- [ ] Existing `documents.toml` files validated against the new schema (backwards-compatible).

## Ownership

- Owner: @matt
- Reviewers: To be assigned
- Stakeholders: Study CLI users who rely on local LLM for document generation

## Links

- Related: `feat/0010-per-service-local-llm` (RAG's per-service local LLM config)
- Related: RAG feature config at `src/study_utils/rag/config.py`
- Reference: `load_client()` at `src/study_utils/core/ai.py`
