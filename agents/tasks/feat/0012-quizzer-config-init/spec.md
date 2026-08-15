# Quizzer Config Init — Spec

## Summary

Add a `config init` / `config validate` CLI command group to the quizzer service that
creates an opinionated `quizzer.toml` template file and wires quizzer's hardcoded local
LLM connection values into the config. Replace all inlined defaults (model, api_base,
use_local, temperature, max_tokens) with config-derived values so that the TOML file
becomes the single source of truth.

## Goals

- Users can scaffold a `quizzer.toml` via `quizzer config init`, mirroring the patterns
  used by `convert-markdown config init` (0007) and `generate-document config init` (0011).
- Replace all hardcoded AI connection values in `manager/quiz.py` (`_ensure_ai_client`,
  `ai_generate_mcqs_for_topic`, `ai_extract_topics`) and `_main.py` (`_QUIZZER_LLM` dict)
  with values resolved from config.
- Support both a workspace-scoped lookup (workspace config directory via
  `study_utils.core.workspace.WorkspaceLayout.path_for("config")`) and an explicit
  `--config PATH` override.
- Environment variable `STUDY_QUIZZER_CONFIG` provides another injection point (same
  pattern as RAG and convert-markdown).
- Fully backwards-compatible: existing hand-written `quizzer.toml` files that lack
  an `[ai]` section fall back to the same defaults (`gpt-4o-mini`,
  `http://localhost:8080/v1`, `use_local=true`).

## Non-Goals

- No changes to RAG's own config (`rag/config.py`) or to `core/ai.py`'s `load_client()` —
  quizzer calls `load_client(local=..., api_base=...)` as before, only the *values* come
  from TOML instead of inline.
- No support for per-quiz-section `[ai]` overrides (e.g., each quiz can have a different
  model yet); all quizzes share the same global AI config.
- `config validate` validates **only** the config file structure and required keys — it does
  not inspect per-quiz sections or question/topics schemas.
- No TOML migration script for legacy configs; existing files simply gain missing keys with
  sensible defaults on first read.

## Behavior (BDD-ish)

- Given a `quizzer.toml` file exists in the workspace config directory, **then**
  `load_config()` resolves it through the standard resolution order:
  CLI `--config` arg > env `STUDY_QUIZZER_CONFIG` > workspace config dir.

- Given a `quizzer.toml` with an `[ai]` section containing all five keys
  (`model`, `api_base`, `use_local`, `temperature`, `max_tokens`), **then**
  quiz generation calls use those values for both question and topic extraction.

- Given a `quizzer.toml` file that is valid TOML but missing the `[ai]` section,
  **then** defaults match the previous hard-coded values:
  `model="gpt-4o-mini"`, `api_base="http://localhost:8080/v1"`,
  `use_local=True`, `temperature=0.2`, `max_tokens=600`.

- Given a user runs `quizzer config init --path ./my/quiz.toml --force`, **then**
  the template is written to the absolute path and any existing file is overwritten.

- Given a user runs `quizzer config validate`, **then** the CLI exits 0 with success
  when the file parses as valid TOML and contains at least `[ai]` or falls back,
  and exits non-zero with a message when the file is missing or malformed.

## Configuration: `quizzer.toml` (AI section)

```toml
[ai]
model = "gpt-4o-mini"
api_base = "http://localhost:8080/v1"
use_local = true
provider = "local"
temperature = 0.2
max_tokens = 600

# Existing per-quiz and storage sections — unchanged by this spec, but the config
# loader also reads them so quiz commands can continue using them.
[storage]
out_dir = ".quizzer/<name>"
```

## Constraints & Dependencies

- **Python version:** Python 3.11+ (`tomllib` is available in stdlib).
- **TOML library:** Uses the same `tomllib` / `tomli` fallback already present in
  `quizzer/utils.py` `_load_toml()`.
- **`load_client()`** at `src/study_utils/core/ai.py` — no changes required. Quizzer
  passes `local=` and `api_base=` values; only the *provenance* of those values changes.
- **Workspace support:** Reuses `study_utils.core.workspace.WorkspaceLayout` (already
  used by other study-utils services) so config resolution is consistent across features.
- **TOML template registration:** `quizzer` template is registered in
  `core/config_templates.py._TEMPLATES` alongside `convert_markdown` and `generate_document`.

## Security & Privacy

- No secrets are stored in the TOML file. Secrets (`OPENAI_API_KEY`, `LOCAL_LLM_API_KEY`)
  flow through environment variables via `load_client()` as before.
- The `[ai].api_base` value is used as-is (no shell injection). The value is validated
  to be a non-empty string when present.

## Telemetry & Operability

- Existing error messages from `manager/quiz.py` (e.g., empty AI responses) remain valid.
- `config path` subcommand prints the resolved config file absolute path for debugging.
- Logging and verbose flags are unaffected, since they belong to quizzer's broader CLI
  configuration in `_main.py`.

## Rollout / Revert

- No runtime migration required: TOML parsing gracefully falls back to hard-coded defaults
  when `[ai]` keys are missing.
- To revert: remove the `config.py` loader and the registered template, and restore the
  original inline values in `manager/quiz.py` and `_main.py`.

## Definition of Done

- [ ] `quizzer/config.py` — dataclass `QuizzerConfig`, `_DEFAULTS` dict tree,
      `load_config()`, `validate_config()`.
- [ ] `quizzer/template.toml` shipped as package resource.
- [ ] `"quizzer"` template registered in `core/config_templates.py._TEMPLATES`.
- [ ] `manager/quiz.py` — `_ensure_ai_client()`, `ai_generate_mcqs_for_topic()`,
      and `ai_extract_topics()` all read model, api_base, use_local, temperature,
      and max_tokens from config.
- [ ] `_main.py` — `_QUIZZER_LLM` dict usage replaced; template string uses config values.
- [ ] CLI extended with `config init`, `config validate`, `config path` subcommands
      (and wiring through `--path`, `--workspace`, `--force`).
- [ ] `utils.py` — `_find_config()` delegates to config loader resolution.
- [ ] Tests added/updated: unit tests for config parsing, template writing, override logic,
      and backward compat with missing `[ai]` section.

## Ownership

- Owner: @matt
- Reviewers: To be assigned
- Stakeholders: Study CLI users who rely on local LLM in quizzer

## Links

- Related: `feat/0011-docs-generator-local-llm-config` (generate-document TOML config)
- Related: `feat/0010-per-service-local-llm` (RAG's per-service local LLM config)
- Reference: `load_client()` at `src/study_utils/core/ai.py`
- Reference: convert-markdown config at `src/study_utils/convert_markdown/config.py`
