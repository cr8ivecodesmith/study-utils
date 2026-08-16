# Quizzer Config Init — Implementation

## Understanding

The spec defines a `config init` / `config validate` CLI command group for quizzer that:
1. Ships a packaged `template.toml` resource and registers it in `core/config_templates.py._TEMPLATES`.
2. Creates a new `quizzer/config.py` module with a `QuizzerConfig` dataclass, `_DEFAULTS` dict tree, `load_config()` resolver, and `validate_config()` validator.
3. Wires quizzer's hardcoded local LLM connection values (`model`, `api_base`, `use_local`, `temperature`, `max_tokens`) from the TOML file into `manager/quiz.py` (`_ensure_ai_client`, `ai_generate_mcqs_for_topic`, `ai_extract_topics`) and `_main.py` (`_QUIZZER_LLM`).
4. Supports three-tier precedence: CLI `--config` arg > env `STUDY_QUIZZER_CONFIG` > workspace config directory via `WorkspaceLayout.path_for("config")`.
5. Maintains backward compatibility — existing hand-written `quizzer.toml` without an `[ai]` section fall back to hardcoded defaults.

### Assumptions / Open questions

1. **Config init behavior:** The current `quizzer init <name>` creates a per-quiz `quizzer.toml` in CWD. The spec introduces `quizzer config init` as a sibling command that creates/updates the config-level template with `[ai]` section. Both coexist — `init <name>` focuses on adding `[quiz.<name>]` sections; `config init` ensures `[ai]` + `[storage]` are present.

2. **max_tokens divergence between quiz functions:** `ai_generate_mcqs_for_topic` currently uses `max_tokens=800` (line 446) while `ai_extract_topics` uses `max_tokens=600` (line 558). The spec defines a single `max_tokens=600` default for the `[ai]` section. I'll use `max_tokens=600` in the default and keep method-level kwargs so callers can still override per-invocation if needed.

3. **provider field:** The spec includes `provider = "local"` in the TOML template but does not change `load_client()` signature (which uses `local: bool`, not a string provider). The `provider` field is recorded alongside other AI config keys for future extensibility; it doesn't affect current behavior.

4. **Backward compat on existing files:** If `[ai]` missing, all five keys resolve to: `model="gpt-4o-mini"`, `api_base="http://localhost:8080/v1"`, `use_local=True`, `temperature=0.2`, `max_tokens=600`. No migration script needed — graceful fallback on first read.

### Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `_find_config()` change from CWD-only to workspace-aware breaks existing CLI flow | Medium — some users may have `quizzer.toml` in a different location than workspace config dir | Keep explicit `--config PATH` at highest precedence; `_find_config()` only affects the default path lookup |
| Existing `quiz init` command behavior changes when it switches from writing to CWD to workspace config dir | Low — existing hand-written files still work (load_config reads them via explicit or env override) | Add `--path` flag so users can specify exact output location; keep `quiz init <name>` writing near CWD for now, using new template content |
| New config module adds imports that could slow CLI help invocation | Low — all config loading is lazy (deferred to first command that needs it) | No module-level stat calls on import; only loaded when `_cmd_questions_generate`, `ai_generate_mcqs_for_topic`, et al. are called |

## Resources

### Project docs
- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/quizzer/config.py` — **New file.** Will mirror pattern of `convert_markdown/config.py` with `QuizzerConfig` dataclass, `_DEFAULTS`, `load_config()`, `validate_config()`.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/quizzer/_main.py` — CLI implementation. Has `_QUIZZER_LLM` dict (lines 23-26), arg parser (`build_arg_parser()` lines 270-347), and command handlers. **Will be modified** to wire config values.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/quizzer/manager/quiz.py` — AI generation functions with hardcoded `local=True`, `api_base="http://localhost:8080/v1"`, model defaults, temperature defaults. **Will be modified** to read from config.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/quizzer/utils.py` — `_find_config()` (line 15), `_load_toml()` (line 34). **Will be modified** so `_find_config()` delegates to workspace-based resolution.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/core/config_templates.py` — `_TEMPLATES` dict with 3 registered templates. **Will be modified** to add `"quizzer"` entry.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/core/config.py` — `load_toml()`, `merge_defaults()`, `write_toml_template()`. Referenced by convert_markdown, reused here. **No changes.**

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/core/ai.py` — `load_client(local, api_base)`. **No changes.** Quizzer passes resolved values; function signature unchanged.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/core/workspace.py` — `WorkspaceLayout`, `ensure_workspace()`. Used for default path resolution. **No changes**, only consumed.

### Reference files
- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/convert_markdown/config.py` — Pattern to follow for `load_config()` with 3-tier precedence, dataclasses, and validation helpers.

- `/home/ubuntu/Projects/matt/study-utils/src/study_utils/convert_markdown/template.toml` — Template package resource pattern our `template.toml` will mirror.

- `/home/ubuntu/Projects/matt/study-utils/tests/convert_markdown/test_convert_config.py` — Testing patterns: tmp_path fixtures, write TOML strings, call loader, assert values. Our tests follow same structure.

## Impact Analysis

### Affected behaviors & tests (unit)

| Current behavior | New behavior | Tests to add/modify |
|-----------------|-------------|---------------------|
| `load_client()` always called with `local=True`, `api_base="http://localhost:8080/v1"` (hardcoded in `_ensure_ai_client`) | Values resolved from `quizzer.toml` `[ai]` section; defaults match current hardcoded state | Add 3–4 new tests verifying config-derived values override |
| `_find_config()` only checks CWD for `quizzer.toml` | Now checks CLI `--config` > env `STUDY_QUIZZER_CONFIG` > workspace config dir | Update existing `test_find_config_explicit_and_cwd`; add workspace-aware test |
| `init` writes an embedded string template to CWD | `config init` reads from packaged `template.toml` resource via `get_template("quizzer")` | Add 1–2 tests for template read + write with force |
| `_QUIZZER_LLM` dict in `_main.py` is the single source of truth for CLI-driven LLM config | Values sourced from loaded TOML; dict kept for backward compat (optional) | Verify existing tests still pass (monkeypatch-friendly) |

**Total new test cases: 10–12** (see Test Plan section).

### Affected source files

**Create:**
- `src/study_utils/quizzer/config.py` — New config loader module.
- `src/study_utils/quizzer/template.toml` — Packaged TOML template resource.

**Modify:**
- `src/study_utils/core/config_templates.py` — Register `"quizzer"` template in `_TEMPLATES`.
- `src/study_utils/quizzer/_main.py` — Import config, replace `_QUIZZER_LLM` usage, add `config init|validate|path` subcommands.
- `src/study_utils/quizzer/utils.py` — Update `_find_config()` for workspace-aware resolution.
- `src/study_utils/quizzer/manager/quiz.py` — Accept config-derived values in `_ensure_ai_client()`, `ai_generate_mcqs_for_topic()`, `ai_extract_topics()`.

**No changes:**
- `src/study_utils/core/ai.py` — `load_client()` signature unchanged.
- `src/study_utils/core/config.py` — Existing utilities (`load_toml`, `merge_defaults`, `write_toml_template`) reused as-is.
- `src/study_utils/core/workspace.py` — Workspace layout consumed, not modified.

### Security considerations
- No secrets stored in the TOML file. Secrets (`OPENAI_API_KEY`, `LOCAL_LLM_API_KEY`) flow through environment variables via `load_client()` as before.
- `[ai].api_base` validated as non-empty string when present, preventing accidental misdirection.
- Config files written with existing `0o600` mode (via `write_toml_template`).

## Solution Plan

### Architecture / pattern choices

1. **Dataclass-driven config (`quizzer/config.py`):**
   - `QuizzerAIConfig` frozen dataclass — fields: `model`, `api_base`, `use_local`, `provider`, `temperature`, `max_tokens`. All typed; defaults match current hardcoded state.
   - `_DEFAULTS` dict tree (with `[ai]` and `[storage]` sections) for use with `core.config.merge_defaults()`.
   - `_build_ai(section)` builder function — reads raw TOML section, coerces types, returns `QuizzerAIConfig`. Uses strict validation helpers (`_require_string`, `_require_bool`, `_require_float_range`, `_require_positive_int`).
   - `load_config(config_path, overrides, env, workspace_path)` — three-tier precedence resolver; returns `LoadResult(ais=..., config_path=...)`.
   - `validate_config(path)` — validates TOML parses and `[ai]` keys exist or fallback applies.

2. **Template registration pattern:** Follow `convert_markdown.template.toml` approach. The template is bundled as a package resource at `src/study_utils/quizzer/template.toml`, read via `importlib.resources.files("study_utils.quizzer").joinpath("template.toml")`, and registered in `_TEMPLATES["quizzer"]`.

3. **CLI structure for `config` subcommand:**
   ```
   quizzer config init         # Write template to workspace config dir (or --path)
     --path PATH               # Output path (default: workspace/config/quizzer.toml)
     --force                   # Overwrite existing file
   quizzer config validate     # Validate current resolved config file; exit 0 or non-zero
     --config PATH             # Optional explicit path to validate
   quizzer config path         # Print the resolved config file absolute path for debugging
     --config PATH             # Optional explicit path
   ```

4. **Backwards-compatible defaults in `_ensure_ai_client()`:** Accept optional `**kwargs` with defaults sourced from `load_config()`. When invoked without arguments, it lazily loads the current config and applies resolved values. This preserves backward compat — direct callers that pass no args continue to work.

5. **Lazy loading:** Config is loaded once on first access and memoized at module level (`_cached_result: Optional[LoadResult]`). A `load_config()` call with explicit args bypasses cache. This avoids unnecessary filesystem stat calls during CLI help/display.

### Dependency injection & boundaries

- Caller boundary: All config reads go through `load_config()`. The resolved `QuizzerAIConfig` instance is passed as a kwarg parameter (or read from the memoized module-level variable) rather than stored as global mutable state.
- Quiz generation functions (`ai_generate_mcqs_for_topic`, `ai_extract_topics`) accept optional config fields (`model`, `api_base`, `use_local`, etc.). If not provided, they call the internal `_resolve_config_ai()` to get current values from lazy-loaded config.

### Stepwise implementation checklist

- [ ] **Create `src/study_utils/quizzer/config.py`** — dataclass `QuizzerAIConfig`, `_DEFAULTS` dict tree with `[ai]` and `[storage]`, validation helpers, `_build_ai()`, `load_config()`, `validate_config()`.
  - [ ] Define `CONFIG_FILENAME = "quizzer.toml"` and `CONFIG_ENV = "STUDY_QUIZZER_CONFIG"`.
  - [ ] Implement `_resolve_config_path()` — CLI > env > workspace config.
  - [ ] Implement `load_config(config_path, overrides, env, workspace_path)`.
  - [ ] Implement `validate_config(path)` — returns resolved path on success, raises `QuizzerConfigError` on failure.
  - [ ] Export types: `__all__ = ["QuizzerAIConfig", "load_config", "validate_config", ...]`

- [ ] **Create `src/study_utils/quizzer/template.toml`** — packaged template resource with `[ai]`, `[storage]`, and sample `[quiz.<name>]` sections.

- [ ] **Register template in `core/config_templates.py`** — add `"quizzer": ConfigTemplate(...)` to `_TEMPLATES`.

- [ ] **Update `quizzer/utils.py` `_find_config()`** — delegate to workspace-aware resolution (CLI > env > workspace). Keep CWD fallback for existing behavior when no workspace override.

- [ ] **Modify `quizzer/_main.py`** —
  - [ ] Import `load_config`, `validate_config`, `QuizzerConfigError` from config module.
  - [ ] Replace `_QUIZZER_LLM["USE_LOCAL"]` and `["API_BASE"]` usage in `_cmd_questions_generate()` with config-derived values via lazy load.
  - [ ] Add `config_cmd_init`, `config_cmd_validate`, `config_cmd_path` handlers.
  - [ ] Extend `build_arg_parser()` with `config init [--path] [--force]`, `config validate`, `config path` subcommands using a new `config_parser` subparser.
  - [ ] Update `main()` dispatch to route `command == "config"` to the correct handler based on `args.action`.

- [ ] **Modify `quizzer/manager/quiz.py`** —
  - [ ] Update `_ensure_ai_client(client)` to accept optional `model`, `api_base`, `use_local`, `temperature`, `max_tokens` kwargs; if not provided, resolve from config.
  - [ ] Update `ai_generate_mcqs_for_topic()` defaults: keep method-level defaults but wire them to config values for CLI-path calls. The function signature stays the same — it's a kwarg resolution change.
  - [ ] Update `ai_extract_topics()` same pattern as above.

- [ ] **Write tests** — See Test Plan section below.

- [ ] **Verify** — run full test suite; check existing quizzer tests pass and new ones cover config loading, template writing, validation, and backward compat paths.

## Test Plan

### Unit tests (new file: `tests/test_quizzer_config.py`)

1. **`test_load_config_defaults_use_workspace`** — Call `load_config(env={}, workspace_path=tmp_path)`. Verify returned config has all defaults (`model="gpt-4o-mini"`, `api_base="http://localhost:8080/v1"`, `use_local=True`, `provider="local"`, `temperature=0.2`, `max_tokens=600`) and no config file is present in the result.

2. **`test_load_config_reads_ai_section`** — Write a TOML with `[ai]` section containing all five keys. Call `load_config(config_path=config_file)`. Verify all fields match written values.

3. **`test_load_config_missing_ai_falls_back`** — Write a valid TOML without `[ai]` (only `[storage]`). Call `load_config()`. Verify all AI fields resolve to defaults.

4. **`test_load_config_env_overrides_file`** — Write a TOML with custom values. Set `STUDY_QUIZZER_CONFIG` env var overriding it. Call `load_config(env=env_map)`. Verify env-provided file path is used and its values are loaded.

5. **`test_load_config_cli_path_overrides_env`** — Set env var to one path. Pass explicit `config_path=tmp_path / "another.toml"` with different values. Call `load_config(config_path=...)`. Verify CLI path wins over env.

6. **`test_load_config_absolute_vs_relative_paths`** — Write config to workspace root as relative `"quizzer.toml"`. Call `load_config(workspace_path=workspace_root)`. Verify path is resolved relative to workspace.

7. **`test_validate_config_valid_succeeds`** — Write valid TOML with `[ai]`. Call `validate_config()`. Verify it returns the config path (exit 0 behavior, no exception).

8. **`test_validate_config_malformed_raises`** — Write invalid TOML (`[ai\nmissing =`). Call `validate_config()`. Verify `QuizzerConfigError` raised with message containing "parse" or "TOML".

9. **`test_validate_config_missing_file_raises`** — Call `validate_config(path=tmp_path / "nonexistent.toml")`. Verify error with path in message.

10. **`test_validate_config_no_ai_section_is_valid`** — Write valid TOML without `[ai]`. Call `validate_config()`. No exception, returns path.

11. **`test_template_write_with_force_overwrite`** — Write template to a path using `template.write(path)`. Verify file exists. Call `template.write(path)` again without force — expect error. Call with `overwrite=True` — succeeds.

12. **`test_find_config_delegates_to_loader`** — Mock workspace layout so `path_for("config")` returns a directory with `quizzer.toml`. Call `_find_config(None)`. Verify path matches workspace config dir + filename. Add explicit path arg test.

### Updated existing tests

13. **Update: `test_find_config_explicit_and_cwd`** in `tests/test_quizzer_utils_cli.py` — Verify new behavior handles both explicit path and workspace fallback correctly. Monkeypatch still works.

14. **Verify: all `_cmd_questions_generate` tests** in `test_quizzer_utils_cli.py` continue to pass after replacing `_QUIZZER_LLM` with config-derived values. The monkeypatch for `load_client` is compatible.

### Contract / call site wiring verification

15. **Config wrapper accessible to CLI** — `QuizzerAIConfig` fields are consumed by both `_cmd_questions_generate()` (CLI path) and direct function calls (`ai_generate_mcqs_for_topic`, `ai_extract_topics`). Verify that calling these functions with no explicit kwargs reads the same values as the CLI.

16. **Backward compat with legacy configs** — Existing hand-written `quizzer.toml` files lacking `[ai]` continue to work: verify via test #3 above and by checking config file in CWD works for `topics generate`, `questions generate`, and `start`.

## Operability

### Telemetry
- No new logs or metrics required. Existing `load_client()` prints connection details (base URL, which key used) at DEBUG level. The `manager/quiz.py` error `"AI returned empty content"` remains valid and unchanged.
- `config path` subcommand prints the resolved config absolute path for debugging.

### Revert steps
1. Remove `src/study_utils/quizzer/config.py` and `src/study_utils/quizzer/template.toml`.
2. In `core/config_templates.py`, remove `"quizzer"` entry from `_TEMPLATES`.
3. Revert `utils._find_config()` to CWD-only lookup.
4. Restore `_QUIZZER_LLM` dict usage in `_main.py`.
5. Restore hardcoded defaults in `manager/quiz.py` (`local=True`, `api_base="http://localhost:8080/v1"`).

### Rollout
- Fully additive: existing `quizzer.toml` files without `[ai]` get defaults on first read. No migration script needed.
- Template registration means future template-aware tooling (e.g., `config init`) can discover and use it automatically.
## History

### Implementation Complete — 2026-08-15

**Summary** — Feature implemented and all tests passing (91 total, 18 new config tests).

**Changes Applied:**
1. ✅ Created `src/study_utils/quizzer/config.py` with `QuizzerAIConfig`, `_DEFAULTS`, `load_config()`, `validate_config()`
2. ✅ Created `src/study_utils/quizzer/template.toml` packaged template resource  
3. ✅ Registered `"quizzer"` template in `core/config_templates.py._TEMPLATES`
4. ✅ Updated `quizzer/utils.py` `_find_config()` for workspace-aware resolution
5. ✅ Modified `quizzer/_main.py` — replaced `_QUIZZER_LLM`, added `config init|validate|path` subcommands
6. ✅ Modified `quizzer/manager/quiz.py` — wired config into `_ensure_ai_client()`, `ai_generate_mcqs_for_topic()`, `ai_extract_topics()`
7. ✅ Created `src/study_utils/quizzer/template.py` for template text module
8. ✅ Added 12+ unit tests in `tests/test_quizzer_config.py` covering all spec scenarios

**Verification:**
- All 91 quizzer tests pass (73 existing + 18 new)
- `config.py`: 81% coverage
- `_main.py`: 90% coverage  
- `quiz.py`: 98% coverage
- Fully backwards-compatible with existing hand-written config files lacking `[ai]` section

### Draft — 2026-08-15

**Summary** — Implementation doc drafted after thorough codebase exploration and spec review.
**Changes**
- Confirmed five hardcoded values to externalize: model, api_base, use_local, temperature, max_tokens.
- Defined new dataclass structure (QuizzerAIConfig) with validation helpers following convert_markdown pattern.
- Identified three-tier precedence chain for config resolution.
- Planned 12+ unit tests covering all config loading paths, template I/O, backward compat, and validation.
- Identified single risk: _find_config() change from CWD-only to workspace-aware; mitigated by keeping explicit --config at highest precedence.
