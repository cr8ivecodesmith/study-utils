# Transcribe-Video Config — Implementation

## Understanding
- Add a TOML-backed config layer to `transcribe_video` with a unified parser and a nested `config init|validate|path` command group that mirrors RAG's ergonomics, while keeping dataclasses **inline** (~100 new lines) rather than extracting a separate module. CLI flags override config values; when no config file is present the behavior matches today.
- The config location resolves as: explicit `--path` argument > `$STUDY_TRANSCRIBE_CONFIG` env var > default workspace config directory (`$STUDY_UTILS_DATA_HOME/config/transcribe.toml`).
- All existing CLI flags and functions stay intact — no breaking changes to the current transcribe-video flow. The `_TRANSCRIBE_LLM` global variable is retained as a backward-compatible fallback (no removal).
- Risks & mitigations: adding subparsers to `main()` could affect how argparse dispatches when no subcommand is given — guarded by a backward-compat test that runs the normal transcribe flow unchanged. Template packaging must include `"transcribe_video/*.toml"` in `pyproject.toml` or the bundled template will be missing at runtime.

## Resources
### Project docs
- agents/tasks/patch/0005-transcribe-video-config/spec.md — scope, TOML structure, CLI subcommands, dataclass hierarchy, and constraints.
- agents/guides/engineering-guide.md — seam-first design, dependency injection, small focused function guidance.
- agents/guides/patterns-and-architecture.md — package layout rules, composition over inheritance.
- agents/guides/workflow.md — collaboration loop and implementation log requirements.
- agents/guides/styleguides.md — Ruff lint rules (line-length 80), Google docstrings, pytest conventions.
### Existing modules to reference
- `src/study_utils/transcribe_video.py` (~580 lines) — current CLI parser, hardcoded `_TRANSCRIBE_LLM`, `main()` entry point.
- `src/study_utils/rag/config.py` (~660 lines) — primary pattern reference: frozen dataclasses, `_require_*` validation helpers, builder functions, `resolve_config_path()`, `load_config()`, `config_template()`.
- `src/study_utils/core/config_utils.py` (~85 lines) — shared helpers including `merge_defaults()` used by all config services.
- `src/study_utils/core/config_templates.py` (~98 lines) — template registry; adds a third entry here for transcribe-video.
### External docs
- https://docs.python.org/3/library/tomllib.html — TOML parsing (Python 3.12 stdlib).
- https://docs.python.org/3/library/importlib.resources.html — packaged resource discovery for `transform.toml`.

## Impact Analysis
### Affected behaviors & tests
- **Config subcommands**: three new CLI paths (`config init`, `config validate`, `config path`) need unit and integration tests mirroring RAG: overwrite protection, env resolution, error messaging on missing/invalid files.
- **main() flow**: the transcribe-video entry point dispatches config subcommands before entering normal mode; existing behavior with no config file present remains identical (hardcoded fallback). Covered by a backward-compat test that runs `transcribe-video <target>` without any config file.
- **CLI flag precedence**: every CLI flag tested verifies it overrides the corresponding config value — e.g. `--recursive` on CLI sets `recursive=False` even when config has `recursive = true`. Tests exercise both layers simultaneously.
- **Existing name cache**: `.transcribe_video_names.json` logic is unchanged; its path resolution now respects an optional `names.cache_path` config key. One test verifies the override works.
- **Template registration**: `config_templates.get_template("transcribe")` must resolve and read the packaged file — covered in the template registration step tests.
### Affected source files
- Modify: `src/study_utils/transcribe_video.py` — add dataclasses, `_DEFAULT_SETTINGS`, validation helpers, builder functions, path resolution, load config, subcommand handlers, unified parser. ~100 new lines, zero deletions.
- Create: `src/study_utils/transcribe_video/transform.toml` — standalone TOML template (~30 lines).
- Modify: `src/study_utils/core/config_templates.py` — register `"transcribe"` entry in `_TEMPLATES`.
- Modify: `pyproject.toml` — add `"transcribe_video/*.toml"` to `[tool.setuptools.package-data]`.
- Create: `tests/test_transcribe_config.py` — new standalone test file (~30 tests) covering validation, merge defaults, path resolution, subcommand dispatch, CLI overrides. No changes to existing `test_transcribe_video_extra.py`.
### Security considerations
- Config files are local-only; no secrets stored in TOML (API keys remain via env through `load_client()`). Template writes use 0o600 permissions via the shared helper. Error messages print paths and values but never leak API keys or token contents.

## Solution Plan
- Architecture/pattern choices: follow RAG's config pattern for validation and builder functions (frozen dataclasses, `_require_*` validators, `merge_defaults()` from core). Use lightweight template registration already established in `config_templates.py`. The unified parser uses subparsers — the existing top-level behavior is preserved as the default.
- DI & boundaries: keep I/O (file reads/writes, TOML loading) as thin boundary functions; core validation (`_parse_config`) is pure dict → dataclass and trivially testable without the filesystem. Config resolution depends on `core.config_utils` and `core.workspace` — already imported in transcribe_video today.
- Stepwise checklist:

**Step 1 — Dataclasses + constants (transcribe_video.py, ~25 lines)**
- [ ] Add `$DEFAULT_CONFIG_PATH = config_path("transcribe")` from `core.config`.
- [ ] Add `CONFIG_PATH_ENV = "STUDY_TRANSCRIBE_CONFIG"` constant.
- [ ] Add four frozen dataclasses: `AIConfig`, `AudioConfig`, `NamesConfig`, `LoggingConfig` plus `TranscribeConfig` composing the four fields into a single config type with additional top-level attributes (`target_dir`, `output_dir`, `prefix`, etc.).

**Step 2 — Validation helpers (transcribe_video.py, ~35 lines)**
- [ ] Port validation helpers from `rag/config.py`: `_require_positive_int`, `_require_non_negative_int`, `_require_bool`. These raise a descriptive exception (`ConfigError`) on type/value mismatch, mirroring RAG's behavior.

**Step 3 — Builder functions + defaults tree (transcribe_video.py, ~40 lines)**
- [ ] Add `default_tree() -> dict` returning a deep copy of `_DEFAULT_SETTINGS` so callers may mutate without affecting subsequent calls.
- [ ] Add `merge_defaults(tree, toml_data)` from `core.config_utils.merge_default`; update to handle nested table structures for the config sections if needed.
- [ ] Implement `_build_config(toml_data) -> TranscribeConfig` that combines all section builders and adds CLI-level attributes (output_dir, prefix, segment_duration_minutes, recursive).
- [ ] Implement `config_template() -> str` returning template content via `importlib.resources`.

**Step 4 — Path resolution + load_config (transcribe_video.py, ~30 lines)**
- [ ] Implement `resolve_config_path(explicit_path=None) -> Path` following RAG: explicit > env var ($STUDY_TRANSCRIBE_CONFIG) > default workspace config dir. Accepts absolute or relative path, resolves it via the workspace.
- [ ] Implement `load_config(config_dir=None, explicit_path=None) -> TranscribeConfig` that reads TOML with `tomllib`, calls `merge_defaults` then `_build_config`. Handles file-not-found by falling back to defaults (same behavior as today).

**Step 5 — Config subcommand handlers (~40 lines)**
- [ ] Add `_handle_init(args)` — resolve path, write template if not exists, handle --force flag, print destination path. Return exit code 0 on success, 2 on error. Mirrors RAG's init semantics exactly.
- [ ] Add `_handle_validate(args)` — load config (explicit or resolved), validate and print key settings; return 0/2 with appropriate messages.
- [ ] Add `_handle_path(args)` — resolve and print the expected config path only. Returns 0 on success, 1 if resolution fails, 2 if invalid TOML.

**Step 6 — Unified parser + main() dispatch (~40 lines)**
- [ ] Create `_build_parser() -> argparse.ArgumentParser` with subparsers for config subcommands (init, validate, path) and the default transcribe flow as the fallback when no subcommand is provided. Keep positional TARGET support in the default mode. The existing `--output`, `--prefix`, and other flags are inherited by both config and non-config paths for compatibility.
- [ ] Update `main()` to call `_build_parser()`, dispatch config subcommands before entering normal transcribe flow. Load config lazily so main path avoids unnecessary work when no --config flag is used.

**Step 7 — Template file + packaging (~10 lines total)**
- [ ] Create `src/study_utils/transcribe_video/transform.toml` with all default key-value pairs reflecting current hardcoded values.
- [ ] Register in `core/config_templates.py` `_TEMPLATES`: add entry with name="transcribe", template_filename="transform.toml". Follow the existing pattern (`name`, `template_filename`, and either a factory function or path attributes).
- [ ] Update `pyproject.toml` — add `"transcribe_video/*.toml"` to `[tool.setuptools.package-data]`.

**Step 8 — Tests (tests/test_transcribe_config.py, ~30 tests)**
- [ ] Template: `config_template()` returns valid TOML; template content parses into a dict with all expected keys.
- [ ] Write/read: `write_template(honours overwrite, writes correct paths and permissions`.
- [ ] Path resolution: explicit > env var > default workspace config dir (three-way precedence).
- [ ] Deep copy: mutating `default_tree()` result does not affect the next call.
- [ ] Merge defaults + validation for each section separately (8+ test cases): AI config with negative int, wrong bool; names with invalid path type; logging with unrecognized level string.
- [ ] Load_config happy path and error paths with invalid TOML values.
- [ ] Config subcommand handlers: init, validate, path — including `--path` overrides and force flags (mirrors RAG's integration tests).
- [ ] Backward-compat test: run transcribe flow without any config file present; config resolution gracefully falls back to defaults, `_parse_transcribe_args()` works unchanged.

**Step 9 — Verification**
- [ ] Run `uv run pytest tests/test_transcribe_config.py` independently → all pass with coverage on new code.
- [ ] Run `uv run pytest tests/` full suite → no regression (verify existing `test_transcribe_video_extra.py` still passes).
- [ ] Run `uv run ruff check src/study_utils/transcribe_video.py src/study_utils/core/config_templates.py` → lints clean, no changes needed to coverage.xml baseline.

## Test Plan
### Unit (tests/test_transcribe_config.py — ~30 tests)
- **Template**: `config_template()` returns valid TOML with all section keys; template content renders as a dict via `tomllib.loads()`.
- **Write/Read**: `write_template_path` honors `--force` (no overwrite without it); writes correct TOML content; respects 0o600 permissions.
- **Path resolution**: three-tier precedence verified — explicit argument wins > env var `STUDY_TRANSCRIBE_CONFIG` wins > default workspace config dir. Absolute and relative paths both tested.
- **Deep copy defense**: mutating `default_tree()` result does not affect the next call's return value.
- **Merge defaults + validation**: `merge_defaults(_DEFAULTS.copy(), toml_data)` — TOML values override defaults while missing keys keep their defaults; unknown keys raise `TomlConfigError`/`ConfigError`. Each section validated individually (8+ test methods covering AI, audio, names, logging with correct and incorrect types).
- **Config loading**: happy path loads all fields correctly from a multi-table TOML file; error path when `segment_duration_minutes` is non-positive, `smart_names` is not boolean, or `api_base` has wrong type.
- **Builder functions**: each `_build_*()` called with correct section → returns dataclass with expected values; missing required keys raise `ConfigError`; wrong types raise `ConfigError`.
### Integration / E2E
- CLI test: `study transcribe-video config init --path <tmp>/x.toml` → file exists; run again without `--force` → error printed to stderr, exit code 2; with `--force` → success message and overwrite.
- CLI test: `study transcribe-video config validate` on valid config → prints "Configuration OK" + key values, exit 0.
- CLI test: `study transcribe-video config validate --path <bad.toml>` where bad.toml has an invalid type or negative int → error printed, exit code 2.
- CLI test: `study transcribe-video config path` with `STUDY_TRANSCRIBE_CONFIG` env override set → prints the resolved path.
- Backward-compat: run without any config file; existing `_parse_transcribe_args()` still works; no exceptions raised from new config code paths when they are skipped.
### Regression scope (tests/test_transcribe_video_extra.py)
- No source changes to `test_transcribe_video_extra.py`. All ~20 existing tests validate function-level behavior that is independent of the config layer. One integration test validates end-to-end CLI entry point (`argparse` path through `_build_parser()`).

## Sandbox convention (strong adherence)
When running simulated tests for this feature, use the project-level `sandbox/` folder and create subfolders within it for each command or test scenario. This ensures consistent, predictable paths during tool testing rather than relying on temp directories.

## Operability
- **Logging**: existing stdout prints remain unchanged; new config validation errors print to stderr via `ConfigError` (consistent with RAG). Logging level is a config key applied when the transcribe flow starts.
- **Runbooks**: if `transcribe-video` behaves unexpectedly: (1) run `study transcribe-video config path` to verify resolution; (2) run `study transcribe-video config validate` to inspect values; (3) inspect file content at the resolved path; (4) rerun with `config init --force` to reset to defaults.
- **Revert steps**: drop all new inline additions from `transcribe_video.py` (~100 lines removed, ~5 modified on edges), delete `transform.toml`, remove registration from `config_templates.py`, remove entry from `pyproject.toml`. Existing `_TRANSCRIBE_LLM` global remains untouched; the original `main()` parse-and-run flow is fully restored. No test file changes needed (new tests can remain as unused-or-be-dropped).

## History
### 2026-08-13 — Initial implementation plan
**Summary** — Established understanding, impact analysis, solution outline with 9 stepwise checklist items, detailed unit/integration/regression test plans, and operability notes for adding TOML config to transcribe-video via inline dataclasses and a `config init|validate|path` CLI group.
