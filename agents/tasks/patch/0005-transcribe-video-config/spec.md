# Transcribe-Video Config — Spec (Mini)

## Description

Give `transcribe-video` a TOML-backed config system with a `config init/validate/path` command group so that AI settings, audio chunk size, name generation defaults, and logging are no longer hard-coded.

## Goals

- Introduce a `study transcribe-video config` command group (`init`, `validate`, `path`) mirroring the RAG service's ergonomics: write, inspect, or resolve the active configuration file.
- Promote all currently hard-coded settings from `_TRANSCRIBE_LLM`, `_parse_transcribe_args()`, and the module body into a validated dataclass hierarchy (`AIConfig`, `AudioConfig`, `NamesConfig`, `LoggingConfig`) that loads from a packaged TOML template via `core.config_templates`.
- Preserve existing CLI flags (`-o/-output-dir`, `-p/--prefix`, `-r/--recursive`, `--smart-names`, `--use-ai`, `--names-file`) as overrides on top of the loaded config — no breaking changes to current usage.
- Register a packaged `transcribe.toml` template in `study_utils.transcribe_video` (as `transform.toml`) and update `pyproject.toml` so it ships with the wheel.
- Bootstrap all config values lazily through `main()`; when no config file exists, fall back to today's hardcoded defaults transparently.

## Non-Goals

- No redesign of the names cache (` .transcribe_video_names.json`) — it stays target-root-relative; a new `cache_path` config key merely offers an optional override.
- No removal of `_TRANSCRIBE_LLM` global in this patch (it is retained as a backward-compat fallback).
- No migration of log file to the RAG-managed `logs/` directory; logging uses the existing stdout/stderr flow and respects the `verbose` config key.
- No new third-party dependencies or telemetry pipeline.

## Behavior (BDD-ish)

### Config Init

- Given any workspace layout, when the user runs `study transcribe-video config init`, then the command writes the packaged template to `$STUDY_UTILS_DATA_HOME/config/transcribe.toml` and prints the destination path.
- Given an existing `transcribe.toml` at the default location without `--force`, when the user reruns `config init`, then the command exits non-zero with a message explaining that the file exists and how to use `--force`.
- Given the CLI flag `--path ./custom.toml`, when the user runs `config init --path ./custom.toml --force`, then the template is written to the specified absolute or (workspace-relative) path regardless of the workspace layout.

### Config Validate

- Given a valid config file at the resolved path, when the user runs `study transcribe-video config validate`, then the command prints "Configuration OK" with key settings (model, chunk duration, smart_names, api_base) and exits 0.
- Given a config file with an invalid value (e.g., `segment_duration_minutes = -1`), when the user runs `config validate`, then the command prints a descriptive error and exits 2.

### Config Path

- Given no arguments, when the user runs `study transcribe-video config path`, then the command resolves the config location via: explicit `--path` > env `STUDY_TRANSCRIBE_CONFIG` > default `<workspace>/config/transcribe.toml` and prints it.

### Transcribe with Config

- Given a scaffolded config where `names.smart_names = true` and `audio.segment_duration_minutes = 15`, when the user runs `study transcribe-video /my/courses`, then smart names are enabled by default (but `--no-smart-names` on CLI disables them) and audio chunks use 15-minute segments.
- Given a config file with AI settings populated, when the user runs `transcribe-video <target>` with no flags overriding them, then the module uses the loaded values for model, title model, api_base, and use_local (instead of hardcoded or os.getenv defaults).
- Given explicit CLI flags, they take precedence over the config for their respective settings; settings not provided on CLI use the config value.

## Configuration Structure (TOML)

```toml
# Study Utils transcribe-video configuration template.
# Generated via `study transcribe-video config init`.

[ai]
use_local = true
api_base = "http://localhost:8080/v1"
provider = "local"
model = "whisper-3"
title_model = "gpt-4o-mini"

[audio]
# Segment duration in minutes (chunks audio before sending to Whisper)
segment_duration_minutes = 10

[names]
smart_names = true
use_ai_titles = false
# Output directory for transcripts. Null = write to current working directory.
output_dir = null
# Recursive search into subdirectories when target is a directory.
recursive = true
# Names cache path: null = auto under target root, or explicit file path.
cache_path = null

[logging]
level = "INFO"
verbose = false
```

## Dataclass Hierarchy (config.py — inline in transcribe_video.py)

```python
@dataclass(frozen=True)
class AIConfig:
    use_local: bool
    api_base: str
    provider: str
    model: str         # whisper transcription model
    title_model: str   # GPT model for smart title generation

@dataclass(frozen=True)
class AudioConfig:
    segment_duration_minutes: int  # chunk size in minutes

@dataclass(frozen=True)
class NamesConfig:
    smart_names: bool
    use_ai_titles: bool
    output_dir: Optional[Path]     # None = CWD
    recursive: bool
    cache_path: Optional[Path]     # None = auto under target

@dataclass(frozen=True)
class LoggingConfig:
    level: str      # DEBUG, INFO, WARNING, ERROR, CRITICAL
    verbose: bool

@dataclass(frozen=True)
class TranscribeConfig:
    ai: AIConfig
    audio: AudioConfig
    names: NamesConfig
    logging: LoggingConfig
```

## CLI Argument Precedence (highest → lowest)

1. **CLI flags** (`--output-dir`, `--prefix`, `--smart-names`, `--use-ai`, `--recursive`, `--names-file`)
2. **Environment variables** (`STUDY_TRANSCRIBE_CONFIG` for config path; `TRANSCRIPTION_MODEL`, `OPENAI_TITLE_MODEL` as env-level overrides)
3. **TOML config file** (`transcribe.toml`)
4. **Hardcoded defaults** (~the current `_TRANSCRIBE_LLM` dict and module constants)

## Files Changed / Created

| Action | File | Details |
|--------|------|---------|
| Modify | `src/study_utils/transcribe_video.py` | Add dataclasses, constants (`CONFIG_PATH_ENV`, `_DEFAULTS`), validation helpers (`_require_positive_int`, `_require_bool`, etc.), `_build_config()`, `resolve_config_path()`, `load_config()`, `config_template()`, `write_template()`, config subcommand handlers (`_handle_config_init/validate/path`), unified parser `_build_parser()` — ~100 new lines added/modified. |
| Create | `src/study_utils/transcribe_video/transform.toml` | Packaged TOML template (~30 lines) for the command group. |
| Modify | `src/study_utils/core/config_templates.py` | Register template in `_TEMPLATES`: `name="transcribe"`, `filename="transform.toml"`, `package="study_utils.transcribe_video"`. |
| Modify | `pyproject.toml` | Add `"transcribe_video/*.toml"` to `[tool.setuptools.package-data]`. |
| Create | `tests/test_transcribe_config.py` | Tests for config validation, merge defaults, path resolution, subcommand dispatch, CLI override behavior (~20-30 tests). |

## Implementation Steps

1. **Add constants & dataclasses** — `CONFIG_PATH_ENV`, `_DEFAULTS`, and all four config dataclasses + `TranscribeConfig`.
2. **Validation helpers** — Add `_require_positive_int`, `_require_bool`, `_require_float_range` (if needed), `_coerce_optional_string`, `_coerce_optional_path`, `_require_string`; reuse patterns from `rag/config.py`.
3. **Builder functions** — Implement `_build_ai()`, `_build_audio()`, `_build_names()`, `_build_logging()`, and the top-level `_build_config(tree)` that assembles all pieces from a TOML tree, using `core_config.merge_defaults()` for defaults resolution.
4. **Path & load** — Add `resolve_config_path(explicit_path, env)` and `load_config(explicit_path?, env?)` mirroring RAG's two-function pattern (`resolve` for location, `load` for validation + assembly).
5. **Config subcommand handlers** — Implement `_handle_config_init()`, `_handle_config_validate()`, `_handle_config_path()` in the same style as `rag/cli.py`.
6. **Unified parser** — Add `_build_parser()` with main transcription args + a config subparser (init/validate/path). Keep positional TARGET support for both modes.
7. **Wire into main()** — Modify main to dispatch config subcommands before entering the normal transcribe flow. Load config lazily; pass it through helpers so CLI flags can override.
8. **Create packaged template** — Write `transform.toml` as a static TOML file next to the module. Update `pyproject.toml`.
9. **Register template** — Update `core/config_templates.py` `_TEMPLATES` dict and verify `resources.files()` lookup works for the new template.
10. **Write tests** — Config validation edge cases, merge defaults with missing/extra keys (`merge_defaults` rejects unknown keys), path resolution (explicit > env > default), subcommand dispatch, backward compat without config file.
11. **Lint + typecheck + full suite** — Run `ruff check`, verify pytest coverage (target 100% for new config code), ensure existing `test_transcribe_video_extra.py` still passes.

## Constraints & Dependencies

- **Reuse**: `study_utils.core.config.load_toml()`, `merge_defaults()`, `write_toml_template()`; `study_utils.core.workspace.ensure_workspace()` + its layout helpers.
- **No new dependencies**: Uses existing `tomllib` (Python 3.12+ as project target per pyproject.toml). The `tomli` fallback is maintained only at the workspace/convert-markdown level and not needed here.
- **Template packaging**: Must register in `pyproject.toml` package-data so the template ships with the wheel, following convert-markdown's pattern.
- **Backward compatibility**: All existing CLI flags preserved; `_TRANSCRIBE_LLM` retained as fallback; module-level function signatures unchanged externally.

## Security & Privacy

- Config files are read from local paths only; no network calls during config resolution.
- No secrets (API keys) are stored in the config TOML — they continue to flow via environment variables through the shared `load_client()` helper.
- Config templates use file permission 0o600 when written, matching existing conventions.

## Rollout / Revert

- **Rollout**: Land all changes together (inline config code, template file, parser update, tests, registration). Verify both `study transcribe-video /dir` and `study transcribe-video config init|validate|path` work immediately.
- **Revert**: The inline additions (~100 lines) can be isolated; dropping them reverts to the existing module-level behavior with `_TRANSCRIBE_LLM` fully restored as primary source of truth.

## History

### 2025-08-13 09:00
**Summary** — Draft spec for transcribe-video config system and CLI command group
**Changes**
- Captured goals, TOML structure, dataclasses, CLI subcommands (init/validate/path), implementation steps, constraints, and tests.
