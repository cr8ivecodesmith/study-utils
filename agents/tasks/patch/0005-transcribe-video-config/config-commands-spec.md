# Transcribe-Video Config Commands — Spec

## Description

Wire up the existing inline config infrastructure in `transcribe_video/__init__.py` with a unified parser (`_build_parser()`), lazy config loading, and config subcommand dispatch in `main()`. This enables:

```
study transcribe-video config init
study transcribe-video config validate
study transcribe-video config path
study transcribe-video /path/to/videos           # still works via loaded config
```

## Motivation

The config layer (dataclasses, builders, validation helpers, template functions, subcommand handlers) was built inline in `transcribe_video/__init__.py` (see implementation.md), but **it is not connected to the CLI flow**. Calling `_parse_transcribe_args()` directly in `main()` bypasses the new parser and config entirely. The spec's original task list (all 9 steps) marks them `[x]`, yet three things are missing:

1. No **unified parser** (`_build_parser()`) — no subparsers for `config init|validate|path`.
2. No **lazy config loading** — `_TRANSCRIBE_CONFIG` is not wired; `main()` still reads from `_TRANSCRIBE_LLM` directly.
3. No **config dispatch in `main()`** — a plain `study transcribe-video /videos` runs the old fast path; there is no check for the `config` subcommand.

This spec turns on what's already been written.

## Goals

- Introduce `_build_parser()` with a `config` subparser supporting `init`, `validate`, `path` (mirroring RAG's nested subcommand pattern).
- Add a module-level `_TRANSCRIBE_CONFIG: TranscribeConfig | None` for lazy loading; provide `_get_config()` accessor.
- Update `main()` to dispatch config subcommands and, in transcribe mode, use loaded config values as defaults (CLI flags override).
- Modify `split_video_to_audio_segments()` to accept an optional `segment_duration_minutes` argument so the video chunk size honors the config value.
- All changes are in a single file (`transcribe_video/__init__.py`) plus tests. No new dependencies, no structural file changes.

## Non-Goals

- No restructuring of inline dataclasses into a separate module.
- No removal of `_TRANSCRIBE_LLM` global (it stays as backward-compat fallback).
- No change to the names cache path logic (`.transcribe_video_names.json`).
- No new logging infrastructure — existing stdout prints remain; `verbose` config key is available for future use.

## Behavior (BDD-ish)

### `config init`

- Given any workspace layout, when the user runs `study transcribe-video config init`, then the packaged template writes to `$STUDY_UTILS_DATA_HOME/config/transcribe.toml` and prints the destination path.
- Given an existing file at the default location without `--force`, when re-run, then the command exits non-zero with a "file already exists" message.
- Given `--path ./custom.toml`, when combined with `--force`, then the template is written to that absolute/workspace-relative path regardless of workspace layout.

### `config validate`

- Given a valid config file at the resolved path, when the user runs `study transcribe-video config validate`, then the command prints "Configuration OK" with key settings (model, chunk duration, smart_names, api_base) and exits 0.
- Given an invalid value (e.g., `segment_duration_minutes = -1`), when the user runs `config validate`, then a descriptive error is printed and exit code is 2.

### `config path`

- Given no arguments, when the user runs `study transcribe-video config path`, then the command resolves the config location via: explicit `--path` > env `$STUDY_TRANSCRIBE_CONFIG` > default `<data_home>/config/transcribe.toml` and prints it.

### Transcribe with Config

- When `smart_names = true` and `segment_duration_minutes = 15` in the config, running `study transcribe-video /my/courses` loads from config by default; `--no-smart-names` on CLI disables smart names, and chunks use the 15-minute value.
- Explicit CLI flags (e.g., `--output-dir`, `--prefix`, `--recursive`) take precedence over config values for their respective settings.
- When no config file exists, loaded defaults match today's hardcoded behavior — zero regression.

## Configuration Structure

Uses the existing `transform.toml` template already packaged in `src/study_utils/transcribe_video/`:

```toml
[ai]
use_local = true
api_base = "http://localhost:8080/v1"
provider = "local"
model = "whisper-3"
title_model = "gpt-4o-mini"

[audio]
segment_duration_minutes = 10

[names]
smart_names = true
use_ai_titles = false
output_dir = ""
recursive = true
cache_path = ""

[logging]
level = "INFO"
verbose = false
```

## Dataclass Hierarchy (existing, inline)

```python
@dataclasses.dataclass(frozen=True)
class AIConfig: ...          # use_local, api_base, provider, model, title_model
@dataclasses.dataclass(frozen=True)
class AudioConfig: ...       # segment_duration_minutes: int
@dataclasses.dataclass(frozen=True)
class NamesConfig: ...       # smart_names, use_ai_titles, output_dir, recursive, cache_path
@dataclasses.dataclass(frozen=True)
class LoggingConfig: ...     # level: str, verbose: bool

@dataclasses.dataclass(frozen=True)
class TranscribeConfig:      # ai, audio, names, logging
    ai: AIConfig
    audio: AudioConfig
    names: NamesConfig
    logging: LoggingConfig
```

## CLI Argument Precedence (highest → lowest)

1. **CLI flags** (`--output-dir`, `--prefix`, `--smart-names`, `--use-ai`, `--recursive`, `--names-file`, `config init/validate/path`)
2. **Environment variables** (`STUDY_TRANSCRIBE_CONFIG` for path; `TRANSCRIPTION_MODEL`, `OPENAI_TITLE_MODEL` as LLM overrides)
3. **TOML config file** (`transform.toml` / `transcribe.toml`)
4. **Hardcoded defaults** (`_TRANSCRIBE_LLM` dict and module constants)

## Implementation Scope

### Files Changed (all in one diff)

| File | Action | Details |
|------|--------|---------|
| `src/study_utils/transcribe_video/__init__.py` | Modify | Add `_build_parser()`, `_TRANSCRIBE_CONFIG`, `_get_config()`, update `main()`, wire `split_video_to_audio_segments()` — ~80-100 net new lines, zero deletions |
| `tests/test_transcribe_config.py` | Modify | Add ~10 new tests for parser, dispatch, CLI overrides, config-wired flow (~120 new lines) |

### Implementation Steps

**Step 1 — Add `_TRANSCRIBE_CONFIG` + lazy accessor**

Add module-level variable and getter near the existing constants:
```python
_TRANSCRIBE_CONFIG: Optional[TranscribeConfig] = None

def _get_config(env: Optional[Dict[str, str]] = None) -> TranscribeConfig:
    """Return the lazily-loaded transcribe config."""
    global _TRANSCRIBE_CONFIG  # noqa: PLW0603
    if _TRANSCRIBE_CONFIG is None:
        _TRANSCRIBE_CONFIG = load_config(env=env)
    return _TRANSCRIBE_CONFIG
```

This avoids loading config when running `config init`/`validate`/`path` (which use `args.path` directly without needing the config values for display).

**Step 2 — Add `_build_parser()` function**

Create a new argparse parser mirroring RAG's pattern. The parser has two tiers:
- **Top-level**: `config` subcommand vs default transcription flow (positional `TARGET`)
- **Config tier**: nested subparsers for `init`, `validate`, `path` with their own args (`--path`, `--force`)

Key structure:
```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study transcribe-video",
        description="Transcribe MP4 video(s) using Whisper."
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Default (transcribe) mode — same flags as _parse_transcribe_args() today
    main_parser = subparsers.add_parser("run", help="Transcribe videos (default)")
    _add_transcribe_flags(main_parser)

    # Config command tier
    config_parser = subparsers.add_parser("config", help="Manage configuration.")
    _build_config_subcommands(config_parser)

    return parser
```

`_build_config_subcommands()` mirrors RAG's version: adds `init` (with `--path`, `--force`), `validate` (with `--path`, `--quiet`), and `path` (with `--path`) as nested subparsers under the `config` parent.

`_add_transcribe_flags()` captures all flags from `_parse_transcribe_args()`: `TARGET`, `-o/--output-dir`, `-p/--prefix`, `-l/--list`, `-r/--recursive`, `--smart-names`, `--use-ai`, `--names-file`, `--refresh-names`.

**Step 3 — Update `main()` to dispatch config + use loaded config**

Replace the current `_parse_transcribe_args()` call with `_build_parser()`:
```python
def main():
    args = _build_parser().parse_args()

    # Dispatch config subcommands before running transcribe flow
    if getattr(args, "command", None) == "config":
        return _handle_config_dispatch(args)

    # Config-wired transcribe flow
    cfg = _get_config()
    target_path = Path(args.TARGET).expanduser().resolve()
    # Use arg.recursive with config fallback (CLI overrides if explicitly set)
    recursive = args.recursive or cfg.names.recursive
    smart_names = getattr(args, "smart_names", True)  # default from config via load
    ...
```

`_handle_config_dispatch()` routes to the existing handlers:
```python
def _handle_config_dispatch(args: argparse.Namespace) -> int:
    subcmd = getattr(args, "config_command", "path")
    if subcmd == "init":   return _handle_config_init(args)
    if subcmd == "validate": return _handle_config_validate(args)
    if subcmd == "path":   return _handle_config_path(args)
    return 1
```

**Step 4 — Wire `split_video_to_audio_segments` to config**

Update the function to accept an optional segment duration:
```python
def split_video_to_audio_segments(
    file_path: Path,
    exist_delete: bool = True,
    segment_duration_minutes: int = 10,  # was hardcoded 10
) -> List[Path]:
    full_audio = AudioSegment.from_file(file_path, format="mp4")
    segment_ms = segment_duration_minutes * 60 * 1000  # use parameter
    ...
```

Update the call site in `main()` to pass `cfg.audio.segment_duration_minutes`.

Note: `transcribe_audio_file()` uses the env variable `TRANSCRIPTION_MODEL` directly — that stays unchanged (no config key for it yet, following spec constraint).

**Step 5 — Update tests**

Add tests to `tests/test_transcribe_config.py`:

| Test | Verifies |
|------|----------|
| `test_build_parser_creates_default()` | Parser has `command` field; `default_mode` detected |
| `test_build_parser_has_config_subparser()` | Config subparsers exist for init, validate, path |
| `test_build_parser_has_transcribe_flags()` | All current CLI flags present in parser |
| `test_get_config_lazy_loads_once()` | Second call returns the same object (not reloaded) |
| `test_handle_config_dispatch_init()` | Routes to `_handle_config_init` when `config_command == "init"` |
| `test_handle_config_dispatch_validate()` | Routes to `_handle_config_validate` |
| `test_handle_config_dispatch_path()` | Routes to `_handle_config_path` |
| `test_main_dispatches_config_subcommand()` | Full integration: `study transcribe-video config init --path ...` writes file |
| `test_main_uses_loaded_config_ai_settings()` | main() passes `cfg.ai.use_local` and `cfg.ai.api_base` to `load_client()` |
| `test_main_cli_flags_override_config()` | `--recursive False` on CLI overrides config's `recursive = true` |
| `test_split_video_with_custom_duration()` | Passes custom segment duration; chunks have correct ms size |
| `test_backward_compat_no_config_file()` | Running without config file yields identical behavior to today |

### Constraints & Dependencies

- **Reuse**: Existing `_handle_config_init/validate/path()` (no signature changes), `load_config()`, `resolve_config_path()`, `get_template()`. The shared `core.config.merge_defaults()` and `write_toml_template()` are already used.
- **No new dependencies**: Uses only existing imports (`argparse`, `tomllib`, built-ins). No added `.py` files, just additions to the existing `__init__.py`.
- **Backward compatibility**: The existing `_parse_transcribe_args()` function is preserved; `_build_parser()` uses the same flag names. A bare `study transcribe-video /dir` without `config` subcommand continues to work identically. `_TRANSCRIBE_LLM` remains untouched.

### Security & Privacy

- Config files read from local paths only; no network calls during config resolution.
- No secrets (API keys) stored in the TOML — they continue to flow via environment through `load_client()`.
- Template writes use 0o600 permissions (via existing `write_toml_template` in `core/config_templates.py`).

### Rollout / Revert

**Rollout**: Land all changes together. Verify:
1. `study transcribe-video config init --path <tmp>/x.toml && study transcribe-video config validate --path <tmp>/x.toml && study transcribe-video config path`
2. `study transcribe-video /some/dir` still works with loaded config defaults
3. All existing tests in `test_transcribe_video_extra.py` and `test_transcribe_config.py` pass

**Revert**: Drop the ~80-100 new lines from `main()`, `_build_parser()`, and helpers; remove `_TRANSCRIBE_CONFIG` global. The code was already there — reverting just means "call `_parse_transcribe_args()` directly again, use `_TRANSCRIBE_LLM`". Template file stays (harmless extra asset). No test file deletions needed.

## History

### 2026-08-13 — Initial implementation plan
Captured in `implementation.md` — all 9 steps mark `[x]`, but the infra layer was built inline without wiring into main().

### 2026-08-13 — Config commands spec (this document)
Identifies that infrastructure exists but CLI wiring is incomplete; specifies the focused changes needed to complete the task.
