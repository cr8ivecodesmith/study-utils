# Transcribe-Video Config Commands — Implementation (Companion)

## Understanding
- The config infrastructure is already built inline in `transcribe_video/__init__.py` (dataclasses, builders, validators, template helpers, subcommand handlers) but **disconnected from the CLI flow**. Calling `_parse_transcribe_args()` directly in `main()` bypasses the new parser and lazy config entirely.
- This companion doc tracks Phase 2: wiring up the infrastructure with a unified parser (`_build_parser()`), lazy config loading via `_TRANSCRIBE_CONFIG`, and config subcommand dispatch in `main()`. All changes target one file except additional tests.

## Resources
### Project docs
- `agents/tasks/patch/0005-transcribe-video-config/spec.md` — original mini-spec for the config system (Phase 1 infra).
- `agents/tasks/patch/0005-transcribe-video-config/implementation.md` — Phase 1 checklist with all steps marked complete.
- `agents/tasks/patch/0005-transcribe-video-config/config-commands-spec.md` — detailed spec for this wiring phase.
- `agents/guides/engineering-guide.md` — seam-first, small units boundary patterns.
- `agents/guides/patterns-and-architecture.md` — module layout, composition over inheritance.
- `agents/guides/styleguides.md` — Ruff lint rules (line-length 80), Google docstrings, pytest conventions.

### Existing modules to reference
- `src/study_utils/transcribe_video/__init__.py` (~1090 lines) — single target for all changes. Contains the full infra layer (dataclasses, builders, handlers, templates) and the current `_parse_transcribe_args()` / `main()` functions.
- `src/study_utils/rag/cli.py` — pattern reference for nested subcommand parser (`_build_parser()`, `_handle_config()`).
- `src/study_utils/core/config.py` — shared `merge_defaults()`, `write_toml_template()` helpers already in use.

## Impact Analysis
### Affected behaviors & tests
- **Config subcommand dispatch**: three new CLI paths (`config init`, `config validate`, `config path`) now route through `_build_parser()` → `_handle_config_dispatch()`. Covered by 3 unit + 1 integration test.
- **main() flow**: replaces the current direct call to `_parse_transcribe_args()` with `_build_parser().parse_args()`. When `command == "config"`, dispatches to subcommand handlers and exits before transcribe logic. Otherwise, loads config lazily via `_get_config()` and uses it for defaults (CLI flag overrides preserved). Covered by integration tests plus one backward-compat test.
- **split_video_to_audio_segments()**: now accepts an optional `segment_duration_minutes` parameter so the audio chunk size honors the loaded config value instead of hardcoding 10 minutes.

### Affected source files
| File | Action | Details |
|------|--------|---------|
| `src/study_utils/transcribe_video/__init__.py` | Modify | Add `_TRANSCRIBE_CONFIG`, `_get_config()`, `_build_parser()`, `_handle_config_dispatch()`. Update `main()` and `split_video_to_audio_segments()` (~80–100 net new lines, zero deletions). |
| `tests/test_transcribe_config.py` | Modify | Add 10 new tests for parser, dispatch, CLI overrides, lazy loading, segment duration (~120 new lines). No changes to existing tests. |

### Security considerations
- Config files read from local paths only; no network calls during resolution.
- No secrets (API keys) stored in TOML — continue flowing via env through `load_client()`.
- Template writes retain 0o600 permissions via the shared helper.

## Solution Plan
- Architecture: follow RAG's nested subparser pattern. Keep all wiring inline in `_init__.py` to avoid introducing new files for a focused change like this. Lazy config loading avoids unnecessary TOML I/O when running `config init|validate|path`.
- DI & boundaries: the lazy getter `_get_config()` is the seam; in tests, the module-level variable can be reset (`_TRANSCRIBE_CONFIG = None`) or bypassed by calling `load_config(explicit_path=...)` directly.

### Stepwise checklist

**Step 1 — Add `_TRANSCRIBE_CONFIG` global + `_get_config()` accessor (~10 lines)**

- [ ] Add `import sys` at the top (needed for stderr reference in handlers)
- [ ] After the existing constants block (~line 123), add:
  ```python
  _TRANSCRIBE_CONFIG: Optional[TranscribeConfig] = None
  ```
- [ ] Add getter function before or after constants:
  ```python
  def _get_config(env: Optional[Dict[str, str]] = None) -> TranscribeConfig:
      """Return the lazily-loaded transcribe-video config."""
      global _TRANSCRIBE_CONFIG  # noqa: PLW0603
      if _TRANSCRIBE_CONFIG is None:
          _TRANSCRIBE_CONFIG = load_config(env=env)
      return _TRANSCRIBE_CONFIG
  ```
- [ ] Verify no type warnings added; `_get_config()` returns `TranscribeConfig` directly.

### Step 2 — Add `_build_parser()` function (~45 lines)

Located after existing helpers, before `find_video_files()` (~line 435).

- [ ] Create the top-level parser with subparsers:
  ```python
  def _build_parser() -> argparse.ArgumentParser:
      """Build the unified argument parser for transcribe-video."""
      import argparse
      parser = argparse.ArgumentParser(
          prog="study transcribe-video",
          description="Transcribe MP4 video(s) using Whisper.",
      )
      subparsers = parser.add_subparsers(dest="command", required=False)

      # Default transcribe mode (same flags as _parse_transcribe_args())
      main_parser = subparsers.add_parser("run", help="Transcribe videos (default)")
      _add_transcribe_flags(main_parser)

      # Config command tier — nested sub-subcommands
      config_parser = subparsers.add_parser("config", help="Manage configuration.")
      _build_config_subcommands(config_parser)

      return parser
  ```
- [ ] Implement `_add_transcribe_flags(parser)` — mirrors the flags currently in `_parse_transcribe_args()`:
  - `TARGET` (positional), `-o/--output-dir`, `-p/--prefix` (append), `-l/--list`, `-r/--recursive`, `--smart-names`, `--use-ai`, `--names-file`, `--refresh-names`.
- [ ] Implement `_build_config_subcommands(parent)` — mirrors RAG's pattern:
  ```python
  def _build_config_subcommands(parent):
      subparsers = parent.add_subparsers(dest="config_command", required=True)

      init_parser = subparsers.add_parser("init", help="Write the default config template.")
      init_parser.add_argument("--path", type=str, help="Destination for the config TOML.")
      init_parser.add_argument("--force", action="store_true", help="Overwrite existing file.")

      validate_parser = subparsers.add_parser(
          "validate", help="Validate the active config file.",
      )
      validate_parser.add_argument(
          "--path", type=str, help="Path to the config TOML (defaults to resolved).",
      )
      validate_parser.add_argument(
          "--quiet", action="store_true", help="Suppress success output.",
      )

      path_parser = subparsers.add_parser("path", help="Print the resolved config path.")
      path_parser.add_argument(
          "--path", type=str, help="Optional path override to resolve/normalise.",
      )
  ```
- [ ] Add `_to_path(value: Optional[str]) -> Optional[Path]` helper (same as RAG's) for the `--path` flag conversion.

### Step 3 — Add `_handle_config_dispatch(args)` (~10 lines)

Add after `_handle_config_path()`, before `find_video_files()` (~line 435).

- [ ] Implement dispatch:
  ```python
  def _handle_config_dispatch(args: Any) -> int:
      """Route to the active config subcommand handler."""
      command = getattr(args, "config_command", "path")
      if command == "init":   return _handle_config_init(args)
      if command == "validate": return _handle_config_validate(args)
      if command == "path":   return _handle_config_path(args)
      print(f"Unknown config subcommand: {command}", file=sys.stderr)
      return 1
  ```

### Step 4 — Update `main()` to use parsed config (~25 lines changed)

Locate existing `def main():` (~line 800).

- [ ] Replace current implementation with:
  ```python
  def main():
      args = _build_parser().parse_args()

      # Dispatch config subcommands before transcribe flow.
      if getattr(args, "command", None) == "config":
          return _handle_config_dispatch(args)

      # Config-wired transcribe flow.
      cfg = _get_config()
      target_path = Path(args.TARGET).expanduser().resolve()
      video_files = find_video_files(
          target_path, recursive=args.recursive or cfg.names.recursive,
      )

      if args.list_only:
          _handle_list_mode(args, video_files, target_path)
          return

      if not video_files:
          print("No .mp4 files found to transcribe.")
          raise SystemExit(1)

      out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path.cwd()
      out_dir.mkdir(parents=True, exist_ok=True)

      client = load_client(
          local=cfg.ai.use_local,
          api_base=cfg.ai.api_base,
      )
      parsed_prefix = parse_prefix_parts(args.prefix)
      names_entries = _prepare_names_for_run(
          args, video_files, target_path, client, parsed_prefix,
      )

      # ... remaining logic: transcribe with cli-flag overrides.
  ```
- [ ] Wire `cfg.ai.use_local` and `cfg.ai.api_base` into the `load_client()` call (previously used `_TRANSCRIBE_LLM["USE_LOCAL"]` / `["API_BASE"]`).
- [ ] Ensure backward compat: when no config file exists, `_get_config()` returns defaults matching `_TRANSCRIBE_LLM`, so behavior is identical.
- [ ] Verify all transcribe helper calls receive the same argument types they expect (args.Namespace vs SimpleNamespace).

### Step 5 — Update `split_video_to_audio_segments()` to accept config duration (~4 lines)

Locate the function at ~line 649.

- [ ] Modify signature:
  ```python
  def split_video_to_audio_segments(
      file_path: Path,
      exist_delete: bool = True,
      segment_duration_minutes: int = 10,
  ) -> List[Path]:
  ```
- [ ] Change the segment calculation from hard-coded `10 * 60 * 1000`:
  ```python
  segment_ms = segment_duration_minutes * 60 * 1000
  ```
- [ ] In `main()`, after loading config, extract `segment_duration` and pass it when splitting:
  ```python
  segments = split_video_to_audio_segments(
      video_path, segment_duration_minutes=cfg.audio.segment_duration_minutes,
  )
  ```

### Step 6 — Add tests (~10 new test methods, ~120 lines)

All new tests go in `tests/test_transcribe_config.py` before the existing `TestBackwardCompat` class.

- [ ] `test_build_parser_creates_subparsers()` — parser has `command`, config subparser present with sub-subparsers
- [ ] `test_build_parser_has_transcribe_flags()` — TARGET, -o, -p, -r, --smart-names, --use-ai all present
- [ ] `test_get_config_lazy_loads_once()` — `_TRANSCRIBE_CONFIG` is None initially; after first call it's set; second call returns same object
- [ ] `test_handle_config_dispatch_init()` — routes to init handler when `config_command == "init"`
- [ ] `test_handle_config_dispatch_validate()` — routes to validate handler
- [ ] `test_handle_config_dispatch_path()` — routes to path handler
- [ ] `test_main_dispatches_config_subcommand()` — full integration: parse + dispatch writes file, prints path, returns 0
- [ ] `test_main_uses_loaded_config_ai_settings()` — main() passes loaded config values to load_client (mock check)
- [ ] `test_main_cli_flags_override_config()` — CLI flags take precedence over config for their settings
- [ ] `test_split_video_with_custom_duration()` — passing custom segment_minutes produces correct ms value in chunks

## Test Plan
### Unit tests
- Parser: `_build_parser()` returns a valid argument parser with subparser hierarchy. All existing transcribe flags are present. Config sub-subparsers (`init`, `validate`, `path`) attach correctly with their own flags.
- Lazy config: `_get_config()` initializes on first call, caches globally, reuses on subsequent calls. Mutating `_TRANSCRIBE_CONFIG` resets it (testable via monkeypatch).
- Dispatch: `_handle_config_dispatch()` routes to the correct handler based on `config_command`. Known invalid subcommand returns 1 with stderr message.
- Segment duration: `split_video_to_audio_segments(..., segment_duration_minutes=15)` produces `segment_ms = 900000` (15 * 60 * 1000).

### Integration / E2E tests
- CLI invocation: `study transcribe-video config init --path <tmp>/x.toml` writes file and prints path. Re-run without `--force` → non-zero exit. With `--force` → success + overwrite.
- Config-wired transcribe: `main()` loads config, uses its values for client params and defaults. When config doesn't exist, falls back to hardcoded values (backward compat test).
- CLI flag override: setting `--recursive=False` on CLI overrides config's `recursive=True`. Only the flag being provided is overridden; other settings remain as loaded from config.

### Regression scope
- No changes to existing `test_transcribe_config.py` tests or `test_transcribe_video_extra.py`. The `_parse_transcribe_args()` function stays intact for backward compatibility. One regression test validates that running without a config file produces identical behavior to today (config defaults match hardcoded defaults).

## Operability
- **Revert**: Remove the ~80–100 new lines from `main()`, `_build_parser()`, `_get_config()`, `_handle_config_dispatch()`. Remove `_TRANSCRIBE_CONFIG` global. Revert segment duration change in `split_video_to_audio_segments()` back to hardcoded. The original `_parse_transcribe_args()` call resumes direct use. No test file deletions needed (new tests remain inert).
- **Troubleshooting**: If transcribe-video runs but seems to use old settings, run `study transcribe-video config path` and `config validate` to inspect resolved config. Check that `_TRANSCRIBE_CONFIG` was initialized correctly — set to `None` to force re-read.

## History
### 2026-08-13 09:00
**Summary**
Initial implementation plan for the wiring phase (Phase 2). Infrastructure from Phase 1 is in place; this doc tracks connecting it to the CLI pipeline.
**Changes**
- Drafted companion impl doc covering understanding, impact analysis, six stepwise implementation steps, test plan, and operability notes.
