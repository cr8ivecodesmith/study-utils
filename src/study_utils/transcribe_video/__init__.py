"""Transcribe mp4 video(s) to plain text using Whisper-1.

Features:
- Discover `.mp4` files from a file or directory, with optional recursion.
- Optional list mode to preview discovered files and proposed output names.
- "Smart" output names derived from directory structure; optional AI-refined
  titles.
- Names cache file (`.transcribe_video_names.json`) that you can edit and
  reuse.
- Composable filename prefixes (text and zero-padded counters).
- Splits audio into ~10-minute mp3 chunks with pydub/ffmpeg, transcribes, and
  concatenates.
- Environment-driven OpenAI client setup via `study_utils.core.load_client()`
  with `.env` support.

Design notes:
- Argparse CLI with small, pure helpers for discovery, naming, and parsing.
- I/O and API calls are isolated in `main()` and `transcribe_*` helpers.
- Avoids global state; cache file path is explicit/deterministic.
"""

import argparse
import dataclasses
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from time import sleep
from typing import Any, Dict, List, Mapping, Optional, Tuple

from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks

from study_utils.core import load_client
from study_utils.core.config import TomlConfigError, merge_defaults
from study_utils.core.config_templates import get_template

try:
    from pathlib import Path as _PathType
except ImportError:
    from pathlib import Path as _PathType  # noqa: F811

_TRANSCRIBE_LLM = {
    "USE_LOCAL": True,
    "API_BASE": "http://localhost:8080/v1",
    "MODEL": os.getenv("TRANSCRIPTION_MODEL", "whisper-3"),
}


class ConfigError(RuntimeError):
    """Raised when transcribe-video config validation fails."""


CONFIG_PATH_ENV = "STUDY_TRANSCRIBE_CONFIG"

DEFAULT_CONFIG_PATH = "config/transcribe.toml"


@dataclasses.dataclass(frozen=True)
class AIConfig:
    use_local: bool
    api_base: str
    provider: str
    model: str
    title_model: str


@dataclasses.dataclass(frozen=True)
class AudioConfig:
    segment_duration_minutes: int


@dataclasses.dataclass(frozen=True)
class NamesConfig:
    smart_names: bool
    use_ai_titles: bool
    output_dir: Optional[_PathType]
    recursive: bool
    cache_path: Optional[_PathType]


@dataclasses.dataclass(frozen=True)
class LoggingConfig:
    level: str
    verbose: bool


@dataclasses.dataclass(frozen=True)
class TranscribeConfig:
    ai: AIConfig
    audio: AudioConfig
    names: NamesConfig
    logging: LoggingConfig


_DEFAULTS = {
    "ai": {
        "use_local": True,
        "api_base": "http://localhost:8080/v1",
        "provider": "local",
        "model": "whisper-3",
        "title_model": "gpt-4o-mini",
    },
    "audio": {
        "segment_duration_minutes": 10,
    },
    "names": {
        "smart_names": True,
        "use_ai_titles": False,
        "output_dir": None,
        "recursive": True,
        "cache_path": None,
    },
    "logging": {
        "level": "INFO",
        "verbose": False,
    },
}

_TRANSCRIBE_CONFIG: Optional[TranscribeConfig] = None


def _get_config(env: Optional[Dict[str, str]] = None) -> TranscribeConfig:
    """Return the lazily-loaded transcribe-video config.

    First call resolves and loads the TOML, subsequent calls return the same
    instance without re-reading the file.
    """
    global _TRANSCRIBE_CONFIG  # noqa: PLW0603
    if _TRANSCRIBE_CONFIG is None:
        _TRANSCRIBE_CONFIG = load_config(env=env)
    return _TRANSCRIBE_CONFIG


def _require_positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(
            f"'{key}' must be a positive integer, got {type(value).__name__}."
        )
    if value <= 0:
        raise ConfigError(f"'{key}' must be positive, got {value}.")

    return value


def _require_non_negative_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(
            f"'{key}' must be an integer, got {type(value).__name__}."
        )
    if value < 0:
        raise ConfigError(f"'{key}' must be non-negative, got {value}.")
    return value


def _require_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            f"'{key}' must be a boolean, got {type(value).__name__}."
        )
    return value


def _require_string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            f"'{key}' must be a string, got {type(value).__name__}."
        )

    result = value.strip()
    if not result:
        raise ConfigError(f"'{key}' must be a non-empty string.")
    return result


def _coerce_optional_string(value: Any, key: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        result = value.strip()
        return result if result else None
    raise ConfigError(
        f"'{key}' must be a string or null, got {type(value).__name__}."
    )


def _coerce_optional_path(value: Any, key: str) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        result = value.strip()
        return Path(result) if result else None
    raise ConfigError(
        f"'{key}' must be a path (str/Path) or null, "
        f"got {type(value).__name__}."
    )


def _build_ai(section: Mapping[str, Any]) -> AIConfig:
    return AIConfig(
        use_local=_require_bool(section.get("use_local", True), "ai.use_local"),
        api_base=(
            _coerce_optional_string(section.get("api_base"), "ai.api_base")
            or _DEFAULTS["ai"]["api_base"]
        ),
        provider=_coerce_optional_string(section.get("provider"), "ai.provider")
        or "local",
        model=_coerce_optional_string(section.get("model"), "ai.model")
        or "whisper-3",
        title_model=(
            _coerce_optional_string(
                section.get("title_model"), "ai.title_model"
            )
            or "gpt-4o-mini"
        ),
    )


def _build_audio(section: Mapping[str, Any]) -> AudioConfig:
    return AudioConfig(
        segment_duration_minutes=_require_positive_int(
            section.get("segment_duration_minutes", 10),
            "audio.segment_duration_minutes",
        ),
    )


def _build_names(section: Mapping[str, Any]) -> NamesConfig:
    return NamesConfig(
        smart_names=_require_bool(
            section.get("smart_names", True), "names.smart_names"
        ),
        use_ai_titles=_require_bool(
            section.get("use_ai_titles", False), "names.use_ai_titles"
        ),
        output_dir=_coerce_optional_path(
            section.get("output_dir"), "names.output_dir"
        ),
        recursive=_require_bool(
            section.get("recursive", True), "names.recursive"
        ),
        cache_path=_coerce_optional_path(
            section.get("cache_path"), "names.cache_path"
        ),
    )


def _build_logging(section: Mapping[str, Any]) -> LoggingConfig:
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level = (
        _coerce_optional_string(section.get("level"), "logging.level") or "INFO"
    )
    if level.upper() not in valid_levels:
        raise ConfigError(
            f"'logging.level' must be one of {valid_levels}, got '{level}'."
        )
    return LoggingConfig(
        level=level.upper(),
        verbose=_require_bool(section.get("verbose", False), "logging.verbose"),
    )


def _build_config(tree: Mapping[str, Any]) -> TranscribeConfig:
    merged = deepcopy(_DEFAULTS)
    try:
        merge_defaults(merged, dict(tree))
    except TomlConfigError as exc:
        raise ConfigError(str(exc)) from exc
    return TranscribeConfig(
        ai=_build_ai(merged["ai"]),
        audio=_build_audio(merged["audio"]),
        names=_build_names(merged["names"]),
        logging=_build_logging(merged["logging"]),
    )


def default_tree() -> Dict[str, Any]:
    return deepcopy(_DEFAULTS)


_DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH


def config_template() -> str:
    """Return the packaged transcribe-video template content."""
    template = get_template("transcribe_video")
    return template.read_text()


def write_template(
    path: Path,
    *,
    overwrite: bool = False,
    mode: int = 0o600,
) -> Path:
    """Write the packaged template to ``path`` using TOML helper semantics."""
    tmpl = get_template("transcribe_video")
    return tmpl.write(path, overwrite=overwrite, mode=mode)


def resolve_config_path(
    explicit_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    """Resolve the config file path.

    Precedence is: explicit argument > environment variable > default.
    """

    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()

    env_path = (env or {}).get(CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()

    data_home_env = os.getenv("STUDY_UTILS_DATA_HOME")
    if data_home_env:
        base = Path(data_home_env) / "config" / "transcribe.toml"
    else:
        from pathlib import Path as _PathType

        config_dir = _PathType.home() / ".study_utils" / "config"
        base = config_dir / "transcribe.toml"

    return base.expanduser().resolve()


def load_config(
    explicit_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> TranscribeConfig:
    """Load and validate the transcribe-video config from TOML file.

    Falls back to defaults when no config file is found.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - defensive guard
        raise ConfigError("tomllib is required (Python 3.11+).")

    config_path = resolve_config_path(explicit_path, env)

    if not config_path.exists():
        return _build_config(default_tree())

    try:
        with config_path.open("rb") as fh:
            raw = fh.read()
            text = raw.decode("utf-8")
        # Handle bare 'null' literals for Python 3.12+ tomllib.
    except Exception as exc:
        raise ConfigError(
            f"Failed to read config {config_path}: {exc}"
        ) from exc

    try:
        import re as _re

        clean_text = _re.sub(r"(\w+)\s*=\s*null\b", r'\1=""', text)
        toml_data = tomllib.loads(clean_text)
    except Exception as exc:
        raise ConfigError(
            f"Failed to parse config {config_path}: {exc}"
        ) from exc

    return _build_config(toml_data)


_handle_config_log = None


def _handle_config_init(args: Any) -> int:
    """Handle the 'config init' subcommand.

    Writes template and returns exit code.
    """

    global _handle_config_log  # noqa: PLW0603 _handle_config_log used for testing

    path = args.path
    force = getattr(args, "force", False)

    resolved_path = resolve_config_path(
        explicit_path=Path(path).expanduser().resolve() if path else None,
    )

    try:
        write_template(resolved_path, overwrite=force)
    except TomlConfigError as exc:
        stderr = getattr(sys, "stderr", None) or sys.stderr
        print(f"Error: {exc}", file=stderr or None)
        return 2

    print(f"Configuration written to {resolved_path}")
    return 0


def _handle_config_validate(args: Any) -> int:
    """Handle the 'config validate' subcommand. Validates config file."""
    try:
        expl = (
            Path(args.path).expanduser().resolve()
            if getattr(args, "path", None)
            else None
        )
        cfg = load_config(explicit_path=expl)
    except ConfigError as exc:
        stderr = getattr(sys, "stderr", None) or sys.stderr
        print(f"Configuration error: {exc}", file=stderr or None)
        return 2

    lines = [
        "Configuration OK",
        f"  model={cfg.ai.model}, title_model={cfg.ai.title_model}",
        f"  api_base={cfg.ai.api_base}, use_local={cfg.ai.use_local}",
        f"  segment_duration_minutes={cfg.audio.segment_duration_minutes}",
        f"  smart_names={cfg.names.smart_names}, "
        f"recursive={cfg.names.recursive}",
        f"  logging.level={cfg.logging.level}, verbose={cfg.logging.verbose}",
    ]
    for line in lines:
        print(line)
    return 0


def _handle_config_path(args: Any) -> int:
    """Handle the 'config path' subcommand. Prints resolved config path."""
    expl = (
        Path(args.path).expanduser().resolve()
        if getattr(args, "path", None)
        else None
    )
    try:
        cfg_path = resolve_config_path(explicit_path=expl)
        print(cfg_path)
        return 0
    except Exception as exc:
        stderr = getattr(sys, "stderr", None) or sys.stderr
        print(f"Error resolving config path: {exc}", file=stderr or None)
        return 1


def _handle_config_dispatch(args: Any) -> int:
    """Route to the active config subcommand handler."""
    command = getattr(args, "config_command", "path")
    if command == "init":
        return _handle_config_init(args)
    if command == "validate":
        return _handle_config_validate(args)
    if command == "path":
        return _handle_config_path(args)
    print(f"Unknown config subcommand: {command}", file=sys.stderr)
    return 1


def _to_path(value: str | None) -> Path | list:
    """Convert a possibly-None path string to a resolved Path."""
    if not value:
        return []
    return Path(value).expanduser().resolve()


def _add_transcribe_flags(parser: "argparse.ArgumentParser") -> None:
    """Add the standard transcribe flags to an existing parser."""
    parser.add_argument(
        "TARGET",
        help="Path to an .mp4 file or a directory containing .mp4 files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write transcripts (default: cwd)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        action="append",
        help=(
            "Composable prefix parts (repeatable). Format: "
            "text:VALUE | counter:N|NN|NNN|NNNN. Order is preserved. "
            "Example: -p 'text:Intro' -p 'counter:NN' -p 'text: '. "
            "Legacy 'sep:VALUE' is treated as text:VALUE."
        ),
    )
    parser.add_argument(
        "-l",
        "--list",
        dest="list_only",
        action="store_true",
        help="List discovered .mp4 files and exit",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Traverse subfolders of the target directory",
    )
    parser.add_argument(
        "--smart-names",
        dest="smart_names",
        action="store_true",
        help="Generate smart output names from directory structure",
    )
    parser.add_argument(
        "--use-ai",
        dest="use_ai",
        action="store_true",
        help="Use OpenAI to refine smart names (optional)",
    )
    parser.add_argument(
        "--names-file",
        dest="names_file",
        help=(
            "Path to cache file for proposed names (defaults to a hidden file "
            "in the target root)"
        ),
    )
    parser.add_argument(
        "--refresh-names",
        dest="refresh_names",
        action="store_true",
        help=(
            "Regenerate names for discovered files (overwrites cache entries)"
        ),
    )


def _build_config_subcommands(parent: "argparse.ArgumentParser") -> None:
    """Build the config sub-subparsers on *parent*."""
    subparsers = parent.add_subparsers(dest="config_command", required=True)

    init_p = subparsers.add_parser(
        "init", help="Write the default config template."
    )
    init_p.add_argument(
        "--path",
        type=str,
        default=None,
        help="Destination for the config TOML.",
    )
    init_p.add_argument(
        "--force", action="store_true", help="Overwrite existing file."
    )

    val_p = subparsers.add_parser(
        "validate", help="Validate the active config file."
    )
    val_p.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the config TOML (defaults to resolved).",
    )
    val_p.add_argument(
        "--quiet", action="store_true", help="Suppress success output."
    )

    path_p = subparsers.add_parser(
        "path", help="Print the resolved config path."
    )
    path_p.add_argument(
        "--path",
        type=str,
        default=None,
        help="Optional path override to resolve/normalise.",
    )


def _build_parser() -> "argparse.ArgumentParser":
    """Build the unified argument parser for transcribe-video."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="study transcribe-video",
        description="Transcribe MP4 video(s) using Whisper.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Default transcribe mode (unnamed key, all transcribe flags attached).
    main_p = subparsers.add_parser(None, help="Transcribe videos (default)")
    _add_transcribe_flags(main_p)

    # Config command tier.
    cfg_p = subparsers.add_parser("config", help="Manage configuration.")
    _build_config_subcommands(cfg_p)

    return parser


def _parse_known(
    parser: "argparse.ArgumentParser", args=None
) -> argparse.Namespace:
    """Parse args using parse_known_args and fix up positional TARGET.

    When subparsers consume all positionals into 'command', unrecognized
    ones remain in the remaining list returned by parse_known_args().
    This helper moves the first remaining positional into TARGET if needed.
    """
    parsed, remaining = parser.parse_known_args(args)
    if getattr(parsed, "TARGET", None) is None and remaining:
        for token in remaining:
            if not token.startswith("-"):
                parsed.TARGET = token
                break
    return parsed


def find_video_files(target: Path, recursive: bool = False) -> List[Path]:
    """Return a flat list of `.mp4` files for the given target.

    - If `target` is a file, validates extension and returns [target].
    - If `target` is a directory and `recursive` is False, return only
      top-level `.mp4` files.
    - If `target` is a directory and `recursive` is True, traverse
      subfolders.
    """
    if target.is_file():
        if target.suffix.lower() != ".mp4":
            raise ValueError("Only .mp4 files are supported")
        return [target]

    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    if not target.is_dir():
        raise ValueError(f"Target must be a file or directory: {target}")

    if recursive:
        return sorted([p for p in target.rglob("*.mp4") if p.is_file()])
    else:
        files = [
            p
            for p in sorted(target.iterdir())
            if p.is_file() and p.suffix.lower() == ".mp4"
        ]
        return files


def default_names_cache_path(target_root: Path) -> Path:
    """Return a default path to store names cache (editable by user).

    Preference order:
    - If target_root is a directory, store a hidden file under it.
    - If it's a file, store under its parent.
    - Fallback to a temp dir file keyed by root name.
    """
    base = target_root if target_root.is_dir() else target_root.parent
    if base.exists() and base.is_dir():
        return base.joinpath(".transcribe_video_names.json")
    # Fallback to temp with a deterministic filename
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(target_root))
    return Path(gettempdir()).joinpath(f"transcribe_video_names_{safe}.json")


def load_names_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        names = data.get("names", {})
        if isinstance(names, dict):
            return {str(Path(k)): v for k, v in names.items()}
    except Exception:
        return {}
    return {}


def save_names_cache(
    cache_path: Path,
    root: Path,
    names: Dict[Path, Any],
    meta: Optional[Dict] = None,
) -> None:
    """Save cache entries as either strings (base) or small dicts.

    When the value is a string, treat it as the base smart name. When it is a
    dict, it may contain keys like {"base": str, "final": str}.
    """
    payload = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "names": {},
    }
    for k, v in names.items():
        if isinstance(v, dict):
            payload["names"][str(Path(k))] = v
        else:
            payload["names"][str(Path(k))] = {"base": str(v)}
    if meta:
        payload["meta"] = meta
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def cache_get_base(entry: Any, fallback_stem: str) -> str:
    if isinstance(entry, dict):
        val = entry.get("base")
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(entry, str) and entry.strip():
        return entry
    return fallback_stem


def cache_get_final(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        val = entry.get("final")
        if isinstance(val, str) and val.strip():
            return val
    return None


def _clean_segment(text: str) -> str:
    """Normalize a path segment for use in a smart name."""
    s = text
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    # drop extension if present
    s = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", s)
    # remove leading ordering like "01 - ", "1.", "m01 -"
    s = re.sub(r"^(?i:m\d+|\d{1,3})(?:\s*[-–.:]|\))\s*", "", s)
    return s


def heuristic_smart_name(video_path: Path, root: Path) -> str:
    """Build a smart base name from directory structure and file stem.

    Uses up to the last two directories plus cleaned file stem.
    """
    try:
        rel = video_path.relative_to(root)
    except Exception:
        rel = video_path.name
    parts = list(rel.parts) if isinstance(rel, Path) else [rel]
    if parts and parts[-1] == video_path.name:
        parts = parts[:-1]
    segments = [_clean_segment(p) for p in parts][-2:]  # last two folders
    stem = _clean_segment(video_path.stem)
    pieces = [p for p in segments + [stem] if p]
    # Ensure not too long
    base = " - ".join(pieces)
    return base[:120] if len(base) > 120 else base


def ai_smart_name(
    client: OpenAI, video_path: Path, root: Path
) -> Optional[str]:
    """Attempt to generate a concise descriptive name using OpenAI.

    Falls back to None on any error.
    """
    prompt = (
        "Generate a concise, file-name-safe, human-friendly title "
        "(<= 80 chars)\n"
        "for a course video based only on its directory path and file name.\n"
        "Use important folder names (e.g., module/week/section) and the file "
        "stem.\n"
        "Avoid quotes; avoid slashes; return only the title.\n\n"
        f"Root: {root}\n"
        f"Path: {video_path}\n"
    )
    try:
        # Prefer a lightweight model if available
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You create concise, file-name-safe titles.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        choice = resp.choices[0].message.content.strip()
        # Basic sanitization
        choice = re.sub(r"[\r\n]+", " ", choice)
        choice = re.sub(r"[\\/:*?\"<>|]", "-", choice)
        return choice[:120]
    except Exception:
        return None


def build_name_mapping(
    files: List[Path],
    root: Path,
    use_ai: bool,
    client: Optional[OpenAI],
) -> Dict[Path, str]:
    """Return mapping of video path -> smart base name (no extension).

    Ensures uniqueness by suffixing duplicates with an index.
    """
    mapping: Dict[Path, str] = {}
    seen: Dict[str, int] = {}
    for p in files:
        base = heuristic_smart_name(p, root)
        if use_ai and client is not None:
            ai_name = ai_smart_name(client, p, root)
            if ai_name:
                base = ai_name
        # prevent empty
        if not base:
            base = p.stem
        # ensure uniqueness
        key = base
        if key in seen:
            seen[key] += 1
            key = f"{base} ({seen[base]})"
        else:
            seen[key] = 0
        mapping[p] = key
    return mapping


def split_video_to_audio_segments(
    file_path: Path,
    exist_delete: bool = True,
    segment_duration_minutes: int = 10,
) -> List[Path]:
    """Extract audio from an mp4 and split into ~N minute mp3 segments.

    Returns a list of mp3 segment file paths. Creates a transient directory
    `<video_stem>_segments` in the current working directory and fills it with
    the mp3 chunks. Caller is responsible for removing this directory when
    done.
    """
    # Load audio track from the video (requires ffmpeg via pydub)
    full_audio = AudioSegment.from_file(file_path, format="mp4")
    segment_ms = (
        segment_duration_minutes * 60 * 1000
    )  # N minutes in milliseconds
    chunks = make_chunks(full_audio, segment_ms)

    chunk_dir = Path(f"{file_path.stem}_segments")
    if exist_delete and chunk_dir.exists():
        rmtree(chunk_dir)
    chunk_dir.mkdir(mode=0o755, exist_ok=True)

    segment_files: List[Path] = []
    for idx, chunk in enumerate(chunks):
        audio_chunk = chunk_dir.joinpath(
            f"{file_path.stem}_segment_{idx:02d}.mp3"
        )
        chunk.export(audio_chunk, format="mp3")
        segment_files.append(audio_chunk)

    return segment_files


def transcribe_audio_file(client: OpenAI, audio_path: Path) -> str:
    """Transcribe audio to text using the configured Whisper model.

    Returns plain text output from the transcription API.
    """

    response = client.audio.transcriptions.create(
        model=os.getenv("TRANSCRIPTION_MODEL", "whisper-3"),
        file=audio_path.open("rb"),
        response_format="text",
    )
    # SDK returns a plain string when response_format='text'
    return (
        response.strip() if isinstance(response, str) else str(response).strip()
    )


def transcribe_video_file(
    client: OpenAI, video_path: Path, segment_duration_minutes: int = 10
) -> str:
    """Transcribe an mp4 by chunking its audio and concatenating results."""
    print(f"Splitting audio for {video_path.name} ...")
    segments = split_video_to_audio_segments(
        video_path, segment_duration_minutes=segment_duration_minutes
    )
    print(f"Segments directory: {segments[0].parent if segments else 'N/A'}")

    transcripts: List[str] = []
    try:
        for seg in segments:
            print(f"Transcribing {seg.name} ...")
            text = transcribe_audio_file(client, seg)
            transcripts.append(text)
            # Gentle pacing to avoid hammering the API
            sleep(1)
    finally:
        # Cleanup segment directory regardless of success
        if segments:
            print("Cleaning up segments ...")
            rmtree(segments[0].parent)

    return "\n".join(transcripts)


def sanitize_filename(name: str) -> str:
    """Remove or replace characters not safe for common filesystems."""
    name = re.sub(r"[\\/:*?\"<>|]", "-", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _strip_outer_quotes(s: str) -> str:
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        return s[1:-1]
    return s


def parse_prefix_parts(parts: Optional[List[str]]) -> List[Tuple[str, str]]:
    """Parse ordered prefix parts.

    Supports items of the form:
    - text:VALUE (may include separators/spaces)
    - counter:N | NN | NNN | NNNN (zero-padded index width 1-4)
    Note: 'sep:VALUE' is accepted for backward compatibility and treated as
    text:VALUE.
    Returns a list of tuples: (kind, value) where kind in {text, sep, counter}.
    Unknown items are treated as text:VALUE (back-compat).
    """
    if not parts:
        return []
    out: List[Tuple[str, str]] = []
    for raw in parts:
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if ":" in s:
            kind, val = s.split(":", 1)
            kind = kind.strip().lower()
            val = _strip_outer_quotes(val.strip())
            if kind in {"text", "sep"}:
                # Treat 'sep' as an alias of 'text' for simplicity
                out.append(("text", val))
                continue
            if kind == "counter":
                n = val.strip()
                if re.fullmatch(r"N{1,4}", n):
                    out.append(("counter", n))
                    continue
        # fallback -> treat whole string as text
        out.append(("text", _strip_outer_quotes(s)))
    return out


def build_prefix_string(parsed_parts: List[Tuple[str, str]], index: int) -> str:
    """Build the prefix string for a given 1-based index."""
    buf: List[str] = []
    for kind, val in parsed_parts:
        if kind == "text":
            buf.append(val)
        elif kind == "counter":
            width = len(val)
            buf.append(str(index).zfill(width))
    return "".join(buf)


def make_output_filename(
    video_path: Path,
    index: int,
    parsed_prefix: Optional[List[Tuple[str, str]]] = None,
    smart_base: Optional[str] = None,
) -> str:
    """Return output filename per spec.

    - Default: `<video_filename_stem>.txt`
    - With smart_base: `<smart_base>.txt`
    - With parsed_prefix: `<prefix_parts><base>.txt` (prefix is combinable)
    """
    base = sanitize_filename(smart_base) if smart_base else video_path.stem
    prefix_str = build_prefix_string(parsed_prefix or [], index)
    return f"{prefix_str}{base}.txt"


def main():
    # Check if we're in config mode by looking at sys.argv.
    has_config = (
        hasattr(sys, "argv") and len(sys.argv) > 1 and sys.argv[1] == "config"
    )

    if has_config:
        parser = _build_parser()
        args, remaining = parser.parse_known_args()
        if getattr(args, "command", None) == "config":
            return _handle_config_dispatch(args)
        # Fall through to transcribe mode with config values.
        # Remaining positional will be handled by the TRANSACT path below.

    # Transcribe flow — use _parse_transcribe_args for backward compatibility.
    args = _parse_transcribe_args()
    cfg = _get_config() if hasattr(args, "use_ai") else None

    target_path = Path(args.TARGET).expanduser().resolve()
    video_files = _discover_video_files(target_path, recursive=args.recursive)

    if args.list_only:
        _handle_list_mode(args, video_files, target_path)
        return

    if not video_files:
        print("No .mp4 files found to transcribe.")
        raise SystemExit(1)

    out_dir = _prepare_output_dir(args.output_dir)
    client = load_client(
        local=cfg.ai.use_local if cfg else _TRANSCRIBE_LLM["USE_LOCAL"],
        api_base=(
            cfg.ai.api_base
            if cfg
            else _TRANSCRIBE_LLM.get("API_BASE", "http://localhost:8080/v1")
        ),
    )
    parsed_prefix = parse_prefix_parts(args.prefix)
    names_entries = _prepare_names_for_run(
        args, video_files, target_path, client, parsed_prefix
    )

    _transcribe_videos(
        video_files,
        client,
        parsed_prefix,
        names_entries,
        out_dir,
        args.smart_names,
        segment_duration_minutes=(
            cfg.audio.segment_duration_minutes if cfg else 10
        ),
    )
    print("Done!")


def _parse_transcribe_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe mp4 video(s) using Whisper-1"
    )
    parser.add_argument(
        "TARGET",
        help="Path to an .mp4 file or a directory containing .mp4 files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write transcripts (default: cwd)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        action="append",
        help=(
            "Composable prefix parts (repeatable). Format: "
            "text:VALUE | counter:N|NN|NNN|NNNN. Order is preserved. "
            "Example: -p 'text:Intro' -p 'counter:NN' -p 'text: '. "
            "Legacy 'sep:VALUE' is treated as text:VALUE."
        ),
    )
    parser.add_argument(
        "-l",
        "--list",
        dest="list_only",
        action="store_true",
        help="List discovered .mp4 files and exit",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Traverse subfolders of the target directory",
    )
    parser.add_argument(
        "--smart-names",
        dest="smart_names",
        action="store_true",
        help="Generate smart output names from directory structure",
    )
    parser.add_argument(
        "--use-ai",
        dest="use_ai",
        action="store_true",
        help="Use OpenAI to refine smart names (optional)",
    )
    parser.add_argument(
        "--names-file",
        dest="names_file",
        help=(
            "Path to cache file for proposed names (defaults to a hidden file "
            "in the target root)"
        ),
    )
    parser.add_argument(
        "--refresh-names",
        dest="refresh_names",
        action="store_true",
        help=(
            "Regenerate names for discovered files (overwrites cache entries)"
        ),
    )
    parsed, remaining = parser.parse_known_args()
    if getattr(parsed, "TARGET", None) is None and remaining:
        # Move the first unrecognized positional back to TARGET.
        for token in remaining:
            if not str(token).startswith("-"):
                parsed.TARGET = token
                break
    return parsed


def _discover_video_files(target_path: Path, recursive: bool) -> List[Path]:
    try:
        return find_video_files(target_path, recursive=recursive)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


def _handle_list_mode(args, video_files: List[Path], target_path: Path) -> None:
    if not video_files:
        print("No .mp4 files found.")
        return
    parsed_prefix = parse_prefix_parts(args.prefix)
    if args.smart_names:
        list_cfg = _TRANSCRIBE_CONFIG
        local_val = (
            list_cfg.ai.use_local if list_cfg else _TRANSCRIBE_LLM["USE_LOCAL"]
        )
        api_base_val = (
            list_cfg.ai.api_base
            if list_cfg
            else _TRANSCRIBE_LLM.get("API_BASE", "http://localhost:8080/v1")
        )
        client = (
            load_client(local=local_val, api_base=api_base_val)
            if args.use_ai
            else None
        )
        root, cache_path = _resolve_names_paths(args, target_path)
        existing = _load_existing_names(cache_path)
        mapping = _build_mapping_base(args, video_files, root, client, existing)
        combined = _combine_name_entries(
            video_files, existing, mapping, parsed_prefix, root
        )
        save_names_cache(
            cache_path,
            root,
            combined,
            meta={
                "use_ai": args.use_ai,
                "refreshed": args.refresh_names,
                "prefix_parts": args.prefix or [],
            },
        )
        print("Proposed names (saved). Edit the cache file to adjust:")
        print(f"Cache file: {cache_path}")
        for path in video_files:
            entry = combined.get(path)
            final = cache_get_final(entry)
            if not final:
                base = cache_get_base(entry, path.stem)
                final = make_output_filename(
                    path,
                    video_files.index(path) + 1,
                    parsed_prefix,
                    smart_base=base,
                )
            print(f"{path} -> {final}")
        return
    for idx, path in enumerate(video_files, start=1):
        preview = make_output_filename(
            path, idx, parsed_prefix, smart_base=None
        )
        print(f"{path} -> {preview}")


def _prepare_output_dir(output_dir: Optional[str]) -> Path:
    out_dir = (
        Path(output_dir).expanduser().resolve() if output_dir else Path.cwd()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _prepare_names_for_run(
    args,
    video_files: List[Path],
    target_path: Path,
    client,
    parsed_prefix,
) -> Dict[Path, Any]:
    if not args.smart_names:
        return {}
    root, cache_path = _resolve_names_paths(args, target_path)
    existing = _load_existing_names(cache_path)
    mapping = _build_mapping_base(args, video_files, root, client, existing)
    entries = _combine_name_entries(
        video_files, existing, mapping, parsed_prefix, root
    )
    save_names_cache(
        cache_path,
        root,
        entries,
        meta={
            "use_ai": args.use_ai,
            "note": "refreshed finals before transcription",
            "prefix_parts": args.prefix or [],
        },
    )
    return entries


def _transcribe_videos(
    video_files: List[Path],
    client,
    parsed_prefix,
    names_entries: Dict[Path, Any],
    out_dir: Path,
    use_smart_names: bool,
    segment_duration_minutes: int = 10,
) -> None:
    for idx, video in enumerate(video_files, start=1):
        print(f"Processing: {video.name}")
        try:
            transcript_text = transcribe_video_file(
                client, video, segment_duration_minutes=segment_duration_minutes
            )
        except Exception as exc:
            print(f"Failed to transcribe {video.name}: {exc}")
            continue
        if use_smart_names:
            entry = names_entries.get(video, {})
            out_name = cache_get_final(entry) or make_output_filename(
                video,
                idx,
                parsed_prefix,
                smart_base=cache_get_base(entry, video.stem),
            )
        else:
            out_name = make_output_filename(
                video, idx, parsed_prefix, smart_base=None
            )
        out_path = out_dir.joinpath(out_name)
        print(f"Saving transcript to: {out_path}")
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(transcript_text)


def _resolve_names_paths(args, target_path: Path) -> Tuple[Path, Path]:
    root = target_path if target_path.is_dir() else target_path.parent
    cache_path = (
        Path(args.names_file).expanduser().resolve()
        if args.names_file
        else default_names_cache_path(root)
    )
    return root, cache_path


def _load_existing_names(cache_path: Path) -> Dict[Path, Any]:
    raw = load_names_cache(cache_path)
    return {Path(k): v for k, v in raw.items()}


def _build_mapping_base(
    args,
    video_files: List[Path],
    root: Path,
    client,
    existing: Dict[Path, Any],
) -> Dict[Path, str]:
    if not args.smart_names:
        return {}
    effective_client = client if args.use_ai else None
    if args.refresh_names:
        return build_name_mapping(
            video_files, root, args.use_ai, effective_client
        )
    missing = [path for path in video_files if path not in existing]
    if not missing:
        return {}
    return build_name_mapping(missing, root, args.use_ai, effective_client)


def _combine_name_entries(
    video_files: List[Path],
    existing: Dict[Path, Any],
    mapping: Dict[Path, str],
    parsed_prefix,
    root: Path,
) -> Dict[Path, Any]:
    combined: Dict[Path, Any] = dict(existing)
    for idx, path in enumerate(video_files, start=1):
        base = mapping.get(path)
        if base is None:
            base = cache_get_base(existing.get(path), path.stem)
            if path not in existing:
                base = heuristic_smart_name(path, root)
        final = make_output_filename(path, idx, parsed_prefix, smart_base=base)
        combined[path] = {"base": base, "final": final}
    return combined


if __name__ == "__main__":  # pragma: no cover
    main()
