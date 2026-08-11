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

import os
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import json
import re
from datetime import datetime, timezone
from tempfile import gettempdir

from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks

from .core import load_client

_TRANSCRIBE_LLM = {
    "USE_LOCAL": True,
    "API_BASE": "http://localhost:8080/v1",
    "MODEL": os.getenv("TRANSCRIPTION_MODEL", "whisper-3"),
}


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
    file_path: Path, exist_delete: bool = True
) -> List[Path]:
    """Extract audio from an mp4 and split into ~10 minute mp3 segments.

    Returns a list of mp3 segment file paths. Creates a transient directory
    `<video_stem>_segments` in the current working directory and fills it with
    the mp3 chunks. Caller is responsible for removing this directory when
    done.
    """
    # Load audio track from the video (requires ffmpeg via pydub)
    full_audio = AudioSegment.from_file(file_path, format="mp4")
    segment_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
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
    """Transcribe a single audio file using the configured model (whisper-3 by default) and return plain text."""
    response = client.audio.transcriptions.create(
        model=os.getenv("TRANSCRIPTION_MODEL", "whisper-3"),
        file=audio_path.open("rb"),
        response_format="text",
    )
    # SDK returns a plain string when response_format='text'
    return (
        response.strip() if isinstance(response, str) else str(response).strip()
    )


def transcribe_video_file(client: OpenAI, video_path: Path) -> str:
    """Transcribe an mp4 by chunking its audio and concatenating results."""
    print(f"Splitting audio for {video_path.name} ...")
    segments = split_video_to_audio_segments(video_path)
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
    args = _parse_transcribe_args()
    target_path = Path(args.TARGET).expanduser().resolve()
    video_files = _discover_video_files(target_path, args.recursive)

    if args.list_only:
        _handle_list_mode(args, video_files, target_path)
        return

    if not video_files:
        print("No .mp4 files found to transcribe.")
        raise SystemExit(1)

    out_dir = _prepare_output_dir(args.output_dir)
    client = load_client(
        local=_TRANSCRIBE_LLM["USE_LOCAL"],
        api_base=_TRANSCRIBE_LLM["API_BASE"],
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
    return parser.parse_args()


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
        client = (
            load_client(
                local=_TRANSCRIBE_LLM["USE_LOCAL"],
                api_base=_TRANSCRIBE_LLM["API_BASE"],
            )
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
) -> None:
    for idx, video in enumerate(video_files, start=1):
        print(f"Processing: {video.name}")
        try:
            transcript_text = transcribe_video_file(client, video)
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
