"""Combine text files into a single output file.

Implements the utility described in spec.md (text_combiner.py):
- Required: output filename (first positional)
- Required: one or more files and/or directories to source text files
- Optional controls for extensions, depth, ordering, separators, and section
  titles

Design notes:
- Uses pathlib and small, testable functions
- Keeps core logic pure; isolates I/O in main combine function
- OpenAI integration is optional and only used when a smart section title is
  requested
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Set

from .core import (
    iter_text_files,
    load_client,
    order_files,
    parse_extensions,
    read_text_file,
)

_TEXT_COMBINER_LLM = {
    "USE_LOCAL": True,
    "API_BASE": "http://localhost:8080/v1",
}

try:  # Optional dependency: openai may not be installed in local envs.
    from openai import BadRequestError as OpenAIBadRequestError  # type: ignore
except Exception:  # pragma: no cover - fallback when OpenAI SDK missing
    OpenAIBadRequestError = Exception  # type: ignore[misc,assignment]


# -----------------------------
# Types and simple structures
# -----------------------------


@dataclass(frozen=True)
class CombineOptions:
    extensions: Set[str]
    level_limit: int
    combine_by: str  # "EOF" | "NEW"
    order_by: Optional[
        str
    ]  # None | created | -created | modified | -modified | name | -name
    section_title: Optional[
        str
    ]  # None | filename | smart-filename | smart-content
    section_title_format: Optional[str]  # None | title | lower | upper
    section_title_heading: Optional[
        str
    ]  # None or markdown heading string like '#' or '##'


# -----------------------------
# Parsing helpers
# -----------------------------


def parse_order_by(value: Optional[str]) -> Optional[str]:
    """Validate and normalize the order-by string.

    Accepts: created, -created, modified, -modified, name, -name
    Returns the normalized string or None.
    """
    if not value:
        return None
    v = value.strip().lower()
    valid = {"created", "-created", "modified", "-modified", "name", "-name"}
    if v not in valid:
        raise ValueError(
            "--order-by must be one of: created, -created, modified, "
            "-modified, name, -name"
        )
    return v


def parse_heading(value: Optional[str]) -> Optional[str]:
    """Normalize a markdown heading spec.

    Accept strings like '#', '##', ..., '######'. Return the same string if
    valid; return ``None`` when not provided.
    """
    if not value:
        return None
    v = value.strip()
    if not re.fullmatch(r"#{1,6}", v):
        raise ValueError(
            "--section-title-heading must be one of: #, ##, ###, ####, "
            "#####, ######"
        )
    return v


# -----------------------------
# Section title generation
# -----------------------------


def _apply_title_format(text: str, fmt: Optional[str]) -> str:
    if not fmt:
        return text
    f = fmt.lower()
    if f == "title":
        return text.title()
    if f == "lower":
        return text.lower()
    if f == "upper":
        return text.upper()
    return text


def _with_heading(text: str, heading: Optional[str]) -> str:
    return f"{heading} {text}" if heading else text


def _ai_title_from_filename(path: Path) -> Optional[str]:
    try:
        client = load_client(
            local=_TEXT_COMBINER_LLM["USE_LOCAL"],
            api_base=_TEXT_COMBINER_LLM["API_BASE"],
        )
    except Exception:
        return None
    prompt = (
        "Generate a concise section title (<= 80 chars) based only on this "
        "file name. Avoid quotes and punctuation-heavy output; return only "
        "the title.\n\n"
        f"Filename: {path.name}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You craft concise, human-friendly section titles."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        title = (resp.choices[0].message.content or "").strip()
        title = re.sub(r"[\r\n]+", " ", title)
        return title[:120] if title else None
    except OpenAIBadRequestError:
        return None
    except Exception:
        return None


def _ai_title_from_content(content: str, filename: str) -> Optional[str]:
    try:
        client = load_client(
            local=_TEXT_COMBINER_LLM["USE_LOCAL"],
            api_base=_TEXT_COMBINER_LLM["API_BASE"],
        )
    except Exception:
        return None

    snippet = content[:4000]
    prompt = (
        "Generate a concise section title (<= 80 chars) from the following "
        "text content. Prefer the main topic or a clear heading. Avoid "
        "quotes; return only the title.\n\n"
        f"Filename: {filename}\n"
        f"Content:\n{snippet}"
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You craft concise, human-friendly section titles."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=64,
        )
        title = (resp.choices[0].message.content or "").strip()
        title = re.sub(r"[\r\n]+", " ", title)
        return title[:120] if title else None
    except OpenAIBadRequestError:
        return None
    except Exception:
        return None


def make_section_title(
    kind: Optional[str],
    file_path: Path,
    file_content: Optional[str],
    fmt: Optional[str],
    heading: Optional[str],
) -> Optional[str]:
    """Return a formatted section title for a file, or None if disabled.

    - kind None -> None
    - filename -> uses file stem
    - smart-filename -> tries AI based on filename, falls back to stem
    - smart-content -> tries AI based on content, falls back to stem
    Applies `fmt` and `heading` if provided.
    """
    if not kind:
        return None
    base: Optional[str] = None
    k = kind.lower()
    if k == "filename":
        base = file_path.stem
    elif k == "smart-filename":
        base = _ai_title_from_filename(file_path) or file_path.stem
    elif k == "smart-content":
        base = (
            _ai_title_from_content(file_content or "", file_path.name)
            or file_path.stem
        )
    else:
        # unknown -> treat as filename
        base = file_path.stem

    text = _apply_title_format(base, fmt)
    return _with_heading(text, heading)


# -----------------------------
# Combining
# -----------------------------
def combine_files(
    files: Sequence[Path],
    output_path: Path,
    options: CombineOptions,
) -> int:
    """Combine files into `output_path`.

    Returns the number of files combined.
    Behavior:
    - combine_by NEW: insert a single newline between file contents
    - combine_by EOF: append as-is
    - when section_title is set: ignore combine_by; write (for each file):
      [optional leading newline if not first]\n
      <section_title>\n
      <content>
    """
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for idx, f in enumerate(files):
            content = read_text_file(f)
            if options.section_title:
                title_line = make_section_title(
                    options.section_title,
                    f,
                    content,
                    options.section_title_format,
                    options.section_title_heading,
                )
                if idx > 0:
                    # Ensure a blank line between the previous content and
                    # the next title.
                    out.write("\n\n")
                if title_line:
                    out.write(title_line)
                    out.write("\n\n")
                out.write(content)
            else:
                if idx > 0 and options.combine_by.upper() == "NEW":
                    out.write("\n")
                out.write(content)
            count += 1
    return count


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine text files into one output file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "OUTPUT",
        help="Output file path (will be created or overwritten)",
    )
    p.add_argument(
        "INPUTS",
        nargs="+",
        help="One or more input files and/or directories to scan",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        help="File extensions to include (e.g. txt md). Defaults to txt only",
    )
    p.add_argument(
        "--level-limit",
        type=int,
        default=0,
        help=(
            "Directory depth to traverse for directories: 0=no limit; "
            "1=files directly under; 2=one extra sublevel; etc."
        ),
    )
    p.add_argument(
        "--combine-by",
        choices=["EOF", "NEW"],
        default="NEW",
        help="How to combine files when no section title is used",
    )
    p.add_argument(
        "--order-by",
        help=(
            "Order files by: created | -created | modified | -modified | "
            "name | -name"
        ),
    )
    p.add_argument(
        "--section-title",
        choices=["filename", "smart-filename", "smart-content"],
        help="Insert a section title before each file's contents",
    )
    p.add_argument(
        "--section-title-format",
        choices=["title", "lower", "upper"],
        help="Format applied to section titles",
    )
    p.add_argument(
        "--section-title-heading",
        help=(
            "Markdown heading prefix for titles, e.g. '#', '##'. Default is "
            "no heading"
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_path = Path(args.OUTPUT).expanduser().resolve()
    input_paths = [Path(s).expanduser().resolve() for s in args.INPUTS]

    try:
        extensions = parse_extensions(args.extensions, default={"txt"})
        order_by = parse_order_by(args.order_by)
        heading = parse_heading(args.section_title_heading)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)

    # Discover files
    try:
        discovered = list(
            iter_text_files(input_paths, extensions, args.level_limit)
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)

    if not discovered:
        print("No matching files found.")
        raise SystemExit(1)

    files = order_files(discovered, order_by)

    options = CombineOptions(
        extensions=extensions,
        level_limit=args.level_limit,
        combine_by=args.combine_by,
        order_by=order_by,
        section_title=args.section_title,
        section_title_format=args.section_title_format,
        section_title_heading=heading,
    )

    try:
        count = combine_files(files, output_path, options)
    except Exception as exc:
        print(f"Failed to combine files: {exc}")
        raise SystemExit(1)

    print(f"Combined {count} file(s) -> {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
