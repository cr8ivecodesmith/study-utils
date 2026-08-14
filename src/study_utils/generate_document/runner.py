"""Document generation orchestration utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from study_utils.core import iter_text_files, load_client, read_text_file

from .config import load_documents_config


def build_reference_block(files: Sequence[Tuple[Path, str]]) -> str:
    """Build a single string that contains all reference files and contents."""
    parts: List[str] = []
    for path, content in files:
        parts.append(
            "\n".join(
                [
                    "\n---",
                    f"File: {path.name}",
                    f"Path: {path}",
                    "Content:",
                    content,
                ]
            )
        )
    return "\n".join(parts)


def build_messages(
    doc_cfg: Dict[str, str],
    files: Sequence[Tuple[Path, str]],
) -> List[Dict[str, str]]:
    """Construct chat messages for the selected document type."""
    system = (
        "You are an expert writing assistant. "
        "Follow the user's instructions exactly. "
        "Only use the provided reference content. "
        "Output valid, readable Markdown for humans."
    )
    prompt = doc_cfg["prompt"]
    refs = build_reference_block(files)
    user = (
        f"Instructions:\n{prompt}\n\n"
        "Reference files are provided below. "
        "Use only these sources; do not invent facts.\n\n"
        f"{refs}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_document(
    doc_type: str,
    output_path: Path,
    inputs: Sequence[Path],
    extensions: Set[str],
    level_limit: int,
    config_path: Path,
) -> int:
    """Generate a document and write it to `output_path`. Returns files used."""
    cfg_all = load_documents_config(config_path)
    if doc_type not in cfg_all:
        known = ", ".join(sorted(cfg_all.keys()))
        raise ValueError(f"Unknown document type '{doc_type}'. Known: {known}")

    discovered = list(iter_text_files(inputs, extensions, level_limit))
    if not discovered:
        raise FileNotFoundError("No matching reference files found")
    ref_pairs: List[Tuple[Path, str]] = [
        (path, read_text_file(path)) for path in discovered
    ]

    llm_cfg = cfg_all.llm

    client = load_client(
        local=llm_cfg.use_local,
        api_base=llm_cfg.api_base,
    )
    doc_cfg = cfg_all[doc_type]
    messages = build_messages(doc_cfg, ref_pairs)
    model = doc_cfg.get("model") or os.getenv(
        "OPENAI_TITLE_MODEL",
        "gpt-4o-mini",
    )
    temperature = llm_cfg.temperature

    if "gpt-5" in model:
        params_temperature = 1.0
        max_tokens_kw = "max_completion_tokens"
        max_tokens_val = 8192
    else:
        params_temperature = temperature
        max_tokens_kw = "max_tokens"
        max_tokens_val = llm_cfg.max_tokens

    params = {
        "model": model,
        "messages": messages,
        "temperature": params_temperature,
        max_tokens_kw: max_tokens_val,
    }

    resp = client.chat.completions.create(**params)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError(
            "AI returned empty content; check API key/model and inputs"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return len(ref_pairs)
