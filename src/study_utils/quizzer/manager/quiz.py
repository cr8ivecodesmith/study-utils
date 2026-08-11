import re
import json
import random

from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Sequence, Tuple, Iterable, Dict, Any

from ...core import load_client
from ..utils import _slugify


def _topic_source_paths(topic: Dict[str, object]) -> List[Path]:
    raw = topic.get("source_paths", []) or []
    if not isinstance(raw, list):
        return []
    paths: List[Path] = []
    for entry in raw:
        try:
            p = Path(entry)
        except Exception:
            continue
        if p.exists() and p.is_file():
            paths.append(p)
    return paths


def _read_topic_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _collect_heading_block(
    lines: List[str], name: str, heading_pattern: "re.Pattern[str]"
) -> List[str]:
    if not name:
        return []
    target = name.lower()
    captured: List[str] = []
    capturing = False
    for line in lines:
        match = heading_pattern.match(line)
        if match:
            heading = match.group(1).strip().strip("# ").lower()
            if capturing:
                break
            capturing = heading == target
            if capturing:
                captured.append(line)
            continue
        if capturing:
            if line.strip().startswith("#"):
                break
            captured.append(line)
    return captured


def _fallback_topic_lines(lines: List[str], name: str) -> List[str]:
    if not name:
        return []
    needle = name.lower()
    return [ln for ln in lines if needle in ln.lower()]


def _extract_topic_snippet(
    lines: List[str], name: str, pattern: "re.Pattern[str]"
) -> str:
    block = _collect_heading_block(lines, name, pattern)
    if not block:
        block = _fallback_topic_lines(lines, name)
    if not block:
        return ""
    return "\n".join(block).strip()


def _gather_topic_context(
    topic: Dict[str, object], max_chars: int = 4000
) -> str:
    """Read source_paths for a topic and extract relevant snippets.

    Heuristics:
    - If headings matching topic name exist, include lines until next heading
    - Else include lines containing the topic name
    - Concatenate across files up to max_chars
    """
    name = str(topic.get("name", "")).strip()
    pattern = re.compile(r"^\s*#+\s+(.+)$")
    parts: List[str] = []
    total = 0
    for path in _topic_source_paths(topic):
        text = _read_topic_file(path)
        if not text:
            continue
        snippet = _extract_topic_snippet(text.splitlines(), name, pattern)
        if not snippet:
            continue
        entry = f"From {path.name}:\n{snippet}"
        parts.append(entry)
        total += len(entry)
        if total >= max_chars:
            break
    out = "\n\n".join(parts)
    return out[:max_chars]


def _group_by_topic(bank: Sequence[Dict[str, object]]):
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for q in bank:
        groups[str(q.get("topic_id", ""))].append(q)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: str(x.get("id", "")))
    return dict(groups)


def select_questions(
    bank: Sequence[Dict[str, object]],
    *,
    strategy: str = "balanced",
    num: int = 10,
    per_topic_stats: Optional[Dict[str, Dict[str, int]]] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Select questions from a bank using a strategy.

    - balanced: round-robin across topics before repeats
    - random: uniform random over the bank
    - weakness: weighted toward topics with lower accuracy (1 - correctness)
    Deterministic when seed is provided.
    """
    if num <= 0 or not bank:
        return []
    rng = random.Random(seed)
    if strategy == "random":
        return [rng.choice(bank) for _ in range(num)]

    groups = _group_by_topic(bank)
    topics = sorted(groups.keys())
    if not topics:
        return []

    if strategy == "weakness":
        weights: List[float] = []
        for t in topics:
            s = (
                per_topic_stats.get(t, {"asked": 0, "correct": 0})
                if per_topic_stats
                else {"asked": 0, "correct": 0}
            )
            asked = max(1, int(s.get("asked", 0)))
            correct = max(0, int(s.get("correct", 0)))
            acc = correct / asked
            w = max(0.05, 1.0 - acc)
            weights.append(w)
        selected: List[Dict[str, object]] = []
        for _ in range(num):
            t = rng.choices(topics, weights=weights, k=1)[0]
            selected.append(rng.choice(groups[t]))
        return selected

    # balanced
    selected: List[Dict[str, object]] = []
    topic_indices = {t: 0 for t in topics}
    i = 0
    while len(selected) < num:
        t = topics[i % len(topics)]
        g = groups[t]
        idx = topic_indices[t]
        if idx >= len(g):
            idx = rng.randrange(0, len(g)) if g else 0
        selected.append(g[idx])
        topic_indices[t] = (idx + 1) % max(1, len(g))
        i += 1
    return selected[:num]


def generate_questions(
    topics: Sequence[Dict[str, object]],
    *,
    per_topic: int = 3,
    qtype: str = "mcq",
    client: object = None,
    ensure_coverage: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Generate a question bank from topics.

    Call AI generation per topic and concatenate results. For now only MCQ is
    supported.
    """
    if qtype != "mcq" or per_topic <= 0:
        return []
    bank: List[Dict[str, object]] = []
    for t in topics:
        context = _gather_topic_context(t)
        qs = ai_generate_mcqs_for_topic(
            t, n=per_topic, client=client, seed=seed, context=context
        )
        if ensure_coverage and not qs:
            topic_label = t.get("name", t.get("id", "this topic"))
            stem = f"Which of the following relates to {topic_label}?"
            placeholder = {
                "id": f"{t.get('id')}-ph-1",
                "topic_id": str(t.get("id")),
                "type": "mcq",
                "stem": stem,
                "choices": [
                    {"key": "A", "text": str(t.get("name", "Concept"))},
                    {"key": "B", "text": "None of the above"},
                ],
                "answer": "A",
                "explanation": "",
            }
            try:
                validate_mcq(placeholder)
                qs = [placeholder]
            except Exception:
                qs = []
        bank.extend(qs)
    return bank


def validate_mcq(q: Dict[str, object]) -> None:
    """Validate an MCQ question dict.

    Required keys: id, topic_id, type=='mcq', stem, choices (list of
    {key,text}), answer (str), explanation (str).
    - choices keys must be unique, 2–6 options, keys like 'A'..'F'
    - exactly one answer and it must be among choices
    Raises ValueError with actionable messages when invalid.
    """
    if not isinstance(q, dict):
        raise ValueError("question must be a dict")
    if q.get("type") != "mcq":
        raise ValueError("type must be 'mcq'")
    choices = q.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list")
    keys = [str(c.get("key", "")).upper() for c in choices]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate choice keys detected")
    if not all(k and len(k) == 1 and k.isalpha() for k in keys):
        raise ValueError("choice keys must be single letters (A..Z)")
    if not all(str(c.get("text", "")).strip() for c in choices):
        raise ValueError("choice text must be non-empty")
    ans = q.get("answer")
    if isinstance(ans, list):
        raise ValueError("MCQ requires exactly one answer, not multiple")
    if not isinstance(ans, str) or not ans:
        raise ValueError("answer is required")
    if ans.upper() not in set(keys):
        raise ValueError("answer must match one of the choice keys")


def _ensure_ai_client(client: Optional[object]) -> Optional[object]:
    if client is not None:
        return client
    if load_client is None:
        return None
    try:
        return load_client(
            local=True,
            api_base="http://localhost:8080/v1",
        )
    except Exception:
        return None


def _build_mcq_prompts(
    topic_name: str,
    n: int,
    prompt: Optional[str],
    context: Optional[str],
) -> Tuple[str, str]:
    sys_prompt = "You generate high-quality multiple-choice study questions."
    ctx_block = ""
    if isinstance(context, str):
        ctx = context.strip()
        if ctx:
            ctx_block = "\n\nContext (from source materials):\n" + ctx
    base_instructions = (
        prompt
        or "Create concise multiple-choice questions covering the topic. "
        "Output a JSON array of objects."
    )
    schema_line = (
        '{"stem": str, "choices": [str or {"key": "A", "text": str}], '
        '"answer": str, "explanation": str}\n'
    )
    constraints = (
        "Constraints: single correct answer; plausible distractors; avoid "
        "ambiguity; keep stems under 200 chars."
    )
    user_prompt = (
        f"{base_instructions}\n\nSchema:\n{schema_line}"
        f"Topic: {topic_name}\n"
        f"Count: {n}\n"
        f"{constraints}"
        f"{ctx_block}"
    )
    return sys_prompt, user_prompt


def _chat_completion_content(
    client: object,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_content = resp.choices[0].message.content  # type: ignore[index]
        return (raw_content or "").strip()
    except Exception:
        return ""


def _extract_json_array(content: str) -> List[Any]:
    if not content:
        return []
    fenced = re.search(r"```(?:json)?\s*(.+?)```", content, re.DOTALL)
    payload = fenced.group(1) if fenced else content
    try:
        data = json.loads(payload)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _normalize_choice_list(raw_choices: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(raw_choices, list):
        for i, ch in enumerate(raw_choices):
            if isinstance(ch, dict):
                key = str(ch.get("key", "")).strip() or chr(ord("A") + i)
                text = str(ch.get("text", "")).strip()
            else:
                key = chr(ord("A") + i)
                text = str(ch).strip()
            if not text:
                continue
            out.append({"key": key.upper()[:1], "text": text})
    return out


def _resolve_mcq_answer(raw_answer: Any, choices: List[Dict[str, str]]) -> str:
    if isinstance(raw_answer, int) and 0 <= raw_answer < len(choices):
        return choices[raw_answer]["key"]
    if isinstance(raw_answer, str):
        candidate = raw_answer.strip().upper()
        keys = {c["key"] for c in choices}
        if candidate in keys:
            return candidate
        for c in choices:
            if candidate == c["text"].strip().upper():
                return c["key"]
    return str(raw_answer or "").strip().upper()


def _build_mcq_items(
    records: List[Any],
    *,
    n: int,
    topic_id: str,
    seed: Optional[int],
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    items: List[Dict[str, object]] = []
    for rec in records:
        if len(items) >= n:
            break
        if not isinstance(rec, dict):
            continue
        stem = str(rec.get("stem", "")).strip()
        if not stem:
            continue
        choices = _normalize_choice_list(rec.get("choices", []))
        if not choices:
            continue
        answer = _resolve_mcq_answer(rec.get("answer"), choices)
        explanation = str(rec.get("explanation", "")).strip()
        generated_id = (
            rec.get("id")
            or f"{topic_id}-{rng.randint(10000, 99999)}-{len(items)}"
        )
        candidate = {
            "id": generated_id,
            "topic_id": topic_id,
            "type": "mcq",
            "stem": stem,
            "choices": choices,
            "answer": answer,
            "explanation": explanation,
        }
        try:
            validate_mcq(candidate)
        except Exception:
            continue
        items.append(candidate)
    return items


def ai_generate_mcqs_for_topic(
    topic: Dict[str, object],
    n: int = 3,
    *,
    client: object = None,
    prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    seed: Optional[int] = None,
    context: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Generate n MCQ questions for a topic using an AI model.

    Return up to ``n`` validated MCQ dicts with normalized shape. Invalid items
    are skipped.
    """
    if n <= 0:
        return []
    resolved_client = _ensure_ai_client(client)
    if resolved_client is None:
        return []

    topic_id = str(topic.get("id") or _slugify(str(topic.get("name", "topic"))))
    topic_name = str(topic.get("name") or topic_id)
    sys_prompt, user_prompt = _build_mcq_prompts(topic_name, n, prompt, context)
    content = _chat_completion_content(
        resolved_client,
        model=model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=800,
    )
    content = content.replace("\n", "")
    if not content:
        return []

    data = _extract_json_array(content)
    items = _build_mcq_items(data, n=n, topic_id=topic_id, seed=seed)
    return items[:n]


def _summarize_topic_sources(
    sources: Iterable[Tuple[Path, str]],
    *,
    source_max_lines: Optional[int],
    source_max_lines_chars: Optional[int],
) -> str:
    parts: List[str] = []
    max_lines = source_max_lines or 0
    max_chars = source_max_lines_chars or 0
    for path, text in sources:
        snippet_lines = (text or "").splitlines()
        if max_lines:
            snippet_lines = [ln for ln in snippet_lines if ln.strip()][
                :max_lines
            ]
        joined = "\n".join(snippet_lines)
        if max_chars and len(joined) > max_chars:
            joined = joined[:max_chars]
        parts.append(f"File: {path.name}\n{joined}\n")
    return "\n\n".join(parts)


def _parse_topic_suggestions(
    data: List[Any], limit: Optional[int]
) -> List[Dict[str, object]]:
    suggestions: List[Dict[str, object]] = []
    max_items = limit or len(data)
    for item in data:
        if len(suggestions) >= max_items:
            break
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            suggestions.append(
                {
                    "id": _slugify(name),
                    "name": name,
                    "description": "",
                    "source_paths": [],
                    "created_at": "",
                }
            )
            continue
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            source_paths = item.get("source_paths", [])
            clean_paths = source_paths if isinstance(source_paths, list) else []
            suggestions.append(
                {
                    "id": _slugify(name),
                    "name": name,
                    "description": str(item.get("description", "")),
                    "source_paths": clean_paths,
                    "created_at": "",
                }
            )
    return suggestions


def ai_extract_topics(
    sources: Iterable[Tuple[Path, str]],
    *,
    k: Optional[int] = None,
    client: object = None,
    prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    seed: Optional[int] = None,
    source_max_lines: Optional[int] = 20,
    source_max_lines_chars: Optional[int] = 400,
) -> List[Dict[str, object]]:
    """Suggest topics from raw text using an AI model.

    Will parse model output (JSON array of names or objects) and
    return a list of topic dicts with id/name/description/source_paths.
    """
    resolved_client = _ensure_ai_client(client)
    if resolved_client is None:
        return []

    summarized = _summarize_topic_sources(
        sources,
        source_max_lines=source_max_lines,
        source_max_lines_chars=source_max_lines_chars,
    )
    base_prompt = (
        prompt
        or "Suggest concise study topics from the following notes. "
        "Output a JSON array of objects with name and optional description."
    )
    user_prompt = (base_prompt + "\n\n" + summarized).strip()

    content = _chat_completion_content(
        resolved_client,
        model=model,
        system_prompt="You extract clean, deduplicated topic names.",
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=600,
    )
    if not content:
        return []

    data = _extract_json_array(content)
    return _parse_topic_suggestions(data, k)


def extract_topics(
    sources: Iterable[Tuple[Path, str]],
    *,
    use_ai: bool = False,
    client: object = None,
    ai_prompt: Optional[str] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Extract topics from Markdown sources.

    Strategy:
    - Use H1/H2 headings (lines starting with '#' or '##') as candidate topics
    - Deduplicate by slug; keep first occurrence
    - Record simple description (first line after heading if available)
    - Track source file path for traceability
    """
    source_list = list(sources)
    heuristic = _collect_heading_topics(source_list)
    if not use_ai:
        return heuristic
    ai_topics = ai_extract_topics(
        source_list, k=k, client=client, prompt=ai_prompt, seed=seed
    )
    return _merge_topic_lists(heuristic, ai_topics)


def _collect_heading_topics(
    sources: Sequence[Tuple[Path, str]],
) -> List[Dict[str, object]]:
    topics: Dict[str, Dict[str, object]] = {}
    for path, text in sources:
        lines = (text or "").splitlines()
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line or not line.startswith("#"):
                continue
            match = re.match(r"^(#+)\s+(.*)$", line)
            if not match:
                continue
            level = len(match.group(1))
            if level > 2:
                continue
            name = match.group(2).strip().rstrip("# ")
            if not name:
                continue
            slug = _slugify(name)
            entry = topics.get(slug)
            if entry:
                existing_paths = entry.get("source_paths", [])
                path_set = (
                    {str(p) for p in existing_paths}
                    if isinstance(existing_paths, list)
                    else set()
                )
                path_set.add(str(path))
                entry["source_paths"] = sorted(path_set)
                continue
            description = ""
            for j in range(i + 1, min(i + 6, len(lines))):
                nxt = lines[j].strip()
                if nxt and not nxt.startswith("#"):
                    description = nxt
                    break
            topics[slug] = {
                "id": slug,
                "name": name,
                "description": description,
                "source_paths": [str(path)],
                "created_at": "",
            }
    return list(topics.values())


def _merge_topic_lists(
    heuristic: Sequence[Dict[str, object]],
    ai_topics: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for item in heuristic:
        merged[str(item["id"])] = dict(item)
    for topic in ai_topics:
        slug = str(topic.get("id") or _slugify(str(topic.get("name", ""))))
        existing = merged.get(slug)
        if existing:
            ai_name = topic.get("name")
            if ai_name and len(str(ai_name)) > len(
                str(existing.get("name", ""))
            ):
                existing["name"] = ai_name
            combined_paths = set(existing.get("source_paths", [])) | set(
                topic.get("source_paths", []) or []
            )
            existing["source_paths"] = sorted(combined_paths)
            continue
        merged[slug] = {
            "id": slug,
            "name": topic.get("name") or slug,
            "description": topic.get("description", ""),
            "source_paths": topic.get("source_paths", []) or [],
            "created_at": "",
        }
    return list(merged.values())
