from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from study_utils.quizzer.manager import quiz as qm


def test_topic_source_paths_and_context(tmp_path: Path) -> None:
    existing = tmp_path / "topic.md"
    existing.write_text("# Heading\nDetails\nSecond line\n", encoding="utf-8")
    topic = {"source_paths": [str(existing), str(tmp_path / "missing"), 123]}
    paths = qm._topic_source_paths(topic)
    assert paths == [existing]

    snippet = qm._gather_topic_context(
        {"name": "Heading", "source_paths": [str(existing)]}, max_chars=20
    )
    assert "from" in snippet.lower()


def test_collect_heading_block_fallback() -> None:
    lines = ["# Intro", "Line", "# Next", "Another"]
    pattern = __import__("re").compile(r"^\s*#+\s+(.+)$")
    assert qm._collect_heading_block(lines, "Intro", pattern) == [
        "# Intro",
        "Line",
    ]
    assert qm._fallback_topic_lines(lines, "Another") == ["Another"]


def test_group_by_topic_and_selection_edges() -> None:
    bank = [
        {"id": "1", "topic_id": "a"},
        {"id": "2", "topic_id": "b"},
        {"id": "3", "topic_id": "a"},
    ]
    grouped = qm._group_by_topic(bank)
    assert grouped["a"][1]["id"] == "3"
    assert qm.select_questions([], num=3) == []
    assert qm.select_questions(bank[:1], num=0) == []


def test_generate_questions_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [{"id": "intro", "name": "Intro"}]
    monkeypatch.setattr(qm, "ai_generate_mcqs_for_topic", lambda *a, **k: [])
    out = qm.generate_questions(topics, per_topic=1, ensure_coverage=True)
    assert out and out[0]["topic_id"] == "intro"


def test_validate_mcq_failure_paths() -> None:
    with pytest.raises(ValueError):
        qm.validate_mcq("not a dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        qm.validate_mcq({"type": "text"})
    with pytest.raises(ValueError):
        qm.validate_mcq({"type": "mcq", "choices": "A"})
    with pytest.raises(ValueError):
        qm.validate_mcq({"type": "mcq", "choices": [{"key": "1", "text": ""}]})
    with pytest.raises(ValueError):
        qm.validate_mcq(
            {
                "type": "mcq",
                "choices": [{"key": "A", "text": ""}],
                "answer": "A",
            }
        )
    with pytest.raises(ValueError):
        qm.validate_mcq(
            {
                "type": "mcq",
                "choices": [{"key": "A", "text": "Ans"}],
                "answer": "Z",
            }
        )


def test_ensure_ai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qm, "load_client", lambda local=True, api_base="": "client"
    )
    assert qm._ensure_ai_client(None) == "client"
    monkeypatch.setattr(
        qm, "load_client", lambda local=False, api_base=None: (_ for _ in ()).throw(RuntimeError)
    )
    assert qm._ensure_ai_client(None) is None
    monkeypatch.setattr(qm, "load_client", None)
    assert qm._ensure_ai_client(None) is None


def test_build_prompts_and_chat_completion() -> None:
    sys, user = qm._build_mcq_prompts("Topic", 2, None, "ctx")
    assert "Topic" in user and "ctx" in user

    class StubClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):  # pragma: no cover - deliberate failure
                    raise RuntimeError("boom")

            completions = Completions()

        chat = Chat()

    assert (
        qm._chat_completion_content(
            StubClient(),
            model="m",
            system_prompt="s",
            user_prompt="u",
            temperature=0.1,
            max_tokens=10,
        )
        == ""
    )


def test_extract_json_array_and_choice_utils() -> None:
    payload = '```json\n[{"stem":"Q","choices":["A","B"],"answer":0}]```'
    arr = qm._extract_json_array(payload)
    assert len(arr) == 1

    choices = qm._normalize_choice_list(["a", {"key": "b", "text": "Option"}])
    assert choices[0]["key"] == "A" and choices[1]["key"] == "B"
    assert qm._resolve_mcq_answer(0, choices) == "A"
    assert qm._resolve_mcq_answer("option", choices) == "B"

    records = [
        {"stem": "", "choices": []},
        {"stem": "Q", "choices": choices, "answer": "A", "explanation": ""},
    ]
    built = qm._build_mcq_items(records, n=1, topic_id="t", seed=1)
    assert len(built) == 1 and built[0]["topic_id"] == "t"


def test_ai_generate_mcqs_for_topic_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert qm.ai_generate_mcqs_for_topic({"id": "t"}, n=0) == []
    monkeypatch.setattr(qm, "_ensure_ai_client", lambda *_: None)
    assert qm.ai_generate_mcqs_for_topic({"id": "t"}, n=2) == []

    class StubClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="[]")
                            )
                        ]
                    )

            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(qm, "_ensure_ai_client", lambda client: client)
    assert (
        qm.ai_generate_mcqs_for_topic({"id": "t"}, n=1, client=StubClient())
        == []
    )


def test_summarize_topic_sources_and_parse_suggestions(tmp_path: Path) -> None:
    file_a = tmp_path / "a.md"
    file_a.write_text("Line1\nLine2\nLine3", encoding="utf-8")
    summary = qm._summarize_topic_sources(
        [(file_a, file_a.read_text())],
        source_max_lines=1,
        source_max_lines_chars=4,
    )
    assert "Line" in summary

    suggestions = qm._parse_topic_suggestions(
        ["Topic", {"name": "Other", "source_paths": ["x"]}], limit=1
    )
    assert suggestions[0]["name"] == "Topic"


def test_ai_extract_topics_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qm, "_ensure_ai_client", lambda *_: None)
    assert qm.ai_extract_topics([], client=None) == []


def test_extract_topics_merge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_a = tmp_path / "a.md"
    file_a.write_text("# Intro\nBody\n# Basics\nDetails", encoding="utf-8")
    pairs = [(file_a, file_a.read_text())]

    monkeypatch.setattr(
        qm,
        "ai_extract_topics",
        lambda *a, **k: [
            {"id": "intro", "name": "Intro"},
            {"id": "extra", "name": "Extra"},
        ],
    )
    topics = qm.extract_topics(pairs, use_ai=True, client=None)
    names = {t["name"] for t in topics}
    assert {"Intro", "Basics", "Extra"}.issubset(names)


def test_topic_source_paths_non_list_and_read_error(tmp_path: Path) -> None:
    assert qm._topic_source_paths({"source_paths": "bad"}) == []
    assert qm._read_topic_file(tmp_path / "nope.txt") == ""


def test_extract_topic_snippet_no_match() -> None:
    lines = ["Line one", "Line two"]
    pattern = __import__("re").compile(r"^#\s+(.*)$")
    assert qm._extract_topic_snippet(lines, "Missing", pattern) == ""


def test_build_mcq_items_skips_invalid_entries() -> None:
    records = [
        "not a dict",
        {"stem": "", "choices": ["A"]},
        {"stem": "Valid", "choices": [], "answer": "A"},
        {"stem": "Valid", "choices": ["Answer"], "answer": "A"},
    ]
    items = qm._build_mcq_items(records, n=2, topic_id="t", seed=1)
    assert len(items) == 1


def test_parse_topic_suggestions_skips_blank() -> None:
    suggestions = qm._parse_topic_suggestions(
        ["", "Topic", {"name": ""}], limit=5
    )
    assert suggestions[0]["name"] == "Topic"


def test_merge_topic_lists_merges_paths() -> None:
    heuristic = [
        {
            "id": "intro",
            "name": "Intro",
            "source_paths": ["a.md"],
            "description": "",
        }
    ]
    ai_topics = [
        {
            "id": "intro",
            "name": "Introduction",
            "source_paths": ["b.md"],
            "description": "",
        }
    ]
    merged = qm._merge_topic_lists(heuristic, ai_topics)
    merged_entry = next(item for item in merged if item["id"] == "intro")
    assert merged_entry["name"] == "Introduction"
    assert set(merged_entry["source_paths"]) == {"a.md", "b.md"}


def test_select_questions_group_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qm, "_group_by_topic", lambda bank: {"t": []})
    with pytest.raises(IndexError):
        qm.select_questions([{"id": "1", "topic_id": "t"}], num=2)


def test_extract_json_array_errors() -> None:
    assert qm._extract_json_array("") == []
    assert qm._extract_json_array("not json") == []


def test_build_mcq_items_with_invalid_answer() -> None:
    records = [
        {"stem": "Invalid", "choices": ["A"], "answer": None},
        {
            "stem": "Bad",
            "choices": [{"key": "A", "text": "Ans"}],
            "answer": "Z",
        },
        {
            "stem": "Good",
            "choices": [{"key": "A", "text": "Ans"}],
            "answer": "A",
        },
    ]
    items = qm._build_mcq_items(records, n=2, topic_id="t", seed=0)
    assert len(items) == 1


def test_ai_generate_mcqs_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubClient:
        pass

    monkeypatch.setattr(qm, "_ensure_ai_client", lambda client: client)
    monkeypatch.setattr(qm, "_chat_completion_content", lambda *a, **k: "")
    assert (
        qm.ai_generate_mcqs_for_topic({"id": "t"}, n=1, client=StubClient())
        == []
    )


def test_ai_extract_topics_empty_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(qm, "_ensure_ai_client", lambda client: client)
    monkeypatch.setattr(qm, "_chat_completion_content", lambda *a, **k: "")
    assert qm.ai_extract_topics([(tmp_path, "text")], client=object()) == []


def test_collect_heading_topics_skips_invalid() -> None:
    text = "#\n### TooDeep\n# Valid\nDescription"
    topics = qm._collect_heading_topics([(Path("dummy.md"), text)])
    names = {t["name"] for t in topics}
    assert "Valid" in names and all("TooDeep" not in name for name in names)


def test_collect_heading_block_empty_name() -> None:
    pattern = __import__("re").compile(r"^#\s+(.*)$")
    assert qm._collect_heading_block(["# Intro"], "", pattern) == []
    assert qm._fallback_topic_lines(["Line"], "") == []


def test_gather_topic_context_skips_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path_file = tmp_path / "a.md"
    path_file.write_text("# Heading", encoding="utf-8")
    monkeypatch.setattr(qm, "_topic_source_paths", lambda topic: [path_file])
    monkeypatch.setattr(qm, "_read_topic_file", lambda path: "")
    assert qm._gather_topic_context({"name": "Heading"}) == ""


def test_generate_questions_non_mcq_and_no_per_topic() -> None:
    assert qm.generate_questions([], qtype="short") == []
    assert qm.generate_questions([], per_topic=0) == []


def test_generate_questions_placeholder_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [{"id": "intro", "name": "", "source_paths": []}]
    monkeypatch.setattr(qm, "ai_generate_mcqs_for_topic", lambda *a, **k: [])
    out = qm.generate_questions(topics, per_topic=1, ensure_coverage=True)
    assert out == []


def test_collect_heading_block_breaks_on_new_heading() -> None:
    lines = ["# Intro", "Details", "#Next", "Outro"]
    pattern = __import__("re").compile(r"^#\s+(.*)$")
    block = qm._collect_heading_block(lines, "Intro", pattern)
    assert block == ["# Intro", "Details"]


def test_gather_topic_context_no_snippet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_a = tmp_path / "a.md"
    file_a.write_text("# Other\nBody", encoding="utf-8")
    monkeypatch.setattr(qm, "_topic_source_paths", lambda topic: [file_a])
    assert qm._gather_topic_context({"name": "Missing"}) == ""


def test_select_questions_no_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qm, "_group_by_topic", lambda bank: {})
    assert qm.select_questions([{"id": "1", "topic_id": ""}], num=1) == []


def test_normalize_choice_list_skips_empty_text() -> None:
    assert qm._normalize_choice_list([{"key": "A", "text": ""}]) == []


def test_collect_heading_topics_skips_empty_name() -> None:
    text = "#   #\n# Valid\nDescription"
    topics = qm._collect_heading_topics([(Path("dummy.md"), text)])
    names = {t["name"] for t in topics}
    assert "Valid" in names and all(name.strip() for name in names)
