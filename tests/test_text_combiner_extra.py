from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from study_utils import text_combiner as tc


def test_parse_order_by_valid_and_invalid() -> None:
    assert tc.parse_order_by(None) is None
    assert tc.parse_order_by(" -Created ") == "-created"
    with pytest.raises(ValueError):
        tc.parse_order_by("size")


def test_parse_heading_valid_and_invalid() -> None:
    assert tc.parse_heading(None) is None
    assert tc.parse_heading(" ## ") == "##"
    with pytest.raises(ValueError):
        tc.parse_heading("heading")


def test_apply_title_format_and_heading() -> None:
    assert tc._apply_title_format("hello", None) == "hello"
    assert tc._apply_title_format("hello world", "title") == "Hello World"
    assert tc._apply_title_format("HELLO", "lower") == "hello"
    assert tc._apply_title_format("mixed", "unknown") == "mixed"
    assert tc._with_heading("Title", "##") == "## Title"


def test_make_section_title_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path("/tmp/Example File.txt")
    assert (
        tc.make_section_title("filename", path, None, None, "#")
        == "# Example File"
    )
    assert (
        tc.make_section_title("mystery", path, None, "upper", None)
        == "EXAMPLE FILE"
    )

    monkeypatch.setattr(tc, "_ai_title_from_filename", lambda *_: "AI Name")
    monkeypatch.setattr(tc, "_ai_title_from_content", lambda *_: "AI Content")
    assert (
        tc.make_section_title("smart-filename", path, None, "title", "##")
        == "## Ai Name"
    )
    assert (
        tc.make_section_title("smart-content", path, "Body", None, None)
        == "AI Content"
    )
    assert tc.make_section_title(None, path, None, None, None) is None


def test_ai_title_helpers_use_openai(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    custom_err = type("CustomErr", (Exception,), {})
    monkeypatch.setattr(tc, "OpenAIBadRequestError", custom_err, raising=False)

    stub = openai_factory()
    stub.queue_response("Generated Title\n")

    def _mock_load_client(
        local: bool = True,
        api_base: str | None = None,
        _stub=stub,
    ):
        return _stub

    monkeypatch.setattr(tc, "load_client", _mock_load_client)
    assert tc._ai_title_from_filename(Path("example.md")) == "Generated Title"

    stub2 = openai_factory()
    stub2.queue_response("Content Title")

    def _mock_load_client2(
        local: bool = True,
        api_base: str | None = None,
        _stub=stub2,
    ):
        return _stub

    monkeypatch.setattr(tc, "load_client", _mock_load_client2)
    assert tc._ai_title_from_content("Body", "file.md") == "Content Title"

    class BadClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    raise tc.OpenAIBadRequestError("boom")

            completions = Completions()

        chat = Chat()

    bad_client = BadClient()

    def _mock_load_client3(
        local: bool = True,
        api_base: str | None = None,
        _client=bad_client,
    ):
        return _client

    monkeypatch.setattr(tc, "load_client", _mock_load_client3)
    assert tc._ai_title_from_filename(Path("example.md")) is None
    assert tc._ai_title_from_content("Body", "file.md") is None

    class GenericClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("fail")

            completions = Completions()

        chat = Chat()

    def _mock_generic(
        local: bool = True,
        api_base: str | None = None,
    ):
        return GenericClient()

    monkeypatch.setattr(tc, "load_client", _mock_generic)
    assert tc._ai_title_from_filename(Path("example.md")) is None
    assert tc._ai_title_from_content("Body", "file.md") is None

    monkeypatch.setattr(
        tc, "load_client", lambda: (_ for _ in ()).throw(RuntimeError("load"))
    )
    assert tc._ai_title_from_filename(Path("example.md")) is None
    assert tc._ai_title_from_content("Body", "file.md") is None


def make_options(**overrides):
    base = tc.CombineOptions(
        extensions={"txt"},
        level_limit=0,
        combine_by="NEW",
        order_by=None,
        section_title=None,
        section_title_format=None,
        section_title_heading=None,
    )
    return replace(base, **overrides)


def test_combine_files_plain_and_with_titles(
    workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = workspace.write("a.txt", "Alpha")
    b = workspace.write("b.txt", "Beta")
    out = workspace.write("out.txt", "")

    count = tc.combine_files([a, b], out, make_options())
    assert count == 2
    assert out.read_text() == "Alpha\nBeta"

    monkeypatch.setattr(
        tc, "make_section_title", lambda *args, **kwargs: "Title"
    )
    out2 = workspace.write("out2.txt", "")
    opts = make_options(section_title="filename", combine_by="EOF")
    tc.combine_files([a, b], out2, opts)
    contents = out2.read_text()
    assert contents.count("Title") == 2


def test_main_success_and_error(
    workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    src_dir = workspace.root
    f1 = workspace.write("alpha.txt", "Alpha")
    f2 = workspace.write("beta.txt", "Beta")
    out_path = src_dir / "combined.txt"

    tc.main([str(out_path), str(f1), str(f2)])
    assert out_path.read_text() == "Alpha\nBeta"

    with pytest.raises(SystemExit) as exc:
        tc.main([str(out_path), str(f1), "--section-title-heading", "bad"])
    assert exc.value.code == 2
    assert "Error" in capsys.readouterr().out


def test_main_iter_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        tc,
        "iter_text_files",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(SystemExit) as exc:
        tc.main(["out.txt", "input.txt"])
    assert exc.value.code == 2
    assert "Error" in capsys.readouterr().out


def test_main_no_matching(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tc, "iter_text_files", lambda *a, **k: [])
    with pytest.raises(SystemExit) as exc:
        tc.main(["out.txt", "input.txt"])
    assert exc.value.code == 1
    assert "No matching files" in capsys.readouterr().out


def test_main_combine_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        tc, "iter_text_files", lambda *a, **k: [Path("file.txt")]
    )
    monkeypatch.setattr(tc, "order_files", lambda files, order: files)
    monkeypatch.setattr(
        tc,
        "combine_files",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    with pytest.raises(SystemExit) as exc:
        tc.main(["out.txt", "input.txt"])
    assert exc.value.code == 1
    assert "Failed" in capsys.readouterr().out
