from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from study_utils import transcribe_video as tv


def test_find_video_files_variants(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    assert tv.find_video_files(video) == [video]

    bad = tmp_path / "sample.txt"
    bad.write_bytes(b"")
    with pytest.raises(ValueError):
        tv.find_video_files(bad)

    with pytest.raises(FileNotFoundError):
        tv.find_video_files(tmp_path / "missing.mp4")

    files = tv.find_video_files(tmp_path, recursive=False)
    assert files == [video]

    nested = tmp_path / "nest"
    nested.mkdir()
    deep = nested / "deep.mp4"
    deep.write_bytes(b"")
    recursive_files = tv.find_video_files(tmp_path, recursive=True)
    assert set(recursive_files) == {video, deep}


def test_find_video_files_invalid_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    special = tmp_path / "special"
    special.touch()
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir

    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True if self == special else original_exists(self),
    )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: False if self == special else original_is_file(self),
    )
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda self: False if self == special else original_is_dir(self),
    )

    with pytest.raises(ValueError):
        tv.find_video_files(special)

    # restore for safety (monkeypatch will undo on teardown)


def test_default_names_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dir_cache = tv.default_names_cache_path(tmp_path)
    assert dir_cache.name == ".transcribe_video_names.json"

    file_cache = tv.default_names_cache_path(tmp_path / "video.mp4")
    assert file_cache == dir_cache

    monkeypatch.setattr(Path, "exists", lambda self: False)
    path = tv.default_names_cache_path(Path("/does/not/exist.mp4"))
    assert path.parent == Path(tv.gettempdir())


def test_load_and_save_names_cache(tmp_path: Path) -> None:
    cache = tmp_path / "names.json"
    entries = {tmp_path / "a.mp4": {"base": "A", "final": "A.txt"}}
    tv.save_names_cache(cache, tmp_path, entries, meta={"foo": "bar"})
    data = tv.load_names_cache(cache)
    assert data[str(tmp_path / "a.mp4")]["base"] == "A"

    tv.save_names_cache(cache, tmp_path, {tmp_path / "b.mp4": "B"})
    data2 = tv.load_names_cache(cache)
    assert data2[str(tmp_path / "b.mp4")]["base"] == "B"

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("invalid", encoding="utf-8")
    assert tv.load_names_cache(corrupt) == {}

    not_dict = tmp_path / "not_dict.json"
    not_dict.write_text(json.dumps({"names": [1, 2]}), encoding="utf-8")
    assert tv.load_names_cache(not_dict) == {}


def test_cache_helpers() -> None:
    assert tv.cache_get_base({"base": "X"}, "fallback") == "X"
    assert tv.cache_get_base("Y", "fallback") == "Y"
    assert tv.cache_get_base({}, "fallback") == "fallback"

    assert tv.cache_get_final({"final": "Z"}) == "Z"
    assert tv.cache_get_final("none") is None


def test_clean_segment_and_heuristic_name(tmp_path: Path) -> None:
    video = tmp_path / "module_01" / "video.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    name = tv.heuristic_smart_name(video, tmp_path)
    assert "module" in name.lower()

    outside = Path("/different/root/video.mp4")
    assert tv.heuristic_smart_name(outside, tmp_path) == tv._clean_segment(
        outside.stem
    )


def test_ai_smart_name_success(
    monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    stub = openai_factory()
    stub.queue_response("Title: Intro/Lesson?")
    result = tv.ai_smart_name(stub, Path("video.mp4"), Path("."))
    assert "Intro" in result

    class BoomClient:
        def chat(self):  # pragma: no cover
            raise RuntimeError

    monkeypatch.setattr(tv, "OpenAI", lambda *a, **k: BoomClient())

    failing = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: (_ for _ in ()).throw(RuntimeError("fail"))
            )
        )
    )
    assert tv.ai_smart_name(failing, Path("video.mp4"), Path(".")) is None


def test_build_name_mapping_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    files = [Path("a.mp4"), Path("b.mp4")]
    monkeypatch.setattr(tv, "heuristic_smart_name", lambda p, root: "Name")
    monkeypatch.setattr(tv, "ai_smart_name", lambda *a, **k: "AI")
    mapping = tv.build_name_mapping(files, Path("."), True, object())
    assert len(set(mapping.values())) == 2

    monkeypatch.setattr(tv, "heuristic_smart_name", lambda p, root: "")
    mapping2 = tv.build_name_mapping([Path("only.mp4")], Path("."), False, None)
    assert mapping2[Path("only.mp4")] == "only"


class _FakeChunk:
    def __init__(self, idx: int, records: list[str]):
        self.idx = idx
        self._records = records

    def export(self, dest: Path, format: str) -> None:
        self._records.append(dest.name)
        dest.write_text(f"chunk-{self.idx}", encoding="utf-8")


def test_split_video_to_audio_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[str] = []

    class FakeAudio:
        pass

    def fake_from_file(path, format):
        return FakeAudio()

    def fake_make_chunks(_audio, _size):
        return [_FakeChunk(0, records), _FakeChunk(1, records)]

    monkeypatch.setattr(
        tv, "AudioSegment", SimpleNamespace(from_file=fake_from_file)
    )
    monkeypatch.setattr(tv, "make_chunks", fake_make_chunks)
    monkeypatch.setattr(tv, "rmtree", lambda path: records.append(f"rm:{path}"))

    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    segments = tv.split_video_to_audio_segments(video, exist_delete=False)
    assert len(segments) == 2
    assert all(seg.exists() for seg in segments)

    # cover exist_delete=True with existing directory
    for seg in segments:
        seg.unlink()
    seg_dir = tmp_path / "video_segments"
    seg_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(tv, "rmtree", lambda path: records.append(f"rm:{path}"))
    records.clear()
    segments2 = tv.split_video_to_audio_segments(video, exist_delete=True)
    assert records[0].startswith("rm:")
    for seg in segments2:
        seg.unlink()
    seg_dir.rmdir()


def test_transcribe_audio_file(tmp_path: Path) -> None:
    audio = tmp_path / "chunk.mp3"
    audio.write_bytes(b"data")

    class Client:
        class Audio:
            class Transcriptions:
                @staticmethod
                def create(**kwargs):
                    return " transcript "

            transcriptions = Transcriptions()

        audio = Audio()

    result = tv.transcribe_audio_file(Client(), audio)
    assert result == "transcript"


def test_transcribe_video_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tv,
        "split_video_to_audio_segments",
        lambda path, **kwargs: [Path(f"seg_{i}.mp3") for i in range(2)],
    )
    monkeypatch.setattr(
        tv, "transcribe_audio_file", lambda client, seg: f"text-{seg}"
    )
    monkeypatch.setattr(tv, "sleep", lambda _: None)
    cleanup = []
    monkeypatch.setattr(tv, "rmtree", lambda path: cleanup.append(path))
    result = tv.transcribe_video_file(object(), Path("video.mp4"))
    assert "text" in result
    assert cleanup


def test_sanitize_and_strip() -> None:
    assert tv.sanitize_filename("Video:Intro") == "Video-Intro"
    assert tv._strip_outer_quotes('"Hello"') == "Hello"
    assert tv._strip_outer_quotes("NoQuotes") == "NoQuotes"


def test_parse_prefix_parts_and_build_prefix() -> None:
    parts = tv.parse_prefix_parts(["text:Intro", "counter:NN", "sep:-", 123])
    assert parts[0] == ("text", "Intro")
    assert tv.build_prefix_string(parts, 3) == "Intro03-"

    assert tv.parse_prefix_parts(["bad"]) == [("text", "bad")]


def test_make_output_filename() -> None:
    base = tv.make_output_filename(
        Path("video.mp4"), 2, [("text", "Intro-")], "Smart"
    )
    assert base.startswith("Intro-")


def test_discover_video_files_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tv,
        "find_video_files",
        lambda *_: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(SystemExit):
        tv._discover_video_files(Path("/tmp"), False)


def test_handle_list_mode_plain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = SimpleNamespace(prefix=None, smart_names=False)
    file_paths = [tmp_path / "a.mp4"]
    for p in file_paths:
        p.write_bytes(b"")
    tv._handle_list_mode(args, file_paths, tmp_path)
    out = capsys.readouterr().out
    assert "->" in out


def test_handle_list_mode_empty(capsys: pytest.CaptureFixture[str]) -> None:
    args = SimpleNamespace(prefix=None, smart_names=False)
    tv._handle_list_mode(args, [], Path("."))
    assert "No .mp4 files" in capsys.readouterr().out


def test_handle_list_mode_smart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        prefix=None,
        smart_names=True,
        use_ai=False,
        names_file=None,
        refresh_names=False,
        list_only=True,
    )
    files = [tmp_path / "a.mp4"]
    for p in files:
        p.write_bytes(b"")

    monkeypatch.setattr(tv, "load_client", lambda: None)
    monkeypatch.setattr(tv, "save_names_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        tv,
        "build_name_mapping",
        lambda files, root, use_ai, client, title_model="gpt-4o-mini": {
            files[0]: "Name"
        },
    )
    monkeypatch.setattr(tv, "cache_get_final", lambda entry: None)

    tv._handle_list_mode(args, files, tmp_path)
    assert "Proposed names" in capsys.readouterr().out


def test_prepare_output_dir(tmp_path: Path) -> None:
    out = tv._prepare_output_dir(str(tmp_path / "out"))
    assert out.exists()


def test_prepare_names_for_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    args = SimpleNamespace(
        smart_names=True,
        use_ai=False,
        names_file=None,
        refresh_names=False,
        prefix=None,
    )
    monkeypatch.setattr(
        tv,
        "_resolve_names_paths",
        lambda *a: (tmp_path, tmp_path / "names.json"),
    )
    monkeypatch.setattr(tv, "_load_existing_names", lambda *_: {})
    monkeypatch.setattr(tv, "_build_mapping_base", lambda *a: {video: "Base"})
    monkeypatch.setattr(
        tv,
        "_combine_name_entries",
        lambda *a: {video: {"base": "Base", "final": "Base.txt"}},
    )
    monkeypatch.setattr(tv, "save_names_cache", lambda *a, **k: None)
    entries = tv._prepare_names_for_run(args, [video], tmp_path, None, None)
    assert entries[video]["final"].endswith(".txt")

    args.smart_names = False
    assert tv._prepare_names_for_run(args, [video], tmp_path, None, None) == {}


def test_transcribe_videos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(
        tv,
        "transcribe_video_file",
        lambda client, path, **kwargs: "text",
    )
    monkeypatch.setattr(tv, "cache_get_final", lambda entry: entry.get("final"))

    entries = {video: {"base": "Base", "final": "Base.txt"}}
    tv._transcribe_videos([video], object(), [], entries, out_dir, True)
    out = capsys.readouterr().out
    assert "Saving transcript" in out
    assert (out_dir / "Base.txt").exists()

    tv._transcribe_videos([video], object(), [], entries, out_dir, False)
    assert (out_dir / f"{video.stem}.txt").exists()

    monkeypatch.setattr(
        tv,
        "transcribe_video_file",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    tv._transcribe_videos([video], object(), [], entries, out_dir, False)


def test_resolve_names_paths(tmp_path: Path) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"")
    root, cache = tv._resolve_names_paths(
        SimpleNamespace(names_file=None), target
    )
    assert cache.name.endswith(".json")

    custom = tmp_path / "cache.json"
    root2, cache2 = tv._resolve_names_paths(
        SimpleNamespace(names_file=str(custom)), target
    )
    assert cache2 == custom


def test_load_existing_names(tmp_path: Path) -> None:
    cache = tmp_path / "names.json"
    entries = {tmp_path / "a.mp4": {"base": "A", "final": "A.txt"}}
    tv.save_names_cache(cache, tmp_path, entries)
    data = tv._load_existing_names(cache)
    assert list(data.keys())[0] == tmp_path / "a.mp4"


def test_build_mapping_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = [tmp_path / "a.mp4"]
    for f in files:
        f.write_bytes(b"")
    args = SimpleNamespace(smart_names=True, use_ai=True, refresh_names=True)
    monkeypatch.setattr(
        tv, "build_name_mapping", lambda *a, **k: {files[0]: "Name"}
    )
    mapping = tv._build_mapping_base(args, files, tmp_path, object(), {})
    assert mapping

    args.refresh_names = False
    monkeypatch.setattr(
        tv,
        "build_name_mapping",
        lambda files, root, use_ai, client, title_model="gpt-4o-mini": {
            files[0]: "Missing"
        },
    )
    mapping2 = tv._build_mapping_base(args, files, tmp_path, object(), {})
    assert mapping2

    mapping3 = tv._build_mapping_base(
        args, files, tmp_path, object(), {files[0]: {}}
    )
    assert mapping3 == {}

    args.smart_names = False
    assert tv._build_mapping_base(args, files, tmp_path, object(), {}) == {}


def test_combine_name_entries(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    result = tv._combine_name_entries(
        [video],
        {},
        {video: "Base"},
        [("text", "Intro-")],
        tmp_path,
    )
    entry = result[video]
    assert entry["base"] == "Base"
    assert entry["final"].endswith(".txt")

    # fallback to heuristic when mapping missing and no existing entry
    video2 = tmp_path / "b.mp4"
    video2.write_bytes(b"")
    combined = tv._combine_name_entries([video2], {}, {}, [], tmp_path)
    assert "b" in combined[video2]["base"].lower()


def test_main_list_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"")
    args = SimpleNamespace(
        TARGET=str(target),
        output_dir=None,
        prefix=None,
        list_only=True,
        recursive=False,
        smart_names=False,
        use_ai=False,
        names_file=None,
        refresh_names=False,
    )
    monkeypatch.setattr(tv, "_parse_transcribe_args", lambda: args)
    tv.main()
    assert "->" in capsys.readouterr().out


def test_main_no_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        TARGET=str(tmp_path),
        output_dir=None,
        prefix=None,
        list_only=False,
        recursive=False,
        smart_names=False,
        use_ai=False,
        names_file=None,
        refresh_names=False,
    )
    monkeypatch.setattr(tv, "_parse_transcribe_args", lambda: args)
    monkeypatch.setattr(tv, "_discover_video_files", lambda *a, **k: [])
    with pytest.raises(SystemExit) as exc:
        tv.main()
    assert exc.value.code == 1
    assert "No .mp4 files" in capsys.readouterr().out


def test_parse_transcribe_args(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    argv = [
        "transcribe-video",
        "target.mp4",
        "--list",
        "--prefix",
        "text:Intro",
        "--smart-names",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = tv._parse_transcribe_args()
    assert args.list_only and args.smart_names and args.prefix == ["text:Intro"]


def test_main_transcribe_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        TARGET=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        prefix=None,
        list_only=False,
        recursive=False,
        smart_names=False,
        use_ai=False,
        names_file=None,
        refresh_names=False,
    )

    video = tmp_path / "vid.mp4"
    video.write_bytes(b"")

    monkeypatch.setattr(tv, "_parse_transcribe_args", lambda: args)
    monkeypatch.setattr(tv, "_discover_video_files", lambda *a, **k: [video])
    monkeypatch.setattr(tv, "_prepare_output_dir", lambda out: tmp_path)

    def _mock_load_client(local: bool = True, api_base: str | None = None):
        return object()

    monkeypatch.setattr(tv, "load_client", _mock_load_client)
    monkeypatch.setattr(tv, "parse_prefix_parts", lambda parts: [])
    monkeypatch.setattr(tv, "_prepare_names_for_run", lambda *a, **k: {})
    monkeypatch.setattr(tv, "_transcribe_videos", lambda *a, **k: None)
    tv.main()
    assert "Done" in capsys.readouterr().out
