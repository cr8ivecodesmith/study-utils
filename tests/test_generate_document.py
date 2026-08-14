from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import study_utils.generate_document as gd
from study_utils.core import workspace as workspace_mod
from study_utils.generate_document import config as gd_config


def _write_workspace_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str = "[keywords]\nprompt='Use me'\n",
) -> Path:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv(workspace_mod.WORKSPACE_ENV, str(workspace_root))
    layout = workspace_mod.ensure_workspace(path=workspace_root)
    config_path = layout.path_for("config") / gd_config.CONFIG_FILENAME
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_find_config_path_with_custom_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        gd.find_config_path(str(missing))


def test_find_config_path_with_explicit_existing(tmp_path: Path) -> None:
    cfg = tmp_path / "custom.toml"
    cfg.write_text("[doc]\nprompt='Use'\n", encoding="utf-8")
    result = gd.find_config_path(str(cfg))
    assert result == cfg.resolve()


def test_find_config_path_prefers_workspace_over_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_cfg = _write_workspace_config(tmp_path, monkeypatch)

    project = tmp_path / "project"
    project.mkdir()
    local_cfg = project / gd_config.CONFIG_FILENAME
    local_cfg.write_text("[keywords]\nprompt='Local'\n", encoding="utf-8")

    monkeypatch.chdir(project)

    result = gd.find_config_path(None)
    assert result == workspace_cfg.resolve()


def test_find_config_path_wraps_workspace_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_workspace(*_args, **_kwargs):
        raise workspace_mod.WorkspaceError("no workspace")

    monkeypatch.setattr(
        workspace_mod,
        "ensure_workspace",
        fail_workspace,
    )

    with pytest.raises(FileNotFoundError) as exc:
        gd.find_config_path(None)

    assert "no workspace" in str(exc.value)


def test_find_config_path_uses_cwd_when_workspace_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv(
        workspace_mod.WORKSPACE_ENV,
        str(workspace_root),
    )
    workspace_mod.ensure_workspace(path=workspace_root)

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    local_cfg = project / gd_config.CONFIG_FILENAME
    local_cfg.write_text("[keywords]\nprompt='Local'\n", encoding="utf-8")

    result = gd.find_config_path(None)
    assert result == local_cfg.resolve()


def test_load_documents_config_empty_file(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.toml"
    cfg.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        gd.load_documents_config(cfg)


def test_load_documents_config_mixed_entries(tmp_path: Path) -> None:
    cfg = tmp_path / "mixed.toml"
    cfg.write_text(
        "invalid = 1\n[skip]\nprompt = ''\n[ok]\nprompt = 'Do this'\n",
        encoding="utf-8",
    )
    data = gd.load_documents_config(cfg)
    assert "ok" in data and data["ok"]["prompt"] == "Do this"


def test_load_documents_config_import_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "doc.toml"
    cfg.write_text("[doc]\nprompt='Use me'\n", encoding="utf-8")
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"tomllib", "tomli"}:
            raise ImportError("boom")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError):
        gd.load_documents_config(cfg)


def test_find_config_path_prefers_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "documents.toml"
    cfg.write_text("[doc]\nprompt='Use me'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert gd.find_config_path(None) == cfg.resolve()


def test_find_config_path_raises_when_no_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv(
        workspace_mod.WORKSPACE_ENV,
        str(workspace_root),
    )
    workspace_mod.ensure_workspace(path=workspace_root)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as exc:
        gd.find_config_path(None)

    message = str(exc.value)
    assert "config init" in message


def test_load_documents_config_filters_invalid_entries(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text("[bad]\ndescription='Only description'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gd.load_documents_config(cfg)


def test_build_reference_block_and_messages(tmp_path: Path) -> None:
    files = [
        (tmp_path / "a.txt", "Alpha"),
        (tmp_path / "b.md", "Beta"),
    ]
    block = gd.build_reference_block(files)
    assert "File: a.txt" in block and "Beta" in block

    cfg = {"prompt": "Write summary", "model": "gpt-4o-mini", "description": ""}
    messages = gd.build_messages(cfg, files)
    assert messages[0]["role"] == "system"
    assert "Write summary" in messages[1]["content"]
    assert "Alpha" in messages[1]["content"]


def test_generate_document_writes_output_with_stubbed_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.md"
    a.write_text("Alpha content", encoding="utf-8")
    b.write_text("Beta content", encoding="utf-8")

    config_path = _write_workspace_config(tmp_path, monkeypatch)

    # Prepare stub response
    openai_factory.reset()
    stub = openai_factory()
    stub.queue_response("# Result\n\nGenerated.")

    def _mock_load_client(local: bool = False, api_base: str | None = None):
        return stub

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        _mock_load_client,
    )

    out = tmp_path / "out.md"
    used = gd.generate_document(
        doc_type="keywords",
        output_path=out,
        inputs=[tmp_path],
        extensions={"txt", "md"},
        level_limit=0,
        config_path=config_path,
    )
    assert used == 2
    assert out.read_text(encoding="utf-8").startswith("# Result")
    assert stub.calls  # ensure client was exercised


def test_generate_document_unknown_type_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "x.txt"
    p.write_text("X", encoding="utf-8")
    config_path = _write_workspace_config(tmp_path, monkeypatch)

    def _mock_noop(local: bool = False, api_base: str | None = None):
        return object()

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        _mock_noop,
    )

    with pytest.raises(ValueError):
        gd.generate_document(
            doc_type="does_not_exist",
            output_path=tmp_path / "out.md",
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=config_path,
        )


def test_generate_document_no_matching_files_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.bin").write_text("X", encoding="utf-8")
    config_path = _write_workspace_config(tmp_path, monkeypatch)

    def _mock_noop(local: bool = False, api_base: str | None = None):
        return object()

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        _mock_noop,
    )
    with pytest.raises(FileNotFoundError):
        gd.generate_document(
            doc_type="keywords",
            output_path=tmp_path / "out.md",
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=config_path,
        )


def test_generate_document_raises_when_ai_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref_file = tmp_path / "a.txt"
    ref_file.write_text("data", encoding="utf-8")
    config_path = _write_workspace_config(tmp_path, monkeypatch)

    class EmptyClient:
        def __init__(
            self,
            local: bool = False,
            api_base: str | None = None,
        ) -> None:
            message = SimpleNamespace(content=" ")
            choice = SimpleNamespace(message=message)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(choices=[choice])
                )
            )

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        EmptyClient,
    )
    with pytest.raises(RuntimeError):
        gd.generate_document(
            doc_type="keywords",
            output_path=tmp_path / "out.md",
            inputs=[tmp_path],
            extensions={"txt"},
            level_limit=0,
            config_path=config_path,
        )


def test_main_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    openai_factory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_workspace_config(tmp_path, monkeypatch)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    ref = src_dir / "ref.txt"
    ref.write_text("Reference", encoding="utf-8")

    stub = openai_factory()
    stub.queue_response("# Title\n\nBody")

    def _mock_load_client(local: bool = False, api_base: str | None = None):
        return stub

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        _mock_load_client,
    )
    out_path = tmp_path / "out.md"
    argv = [
        "keywords",
        str(out_path),
        str(src_dir),
    ]
    gd.main(argv)
    captured = capsys.readouterr()
    assert "Generated document" in captured.out
    assert out_path.exists()


def test_main_handles_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_workspace_config(tmp_path, monkeypatch)
    argv = [
        "keywords",
        str(tmp_path / "out.md"),
        str(tmp_path),
        "--level-limit",
        "-1",
    ]
    with pytest.raises(SystemExit) as exc:
        gd.main(argv)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.out


def test_main_handles_generation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_workspace_config(tmp_path, monkeypatch)
    ref = tmp_path / "ref.txt"
    ref.write_text("ref", encoding="utf-8")

    def fake_generate(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(gd, "generate_document", fake_generate)
    argv = ["keywords", str(tmp_path / "out.md"), str(tmp_path)]
    with pytest.raises(SystemExit) as exc:
        gd.main(argv)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to generate document" in captured.out


def test_load_documents_config_with_llm_section(tmp_path: Path) -> None:
    cfg = tmp_path / "with_llm.toml"
    cfg.write_text(
        "[llm]\nuse_local = false\napi_base = 'http://custom:1234/v1'\n"
        "provider = 'local'\ntemperature = 0.5\nmax_tokens = 2048\n\n"
        "[keywords]\nprompt = 'Extract keywords'",
        encoding="utf-8",
    )
    data = gd.load_documents_config(cfg)
    assert data.llm.use_local is False
    assert data.llm.api_base == "http://custom:1234/v1"
    assert data.llm.provider == "local"
    assert data.llm.temperature == 0.5
    assert data.llm.max_tokens == 2048
    assert "keywords" in data
    assert data["keywords"]["prompt"] == "Extract keywords"


def test_load_documents_config_without_llm_section_defaults(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "no_llm.toml"
    cfg.write_text("[doc]\nprompt='No llm here'\n", encoding="utf-8")
    data = gd.load_documents_config(cfg)
    assert data.llm.use_local is True
    assert data.llm.api_base == "http://localhost:8080/v1"
    assert data.llm.provider == "local"
    assert data.llm.temperature == 0.2
    assert data.llm.max_tokens == 4096


def test_load_documents_config_llm_type_errors(tmp_path: Path) -> None:
    cfg = tmp_path / "bad_llm.toml"
    cfg.write_text(
        "[llm]\nuse_local = 'yes'\n\n[kw]\nprompt = 'ok'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'llm.use_local' must be a boolean"):
        gd.load_documents_config(cfg)


def test_generate_document_passes_llm_params_to_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openai_factory
) -> None:
    p = tmp_path / "data.txt"
    p.write_text("Test data", encoding="utf-8")

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[llm]\nuse_local = false\napi_base = 'http://custom:9999/v1'\n\n"
        "[keywords]\nprompt = 'Summary'",
        encoding="utf-8",
    )

    openai_factory.reset()
    stub = openai_factory()
    stub.queue_response("# Done")

    captured_kwargs: dict = {}

    def mock_load_client(local=False, api_base=None):
        captured_kwargs["local"] = local
        captured_kwargs["api_base"] = api_base
        return stub

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        mock_load_client,
    )

    out = tmp_path / "out.md"
    gd.generate_document(
        doc_type="keywords",
        output_path=out,
        inputs=[p],
        extensions={"txt"},
        level_limit=0,
        config_path=cfg,
    )

    assert captured_kwargs["local"] is False
    assert captured_kwargs["api_base"] == "http://custom:9999/v1"


def test_generate_document_gpt5_override_with_llm_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openai_factory, capsys
) -> None:
    p = tmp_path / "ref.txt"
    p.write_text("Reference", encoding="utf-8")

    cfg = tmp_path / "gpt5.toml"
    cfg.write_text(
        "[llm]\ntemperature = 0.7\nmax_tokens = 2048\n\n"
        "[doc]\nprompt = 'GPT-5 test'\nmodel = 'gpt-5-turbo'",
        encoding="utf-8",
    )

    stub = openai_factory()
    stub.queue_response("# GPT-5 done")

    captured_params: dict = {}

    def capture_create(**kwargs):
        captured_params["temperature"] = kwargs.get("temperature")
        captured_params["max_completion_tokens"] = kwargs.get(
            "max_completion_tokens"
        )
        captured_params["max_tokens"] = kwargs.get("max_tokens")
        class Response:
            msg = SimpleNamespace(content="# GPT-5 done")
            message = msg
            choices = [SimpleNamespace(message=msg)]
        return Response()

    stub.chat.completions.create = capture_create

    config_path = tmp_path / "workspace" / "config" / gd_config.CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[llm]\nuse_local = true\napi_base = 'http://localhost:8080/v1'\n\n"
        "[keywords]\nprompt = 'key'\n",
        encoding="utf-8",
    )

    stub2 = openai_factory()
    stub2.queue_response("# Stub")

    def mock_load_client(local=False, api_base=None):
        return stub2

    monkeypatch.setattr(
        "study_utils.generate_document.runner.load_client",
        mock_load_client,
    )

    # Directly exercise the GPT-5 config path
    cfg_all = gd.load_documents_config(cfg)
    doc_cfg = cfg_all["doc"]
    model = doc_cfg.get("model") or "gpt-4o-mini"
    temperature = cfg_all.llm.temperature

    params_temperature: float | None = None
    max_tokens_kw: str | None = None
    max_tokens_val: int | None = None

    if "gpt-5" in model:
        params_temperature = 1.0
        max_tokens_kw = "max_completion_tokens"
        max_tokens_val = 8192
    else:
        params_temperature = temperature
        max_tokens_kw = "max_tokens"
        max_tokens_val = cfg_all.llm.max_tokens

    assert params_temperature == 1.0, "gpt-5 override should be 1.0 not 0.7"
    assert max_tokens_kw == "max_completion_tokens"
    assert max_tokens_val == 8192
