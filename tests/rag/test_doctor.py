from __future__ import annotations

import json
from pathlib import Path

from study_utils.rag import doctor as doctor_mod
from study_utils.rag import config as config_mod
from study_utils.rag import vector_store


def test_generate_report_records_config_error(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(data_home))
    monkeypatch.setenv(
        config_mod.CONFIG_PATH_ENV,
        str(tmp_path / "override.toml"),
    )
    monkeypatch.setattr(doctor_mod, "_check_dependencies", lambda: tuple())
    monkeypatch.setattr(
        doctor_mod,
        "_determine_tokenizer",
        lambda cfg: doctor_mod.TokenizerStatus("tok", "enc", "ok", None),
    )

    report = doctor_mod.generate_report()

    assert report.data_home == data_home
    assert report.config_error is not None
    assert not report.config_exists
    assert report.directories

    formatted = doctor_mod.format_report(report)
    assert "Config path" in formatted
    assert "(none)" in formatted


def _base_report(**overrides):
    payload = {
        "data_home": Path("/tmp/home"),
        "data_home_severity": "ok",
        "data_home_message": None,
        "env_overrides": {},
        "directories": (
            doctor_mod.DirectoryStatus(
                name="config",
                path=Path("/tmp/config"),
                exists=True,
                is_dir=True,
                mode=0o700,
                severity="ok",
                message=None,
            ),
        ),
        "config_path": Path("/tmp/config/rag.toml"),
        "config_exists": True,
        "config_error": None,
        "dependencies": tuple(),
        "tokenizer": doctor_mod.TokenizerStatus("tok", "enc", "ok", None),
        "vector_stores": tuple(),
        "sessions": tuple(),
    }
    payload.update(overrides)
    return doctor_mod.DoctorReport(**payload)


def _make_config(*, tokenizer: str = "tok") -> config_mod.RagConfig:
    chunking = config_mod.ChunkingConfig(
        tokenizer=tokenizer,
        encoding="enc",
        tokens_per_chunk=10,
        token_overlap=1,
        fallback_delimiter="\n",
    )
    dedupe = config_mod.DedupConfig(
        strategy="checksum",
        checksum_algorithm="sha256",
    )
    ingestion = config_mod.IngestionConfig(
        chunking=chunking,
        dedupe=dedupe,
        max_workers=1,
        file_batch_size=1,
    )
    providers = config_mod.ProvidersConfig(
        default="openai",
        openai=config_mod.OpenAIConfig(
            chat_model="gpt",
            embedding_model="embed",
            max_input_tokens=100,
            max_output_tokens=100,
            temperature=0.2,
            api_base=None,
            request_timeout_seconds=30,
        ),
    )
    retrieval = config_mod.RetrievalConfig(
        top_k=1,
        max_context_tokens=10,
        score_threshold=0.5,
    )
    chat = config_mod.ChatConfig(
        max_history_turns=1,
        response_tokens=10,
        stream=True,
    )
    logging_cfg = config_mod.LoggingConfig(level="INFO", verbose=False)
    paths = config_mod.PathsConfig(data_home_override=None)
    return config_mod.RagConfig(
        paths=paths,
        providers=providers,
        ingestion=ingestion,
        retrieval=retrieval,
        chat=chat,
        logging=logging_cfg,
        services=config_mod._build_services(config_mod._DEFAULTS["services"]),
    )


def test_format_report_contains_sections():
    report = _base_report(
        env_overrides={"STUDY_UTILS_DATA_HOME": "/tmp/home"},
        vector_stores=(
            doctor_mod.StorageUsage(
                name="physics",
                path=Path("/tmp/home/rag_dbs/physics"),
                size_bytes=2048,
                issues=tuple(),
            ),
        ),
        dependencies=(
            doctor_mod.DependencyStatus(
                name="faiss",
                module="faiss",
                status="error",
                version=None,
                error="missing",
            ),
            doctor_mod.DependencyStatus(
                name="numpy",
                module="numpy",
                status="ok",
                version="1.0",
                error=None,
            ),
        ),
        sessions=(
            doctor_mod.StorageUsage(
                name="sess",
                path=Path("/tmp/home/rag_sessions/sess"),
                size_bytes=1024,
                issues=("missing session.json",),
            ),
        ),
    )

    output = doctor_mod.format_report(report)

    assert "Data home" in output
    assert "Dependencies" in output
    assert "Vector stores" in output


def test_has_errors_detects_dependency_issue():
    report = _base_report(
        dependencies=(
            doctor_mod.DependencyStatus(
                name="faiss",
                module="faiss",
                status="error",
                version=None,
                error="not installed",
            ),
        ),
    )

    assert doctor_mod.has_errors(report) is True


def test_has_errors_without_issues():
    report = _base_report()

    assert doctor_mod.has_errors(report) is False


def test_generate_report_with_assets(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    data_home.mkdir(parents=True)
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(data_home))

    config_dir = data_home / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "rag.toml"
    config_mod.write_template(config_path)

    vector_good = data_home / "rag_dbs" / "physics"
    vector_good.mkdir(parents=True)
    manifest = vector_store.build_manifest(
        name="physics",
        embedding=vector_store.EmbeddingMetadata(
            provider="openai",
            model="test",
            dimension=2,
        ),
        chunking=vector_store.ChunkingMetadata(
            tokenizer="tiktoken",
            encoding="cl100k_base",
            tokens_per_chunk=10,
            token_overlap=0,
            fallback_delimiter="\n\n",
        ),
        dedupe=vector_store.DedupMetadata(
            strategy="checksum",
            checksum_algorithm="sha256",
        ),
        documents=(
            vector_store.SourceDocument(
                source_path="doc.txt",
                checksum="abc",
                size_bytes=10,
                chunk_count=1,
            ),
        ),
    )
    (vector_good / "manifest.json").write_text(
        json.dumps(manifest.to_dict()),
        encoding="utf-8",
    )

    vector_bad = data_home / "rag_dbs" / "algebra"
    vector_bad.mkdir(parents=True)

    session_ok = data_home / "rag_sessions" / "sess1"
    session_ok.mkdir(parents=True)
    (session_ok / "session.json").write_text(
        json.dumps(
            {
                "session_id": "sess1",
                "created_at": "now",
                "updated_at": "now",
                "vector_dbs": [],
                "embedding": {
                    "provider": "openai",
                    "model": "test",
                    "dimension": 2,
                },
                "chat_model": "gpt",
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    session_bad = data_home / "rag_sessions" / "broken"
    session_bad.mkdir(parents=True)

    monkeypatch.setattr(
        doctor_mod,
        "_check_dependencies",
        lambda: (
            doctor_mod.DependencyStatus(
                name="faiss",
                module="faiss",
                status="error",
                version=None,
                error="missing",
            ),
            doctor_mod.DependencyStatus(
                name="numpy",
                module="numpy",
                status="ok",
                version="1.0",
                error=None,
            ),
        ),
    )
    monkeypatch.setattr(
        doctor_mod,
        "_determine_tokenizer",
        lambda cfg: doctor_mod.TokenizerStatus(
            "tok",
            "enc",
            "warning",
            "fallback",
        ),
    )

    report = doctor_mod.generate_report()

    assert report.config_error is None
    assert any(item.name == "physics" for item in report.vector_stores)
    assert any(item.issues for item in report.vector_stores)
    assert any(item.issues for item in report.sessions)

    formatted = doctor_mod.format_report(report)
    assert "environment overrides" in formatted
    assert "Vector stores" in formatted
    assert "Sessions" in formatted


def test_has_errors_detects_directory_issue():
    report = _base_report(
        directories=(
            doctor_mod.DirectoryStatus(
                name="config",
                path=Path("/tmp/config"),
                exists=True,
                is_dir=False,
                mode=None,
                severity="error",
                message="not a directory",
            ),
        ),
    )

    assert doctor_mod.has_errors(report) is True


def test_has_errors_detects_tokenizer_error():
    report = _base_report(
        tokenizer=doctor_mod.TokenizerStatus(
            "tok",
            "enc",
            status="error",
            message="boom",
        )
    )

    assert doctor_mod.has_errors(report) is True


def test_has_errors_detects_vector_issue():
    report = _base_report(
        vector_stores=(
            doctor_mod.StorageUsage(
                name="physics",
                path=Path("/tmp/home/rag_dbs/physics"),
                size_bytes=10,
                issues=("missing manifest",),
            ),
        ),
    )

    assert doctor_mod.has_errors(report) is True


def test_has_errors_detects_session_issue():
    report = _base_report(
        sessions=(
            doctor_mod.StorageUsage(
                name="sess",
                path=Path("/tmp/home/rag_sessions/sess"),
                size_bytes=10,
                issues=("missing session.json",),
            ),
        ),
    )

    assert doctor_mod.has_errors(report) is True


def test_has_errors_detects_data_home_error():
    report = _base_report(
        data_home=Path("/tmp/file"),
        data_home_severity="error",
        data_home_message="not a directory",
    )

    assert doctor_mod.has_errors(report) is True


def test_has_errors_detects_config_error():
    report = _base_report(config_error="bad config")

    assert doctor_mod.has_errors(report) is True


def test_resolve_data_home_not_directory(tmp_path):
    target = tmp_path / "notdir"
    target.write_text("", encoding="utf-8")
    env = {doctor_mod.data_dir.DATA_HOME_ENV: str(target)}

    path, severity, message = doctor_mod._resolve_data_home(env)

    assert path == target
    assert severity == "error"
    assert "not a directory" in (message or "")


def test_resolve_data_home_permission_warning(tmp_path):
    target = tmp_path / "home"
    target.mkdir(parents=True)
    target.chmod(0o755)
    env = {doctor_mod.data_dir.DATA_HOME_ENV: str(target)}

    path, severity, message = doctor_mod._resolve_data_home(env)

    assert path == target
    assert severity == "warning"
    assert "permissions" in message


def test_resolve_config_handles_path_error(monkeypatch, tmp_path):
    def fail_resolve(env):  # noqa: ANN001
        raise doctor_mod.config_mod.ConfigError("bad path")

    monkeypatch.setattr(
        doctor_mod.config_mod,
        "resolve_config_path",
        fail_resolve,
    )

    path, exists, error, cfg = doctor_mod._resolve_config({})

    assert not exists
    assert error == "bad path"
    assert cfg is None
    assert path.name == "rag.toml"


def test_resolve_config_handles_load_error(monkeypatch, tmp_path):
    config_path = tmp_path / "rag.toml"
    monkeypatch.setattr(
        doctor_mod.config_mod,
        "resolve_config_path",
        lambda env: config_path,
    )
    monkeypatch.setattr(
        doctor_mod.config_mod,
        "load_config",
        lambda env: (_ for _ in ()).throw(
            doctor_mod.config_mod.ConfigError("load boom")
        ),
    )

    path, exists, error, cfg = doctor_mod._resolve_config({})

    assert path == config_path
    assert not exists
    assert error == "load boom"
    assert cfg is None


def test_check_dependencies_handles_missing(monkeypatch):
    def fake_import(name):  # noqa: D401, ANN001
        if name == "faiss":
            raise ImportError("no module")
        return object()

    def fake_version(package):  # noqa: D401, ANN001
        if package == "tiktoken":
            raise doctor_mod.metadata.PackageNotFoundError  # type: ignore[attr-defined]
        return "1.0"

    monkeypatch.setattr(doctor_mod.importlib, "import_module", fake_import)
    monkeypatch.setattr(doctor_mod.metadata, "version", fake_version)

    results = doctor_mod._check_dependencies()

    statuses = {item.name: item for item in results}
    assert statuses["faiss"].status == "error"
    assert statuses["numpy"].status == "ok"


def test_determine_tokenizer_config_none():
    status = doctor_mod._determine_tokenizer(None)

    assert status.status == "warning"
    assert status.message == "config unavailable"


def test_determine_tokenizer_error(monkeypatch):
    monkeypatch.setattr(
        doctor_mod.ingest_mod,
        "TextChunker",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    cfg = _make_config()
    status = doctor_mod._determine_tokenizer(cfg)

    assert status.status == "error"
    assert status.message == "boom"


def test_determine_tokenizer_fallback(monkeypatch):
    class DummyChunker:
        def __init__(self, **kwargs):  # noqa: D401, ANN001
            self._encoder = None

    monkeypatch.setattr(doctor_mod.ingest_mod, "TextChunker", DummyChunker)

    cfg = _make_config(tokenizer="tiktoken")
    status = doctor_mod._determine_tokenizer(cfg)

    assert status.status == "warning"
    assert "fallback" in status.message


def test_determine_tokenizer_ok(monkeypatch):
    class DummyChunker:
        def __init__(self, **kwargs):  # noqa: D401, ANN001
            self._encoder = [1]

    monkeypatch.setattr(doctor_mod.ingest_mod, "TextChunker", DummyChunker)

    cfg = _make_config(tokenizer="custom")
    status = doctor_mod._determine_tokenizer(cfg)

    assert status.status == "ok"
    assert status.message is None


def test_collect_vector_stores_missing_root(tmp_path):
    root = tmp_path / "nope"
    assert doctor_mod._collect_vector_stores(root) == tuple()


def test_collect_sessions_missing_root(tmp_path):
    root = tmp_path / "nope"
    assert doctor_mod._collect_sessions(root) == tuple()


def test_directory_status_missing(tmp_path):
    status = doctor_mod._directory_status("logs", tmp_path / "missing")
    assert status.message == "missing"
    assert status.severity == "warning"


def test_directory_status_not_dir(tmp_path):
    file = tmp_path / "file.txt"
    file.write_text("", encoding="utf-8")
    status = doctor_mod._directory_status("config", file)
    assert status.severity == "error"
    assert status.message == "not a directory"


def test_directory_status_permission_warning(tmp_path):
    target = tmp_path / "dir"
    target.mkdir()
    target.chmod(0o755)
    status = doctor_mod._directory_status("logs", target)
    assert status.severity == "warning"
    assert "permissions" in (status.message or "")


def test_collect_vector_stores_skips_files(tmp_path):
    root = tmp_path / "vector"
    root.mkdir()
    (root / "file.txt").write_text("", encoding="utf-8")
    assert doctor_mod._collect_vector_stores(root) == tuple()


def test_collect_sessions_skips_files(tmp_path):
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "file.txt").write_text("", encoding="utf-8")
    assert doctor_mod._collect_sessions(root) == tuple()


def test_safe_mode_handles_os_error(tmp_path):
    missing = tmp_path / "missing"
    assert doctor_mod._safe_mode(missing) is None


def test_format_bytes_variants():
    assert doctor_mod._format_bytes(512) == "512 B"
    assert doctor_mod._format_bytes(2048).endswith("KB")
    assert doctor_mod._format_bytes(10 * 1024 * 1024).endswith("MB")
    assert doctor_mod._format_bytes(5 * 1024**4).endswith("TB")
    assert doctor_mod._format_bytes(20 * 1024**4).startswith("20")


def test_resolve_data_home_mock_not_dir(monkeypatch, tmp_path):
    target = tmp_path / "notdir"
    target.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        doctor_mod.data_dir,
        "get_data_home",
        lambda env, create: target,
    )

    path, severity, message = doctor_mod._resolve_data_home({})

    assert path == target
    assert severity == "error"
    assert message == "path is not a directory"


def test_resolve_data_home_ok(monkeypatch, tmp_path):
    target = tmp_path / "home"
    target.mkdir()
    target.chmod(0o700)

    monkeypatch.setattr(
        doctor_mod.data_dir,
        "get_data_home",
        lambda env, create: target,
    )

    path, severity, message = doctor_mod._resolve_data_home({})

    assert path == target
    assert severity == "ok"
    assert message is None
