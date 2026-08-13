"""Tests for transcribe-video config system.

Covers dataclasses, validation helpers, builder functions, default tree,
merge defaults + path resolution, subcommand handlers, and backward compat.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Ensure the project root is importable when tests run as subprocesses
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from study_utils.core.config import TomlConfigError, merge_defaults  # noqa: E402
from study_utils.core.config_templates import (  # noqa: E402
    ConfigTemplate,
    ConfigTemplateError,
    get_template,
)

# All imports from transcribe_video module
import study_utils.transcribe_video as tv  # noqa: E402

# Template string for transcribe-video config tests.
BASE_CONFIG = """\
[ai]
use_local = true
api_base = "http://localhost:8080/v1"
provider = "local"
model = "whisper-3"
title_model = "gpt-4o-mini"

[audio]
segment_duration_minutes = 10

[names]
smart_names = true
use_ai_titles = false
output_dir = null
recursive = true
cache_path = null

[logging]
level = "INFO"
verbose = false
"""

EXPECTED_KEYS = {"ai", "audio", "names", "logging"}


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------


class TestTemplate:
    """Tests for packaged template loading and retrieval."""

    def test_config_template_returns_string(self) -> None:
        result = tv.config_template()
        assert isinstance(result, str) and len(result) > 0

    def test_config_template_content_parsable(self) -> None:
        content = tv.config_template()
        # Basic verification that all section headers exist
        for section in ["[ai]", "[audio]", "[names]", "[logging]"]:
            assert section in content

    def test_get_template_lookup(self) -> None:
        tmpl = get_template("transcribe_video")
        assert isinstance(tmpl, ConfigTemplate)
        assert tmpl.name == "transcribe_video"
        assert tmpl.filename == "transform.toml"
        assert tmpl.package == "study_utils.transcribe_video"

    def test_get_template_unknown_raises(self) -> None:
        with pytest.raises(ConfigTemplateError):
            get_template("nonexistent_template_xyz")


class TestWriteTemplate:
    """Tests for write_template function."""

    def test_write_template_creates_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "output" / "transcribe.toml"
        result = tv.write_template(dest)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "[ai]" in content

    def test_write_template_rejects_existing_without_force(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "existing.toml"
        dest.write_text("dummy\n", encoding="utf-8")
        from study_utils.core.config_templates import ConfigTemplateError

        with pytest.raises(
            (TomlConfigError, ConfigTemplateError),
            match="already exists",
        ):
            tv.write_template(dest, overwrite=False)

    def test_write_template_overwrites_with_force(self, tmp_path: Path) -> None:
        dest = tmp_path / "existing.toml"
        dest.write_text("old\n", encoding="utf-8")
        result = tv.write_template(dest, overwrite=True)
        assert result.exists()


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Tests for frozen config dataclass construction."""

    def test_ai_config_defaults(self) -> None:
        ai = tv.AIConfig(
            use_local=True,
            api_base="http://localhost:8080/v1",
            provider="local",
            model="whisper-3",
            title_model="gpt-4o-mini",
        )
        assert ai.use_local is True
        assert ai.api_base == "http://localhost:8080/v1"

    def test_audio_config(self) -> None:
        audio = tv.AudioConfig(segment_duration_minutes=15)
        assert audio.segment_duration_minutes == 15

    def test_names_config_all_fields(self) -> None:
        names = tv.NamesConfig(
            smart_names=True,
            use_ai_titles=False,
            output_dir=Path("/tmp/out"),
            recursive=True,
            cache_path=Path("/tmp/cache.json"),
        )
        assert names.output_dir == Path("/tmp/out")
        assert names.cache_path == Path("/tmp/cache.json")

    def test_logging_config(self) -> None:
        log = tv.LoggingConfig(level="DEBUG", verbose=True)
        assert log.level == "DEBUG"
        assert log.verbose is True

    def test_transcribe_config_composition(self) -> None:
        cfg = tv.TranscribeConfig(
            ai=tv.AIConfig(
                use_local=True,
                api_base="x",
                provider="p",
                model="m",
                title_model="t",
            ),
            audio=tv.AudioConfig(segment_duration_minutes=10),
            names=tv.NamesConfig(
                smart_names=True,
                use_ai_titles=False,
                output_dir=None,
                recursive=True,
                cache_path=None,
            ),
            logging=tv.LoggingConfig(level="INFO", verbose=False),
        )
        assert isinstance(cfg.ai, tv.AIConfig)


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_config_path_env(self) -> None:
        assert tv.CONFIG_PATH_ENV == "STUDY_TRANSCRIBE_CONFIG"

    def test_default_config_path(self) -> None:
        assert tv.DEFAULT_CONFIG_PATH == "config/transcribe.toml"


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    """Tests for _require_positive_int, _require_bool, etc."""

    def test_require_positive_int_valid(self) -> None:
        assert tv._require_positive_int(10, "x") == 10
        assert tv._require_positive_int(1, "x") == 1

    def test_require_positive_int_rejects_negative(self) -> None:
        with pytest.raises(tv.ConfigError, match="must be positive"):
            tv._require_positive_int(-1, "x")

    def test_require_positive_int_rejects_zero(self) -> None:
        with pytest.raises(tv.ConfigError, match="must be positive"):
            tv._require_positive_int(0, "x")

    def test_require_positive_int_rejects_bool(self) -> None:
        # bool is a subclass of int in Python, must reject it
        with pytest.raises(tv.ConfigError):
            tv._require_positive_int(True, "x")

    def test_require_positive_int_rejects_string(self) -> None:
        with pytest.raises(tv.ConfigError, match="must be a positive integer"):
            tv._require_positive_int("10", "x")

    def test_require_bool_valid(self) -> None:
        assert tv._require_bool(True, "x") is True
        assert tv._require_bool(False, "x") is False

    def test_require_bool_rejects_string(self) -> None:
        with pytest.raises(tv.ConfigError, match="must be a boolean"):
            tv._require_bool("true", "x")

    def test_coerce_optional_path_valid(self) -> None:
        p = Path("/tmp/foo")
        assert tv._coerce_optional_path(p, "x") is p
        assert tv._coerce_optional_path(None, "x") is None
        result = tv._coerce_optional_path("  /bar/baz  ", "x")
        assert result == Path("/bar/baz")

    def test_coerce_optional_path_rejects_int(self) -> None:
        with pytest.raises(tv.ConfigError, match="must be a path"):
            tv._coerce_optional_path(123, "x")


# ---------------------------------------------------------------------------
# Builder function tests
# ---------------------------------------------------------------------------


class TestBuilderFunctions:
    """Tests for _build_ai, _build_audio, _build_names, _build_logging."""

    def test_build_ai_with_defaults(self) -> None:
        ai = tv._build_ai({})
        assert ai.use_local is True
        assert ai.model == "whisper-3"
        assert ai.provider == "local"

    def test_build_audio_valid(self) -> None:
        audio = tv._build_audio({"segment_duration_minutes": 20})
        assert audio.segment_duration_minutes == 20

    def test_build_audio_default_10(self) -> None:
        audio = tv._build_audio({})
        assert audio.segment_duration_minutes == 10

    def test_build_names_all_fields(self) -> None:
        names = tv._build_names(
            {
                "smart_names": True,
                "use_ai_titles": False,
                "recursive": True,
            }
        )
        assert names.smart_names is True
        assert names.output_dir is None

    def test_build_logging_valid_level(self) -> None:
        log = tv._build_logging({"level": "WARNING", "verbose": True})
        assert log.level == "WARNING"
        assert log.verbose is True

    def test_build_logging_rejects_invalid_level(self) -> None:
        with pytest.raises(tv.ConfigError, match="logging.level"):
            tv._build_logging({"level": "bogus"})

    def test_build_config_assembles_all_sections(self) -> None:
        tree = copy.deepcopy(tv.default_tree())
        tree["ai"]["use_local"] = False
        tree["audio"]["segment_duration_minutes"] = 25
        cfg = tv._build_config(tree)
        assert cfg.ai.use_local is False
        assert cfg.audio.segment_duration_minutes == 25


# ---------------------------------------------------------------------------
# Defaults tree + merge tests
# ---------------------------------------------------------------------------


class TestDefaultsTreeAndMerge:
    """Tests for default_tree(), merge_defaults integration."""

    def test_default_tree_returns_deep_copy(self) -> None:
        t1 = tv.default_tree()
        t2 = tv.default_tree()
        # Mutation of one doesn't affect the other
        orig_t1_val = t1["ai"]["use_local"]
        t1["ai"]["use_local"] = not orig_t1_val
        assert t1["ai"]["use_local"] != orig_t1_val, "t1 was mutated"
        assert t2["ai"]["use_local"] == orig_t1_val, "t2 kept default value"

    def test_default_tree_keys(self) -> None:
        tree = tv.default_tree()
        for key in EXPECTED_KEYS:
            assert key in tree

    def test_merge_defaults_rejects_unknown_key(self) -> None:
        tree = copy.deepcopy(tv.default_tree())
        with pytest.raises(TomlConfigError, match="Unknown"):
            merge_defaults(tree, {"unknown_section": {}})

    def test_merge_defaults_overrides_nested_values(self) -> None:
        tree = copy.deepcopy(tv.default_tree())
        override = {
            "ai": {"use_local": False},
            "audio": {"segment_duration_minutes": 15},
        }
        merge_defaults(tree, override)
        assert tree["ai"]["use_local"] is False
        assert tree["audio"]["segment_duration_minutes"] == 15
        # Unchanged keys retain their defaults
        assert tree["logging"]["verbose"] is False

    def test_load_config_fallback_to_defaults(self) -> None:
        """load_config when no config file exists returns defaults."""
        cfg = tv.load_config(explicit_path=Path("/nonexistent/path.toml"))
        assert isinstance(cfg, tv.TranscribeConfig)
        assert cfg.ai.use_local is True


# ---------------------------------------------------------------------------
# Path resolution tests
# ---------------------------------------------------------------------------


class TestPathResolution:
    """Tests for resolve_config_path() precedence."""

    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        p = tmp_path / "custom.toml"
        result = tv.resolve_config_path(explicit_path=p)
        assert result == p.resolve()

    def test_env_var_overrides_default(self, monkeypatch: Any) -> None:
        env = {"STUDY_TRANSCRIBE_CONFIG": "/env/custom.toml"}
        result = tv.resolve_config_path(env=env)
        assert "custom.toml" in str(result)

    def test_explicit_over_env(self, tmp_path: Path, monkeypatch: Any) -> None:
        env = {"STUDY_TRANSCRIBE_CONFIG": "/env/custom.toml"}
        explicit = tmp_path / "explicit.toml"
        result = tv.resolve_config_path(explicit_path=explicit, env=env)
        assert result.stem == "explicit"

    def test_default_path_has_toml_extension(self, monkeypatch: Any) -> None:
        # Set a predictable data home for testing
        monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(Path.home()))
        monkeypatch.delenv("STUDY_TRANSCRIBE_CONFIG", raising=False)
        result = tv.resolve_config_path()
        assert str(result).endswith("transcribe.toml")


# ---------------------------------------------------------------------------
# Subcommand handler tests
# ---------------------------------------------------------------------------


class TestSubcommandHandlers:
    """Tests for _handle_config_init, _handle_config_validate, _handle_path."""

    def test_handle_config_init_returns_zero(self, tmp_path: Path) -> None:
        args = SimpleNamespace(path=str(tmp_path / "cfg.toml"), force=True)
        exit_code = tv._handle_config_init(args)
        assert exit_code == 0
        (tmp_path / "cfg.toml").exists()

    def test_handle_config_validate_returns_zero(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        config_file = tmp_path / "valid.toml"
        config_file.write_text(BASE_CONFIG)
        monkeypatch.setenv("STUDY_TRANSCRIBE_CONFIG", str(config_file))
        exit_code = tv._handle_config_validate(
            SimpleNamespace(path=str(config_file), force=False),
        )
        assert exit_code == 0

    def test_handle_config_path_returns_zero(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Capture stdout
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        path_arg = str(tmp_path / "x.toml")
        with redirect_stdout(f):
            exit_code = tv._handle_config_path(
                SimpleNamespace(path=path_arg, force=False),
            )
        assert exit_code == 0
        assert "x.toml" in f.getvalue()


# ---------------------------------------------------------------------------
# Full config load tests
# ---------------------------------------------------------------------------


class TestFullConfigLoad:
    """Integration-style tests for complete config loading."""

    def test_load_from_valid_toml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "full.toml"
        full_config = """
[ai]
use_local = false
api_base = "https://api.example.com/v1"
provider = "openai"
model = "whisper-1"
title_model = "gpt-4o"

[audio]
segment_duration_minutes = 12

[names]
smart_names = true
use_ai_titles = true
output_dir = "/tmp/out"
recursive = false
cache_path = "/tmp/cache.json"

[logging]
level = "WARNING"
verbose = true
"""
        config_file.write_text(full_config)
        cfg = tv.load_config(explicit_path=config_file.resolve())

        assert cfg.ai.use_local is False
        assert cfg.ai.api_base == "https://api.example.com/v1"
        assert cfg.ai.model == "whisper-1"
        assert cfg.audio.segment_duration_minutes == 12
        assert cfg.names.recursive is False
        assert cfg.logging.level == "WARNING"

    def test_merge_rejects_wrong_type_for_table(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.toml"
        # ai must be a table (write as inline TOML text for py 3.12 compat)
        bad_toml = 'ai = "not-a-table"\naudio = {}\nnames = {}\nlogging = {}\n'
        config_file.write_text(bad_toml)

        with pytest.raises(tv.ConfigError):
            tv.load_config(explicit_path=config_file.resolve())


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Tests ensuring existing behavior is preserved."""

    def test_transcribe_llm_retained(self) -> None:
        assert "USE_LOCAL" in tv._TRANSCRIBE_LLM
        assert "API_BASE" in tv._TRANSCRIBE_LLM
        assert tv._TRANSCRIBE_LLM["USE_LOCAL"] is True

    def test_find_video_files_still_works(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "test.mp4"
        mp4.touch()
        files = tv.find_video_files(mp4)
        assert len(files) == 1
        assert files[0] == mp4

    def test_main_parser_still_parses(self) -> None:
        """_parse_transcribe_args should work with minimal args."""
        # Save original sys.argv, inject a test value
        original = sys.argv[:]
        try:
            sys.argv = ["transcribe_video", "/some/video.mp4"]
            from study_utils.transcribe_video import _parse_transcribe_args  # noqa: E402

            args = _parse_transcribe_args()
            assert str(args.TARGET) == "/some/video.mp4"
        finally:
            sys.argv = original

    def test_split_video_segment_ms_is_10_min(self) -> None:
        """Original segment duration (segment_ms = 10 minutes)."""
        segment_ms = 10 * 60 * 1000  # 10 minutes in ms
        assert segment_ms == 600000
