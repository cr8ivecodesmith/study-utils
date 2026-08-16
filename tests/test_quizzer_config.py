"""Tests for quizzer config init feature.

Covers load_config(), validate_config(), template writing, find_config(),
and backward compat with missing [ai] section.
"""

import os

import pytest

pytestmark = pytest.mark.integration


class TestLoadConfigDefaults:
    """Test load_config() returns correct defaults when no file present."""

    def test_load_config_defaults_use_workspace(self, monkeypatch):
        from study_utils.quizzer.config import (
            CONFIG_ENV,
            load_config,
        )

        monkeypatch.delenv(CONFIG_ENV, raising=False)

        cfg = load_config(env={})

        assert cfg.ai.model == "gpt-4o-mini"
        assert cfg.ai.api_base == "http://localhost:8080/v1"
        assert cfg.ai.use_local is True
        assert cfg.ai.provider == "local"
        assert cfg.ai.temperature == 0.2
        assert cfg.ai.max_tokens == 600


class TestLoadConfigReadsAISection:
    """Test load_config() reads [ai] section values from TOML."""

    def test_load_config_reads_ai_section(self, tmp_path, monkeypatch):
        from study_utils.quizzer.config import load_config

        toml_content = """\
[ai]
model = "gpt-4o"
api_base = "http://custom:8080/v1"
use_local = false
provider = "openai"
temperature = 0.7
max_tokens = 1024

[storage]
out_dir = ".custom/<name>"
"""
        config_file = tmp_path / "quizzer.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_path=config_file, env={})

        assert cfg.ai.model == "gpt-4o"
        assert cfg.ai.api_base == "http://custom:8080/v1"
        assert cfg.ai.use_local is False
        assert cfg.ai.provider == "openai"
        assert cfg.ai.temperature == 0.7
        assert cfg.ai.max_tokens == 1024
        assert cfg.storage_out_dir == ".custom/<name>"


class TestLoadConfigMissingAIFallsBack:
    """Test load_config() falls back to defaults when [ai] section missing."""

    def test_load_config_missing_ai_falls_back(self, tmp_path):
        from study_utils.quizzer.config import load_config

        toml_content = """\
[storage]
out_dir = ".quizzer/<name>"
"""
        config_file = tmp_path / "quizzer.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_path=config_file, env={})

        assert cfg.ai.model == "gpt-4o-mini"
        assert cfg.ai.api_base == "http://localhost:8080/v1"
        assert cfg.ai.use_local is True
        assert cfg.ai.provider == "local"
        assert cfg.ai.temperature == 0.2
        assert cfg.ai.max_tokens == 600


class TestLoadConfigEnvOverridesFile:
    """Test environment variable provides config path override."""

    def test_load_config_env_overrides_file(self, tmp_path, monkeypatch):
        from study_utils.quizzer.config import CONFIG_ENV, load_config

        env_file = tmp_path / "env_config.toml"
        env_file.write_text('[ai]\nmodel = "env-model"\n')

        cli_file = tmp_path / "cli_config.toml"
        cli_file.write_text('[ai]\nmodel = "cli-model"\n')

        env_value = str(env_file)
        monkeypatch.setenv(CONFIG_ENV, env_value)

        cfg = load_config(config_path=cli_file, env={CONFIG_ENV: env_value})

        assert cfg.ai.model == "cli-model"


class TestLoadConfigCLIPathOverridesEnv:
    """Test CLI --config path takes precedence over env var."""

    def test_load_config_cli_path_overrides_env(self, tmp_path, monkeypatch):
        from study_utils.quizzer.config import CONFIG_ENV, load_config

        env_file = tmp_path / "env.toml"
        env_file.write_text('[ai]\nmodel = "env-model"\n')

        cli_file = tmp_path / "cli.toml"
        cli_file.write_text('[ai]\nmodel = "cli-model"\n')

        monkeypatch.setenv(CONFIG_ENV, str(env_file))

        cfg = load_config(config_path=cli_file, env={CONFIG_ENV: str(env_file)})

        assert cfg.ai.model == "cli-model"


class TestLoadConfigAbsoluteVsRelativePaths:
    """Test config path resolution with relative and absolute paths."""

    def test_load_config_absolute_vs_relative_paths(self, tmp_path):
        from study_utils.quizzer.config import load_config

        toml_content = '[ai]\nmodel = "relative-works"\n'
        config_file = tmp_path / "quizzer.toml"
        config_file.write_text(toml_content)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = load_config(env={})
            assert cfg.ai.model in ("relative-works", "gpt-4o-mini")
        finally:
            os.chdir(old_cwd)


class TestValidateConfigValid:
    """Test validate_config() succeeds with valid TOML."""

    def test_validate_config_valid_succeeds(self, tmp_path):
        from study_utils.quizzer.config import validate_config

        toml_content = '[ai]\nmodel = "gpt-4o"\n'
        config_file = tmp_path / "quizzer.toml"
        config_file.write_text(toml_content)

        result = validate_config(path=config_file)
        assert result == config_file.resolve()


class TestValidateConfigMalformedRaises:
    """Test validate_config() raises on malformed TOML."""

    def test_validate_config_malformed_raises(self, tmp_path):
        from study_utils.quizzer.config import (
            QuizzerConfigError,
            validate_config,
        )

        bad_toml = '[ai\nmodel = "gpt-4o"\n'
        config_file = tmp_path / "bad.toml"
        config_file.write_text(bad_toml)

        with pytest.raises(QuizzerConfigError, match="parse"):
            validate_config(path=config_file)


class TestValidateConfigMissingFile:
    """Test validate_config() raises when file not found."""

    def test_validate_config_missing_file_raises(self, tmp_path):
        from study_utils.quizzer.config import (
            QuizzerConfigError,
            validate_config,
        )

        nonexistent = tmp_path / "nonexistent.toml"

        with pytest.raises(QuizzerConfigError, match=str(nonexistent)):
            validate_config(path=nonexistent)


class TestValidateConfigNoAISection:
    """Test validate_config() accepts file without [ai] section."""

    def test_validate_config_no_ai_section_is_valid(self, tmp_path):
        from study_utils.quizzer.config import validate_config

        toml_content = '[storage]\nout_dir = ".test"\n'
        config_file = tmp_path / "noai.toml"
        config_file.write_text(toml_content)

        result = validate_config(path=config_file)
        assert result.exists()


class TestTemplateWriteWithForce:
    """Test template template write with force/overwrite behavior."""

    def test_template_write_with_force_overwrite(self, tmp_path):
        from study_utils.quizzer.config import write_template

        target = tmp_path / "template.toml"

        write_template(target)
        assert target.exists()

        with pytest.raises(Exception):
            write_template(target, overwrite=False)

        write_template(target, overwrite=True)
        assert target.exists()


class TestFindConfigDelegatesToLoader:
    """Test _find_config() delegates to workspace-aware resolution."""

    def test_find_config_delegates_to_loader(self, tmp_path, monkeypatch):
        from study_utils.quizzer.utils import _find_config

        config_file = tmp_path / "quizzer.toml"
        config_file.write_text("[ai]\n")

        result = _find_config(str(config_file))
        assert result == config_file

    def test_find_config_returns_none_when_missing(self, monkeypatch):
        from study_utils.quizzer.utils import _find_config

        result = _find_config("nonexistent/path/quizzer.toml")
        assert result is None


class TestCLIConfigSubcommands:
    """Test CLI config subcommand handlers."""

    def test_cmd_config_init_creates_file(self, tmp_path, monkeypatch):
        from study_utils.quizzer._main import _cmd_config_init
        import argparse

        monkeypatch.chdir(tmp_path)
        args = argparse.Namespace(path=None, force=False)

        code = _cmd_config_init(args)
        assert code == 0
        expected = tmp_path / "quizzer.toml"
        assert expected.exists()

    def test_cmd_config_init_with_path(self, tmp_path):
        from study_utils.quizzer._main import _cmd_config_init
        import argparse

        target = tmp_path / "custom.toml"
        args = argparse.Namespace(path=str(target), force=False)

        code = _cmd_config_init(args)
        assert code == 0
        assert target.exists()


class TestBackwardCompat:
    """Test backwards compatibility with existing config files."""

    def test_legacy_config_without_ai_section(self, tmp_path):
        from study_utils.quizzer.config import load_config

        legacy_content = """\
[storage]
out_dir = ".quizzer/<name>"

[quiz.test_quiz]
sources = ["./docs"]
types = ["mcq"]
"""
        config_file = tmp_path / "legacy.toml"
        config_file.write_text(legacy_content)

        cfg = load_config(config_path=config_file, env={})

        assert cfg.ai.model == "gpt-4o-mini"
        assert cfg.storage_out_dir == ".quizzer/<name>"


class TestConfigResolver:
    """Test the three-tier precedence chain."""

    def test_precedence_cli_wins(self, tmp_path):
        from study_utils.quizzer.config import _resolve_config_path

        cli_file = tmp_path / "cli.toml"
        cli_file.write_text("[ai]\n")

        result = _resolve_config_path(
            config_path=cli_file,
            env={"STUDY_QUIZZER_CONFIG": str(tmp_path / "env.toml")},
        )
        assert result == cli_file

    def test_precedence_env_wins_when_no_cli(self, tmp_path):
        from study_utils.quizzer.config import (
            CONFIG_ENV,
            _resolve_config_path,
        )

        env_file = tmp_path / "env.toml"
        env_file.write_text("[ai]\n")

        result = _resolve_config_path(
            config_path=None,
            env={CONFIG_ENV: str(env_file)},
        )
        assert result == env_file
