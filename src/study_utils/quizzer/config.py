"""Quizzer configuration loader and validator.

Provides ``QuizzerAIConfig`` dataclass, defaults resolution, TOML loading,
and environment-variable overrides following the three-tier precedence:

1. CLI ``--config`` argument (highest)
2. ``STUDY_QUIZZER_CONFIG`` environment variable
3. Workspace config directory (lowest)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TOML support not available. Use Python 3.11+ or install 'tomli'."
        ) from exc

from ..core.workspace import ensure_workspace

TEMPLATE_TEXT: str = ""  # Set below after `_DefaultTree` is defined.

__all__ = [
    "CONFIG_ENV",
    "CONFIG_FILENAME",
    "QuizzerAIConfig",
    "QuizzerConfigError",
    "LoadResult",
    "load_config",
    "validate_config",
]

CONFIG_FILENAME = "quizzer.toml"
CONFIG_ENV = "STUDY_QUIZZER_CONFIG"


@dataclass(frozen=True)
class QuizzerAIConfig:
    """Immutable AI connection configuration for quizzer."""

    model: str = "gpt-4o-mini"
    api_base: str = "http://localhost:8080/v1"
    use_local: bool = True
    provider: str = "local"
    temperature: float = 0.2
    max_tokens: int = 600


@dataclass(frozen=True)
class LoadResult:
    """Result of loading quizzer configuration."""

    ai: QuizzerAIConfig
    config_path: Optional[Path] = None
    storage_out_dir: str = ".quizzer/<name>"


# Internal cache for lazy-loaded config (memoization).
_cached_result: Optional[LoadResult] = None


@dataclass(frozen=True)
class _DefaultTree:
    """Default values as a nested dict matching the TOML structure."""

    ai: MutableMapping[str, Any] = field(
        default_factory=lambda: {
            "model": "gpt-4o-mini",
            "api_base": "http://localhost:8080/v1",
            "use_local": True,
            "provider": "local",
            "temperature": 0.2,
            "max_tokens": 600,
        }
    )
    storage: MutableMapping[str, Any] = field(
        default_factory=lambda: {
            "out_dir": ".quizzer/<name>",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {"ai": dict(self.ai), "storage": dict(self.storage)}


# Initialize TEMPLATE_TEXT from template.py when this module loads.
_template_text_loaded = False
try:
    from .template import TEMPLATE_TEXT as TemplateText  # noqa: N811 type: ignore

    TEMPLATE_TEXT = TemplateText
    _template_text_loaded = True
except (ImportError, ModuleNotFoundError):
    pass


DEFAULTS = _DefaultTree()


class QuizzerConfigError(RuntimeError):
    """Raised when quizzer config IO or validation fails."""


def _require_string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise QuizzerConfigError(
            f"[ai].{key} must be a string, got {type(value).__name__}"
        )
    return value


def _require_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise QuizzerConfigError(
            f"[ai].{key} must be a boolean, got {type(value).__name__}"
        )
    return value


def _require_float_range(value: Any, key: str, lo: float, hi: float) -> float:
    if not isinstance(value, (int, float)):
        raise QuizzerConfigError(
            f"[ai].{key} must be a number in range [{lo}, {hi}], "
            f"got {type(value).__name__}"
        )
    result = float(value)
    if result < lo or result > hi:
        raise QuizzerConfigError(
            f"[ai].{key} out of range [{lo}, {hi}]: {result}"
        )
    return result


def _require_positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise QuizzerConfigError(
            f"[ai].{key} must be a positive integer, got {type(value).__name__}"
        )
    if value <= 0:
        raise QuizzerConfigError(f"[ai].{key} must be positive: {value}")
    return value


def _build_ai(raw_section: Optional[Mapping[str, Any]]) -> QuizzerAIConfig:
    """Build a QuizzerAIConfig from raw TOML [ai] section dict."""
    defaults = DEFAULTS.ai
    if raw_section is None:
        return QuizzerAIConfig(
            model=_require_string(
                defaults.get("model", "gpt-4o-mini"), "model"
            ),
            api_base=_require_string(
                defaults.get("api_base", "http://localhost:8080/v1"),
                "api_base",
            ),
            use_local=_require_bool(
                defaults.get("use_local", True), "use_local"
            ),
            provider=_require_string(
                defaults.get("provider", "local"), "provider"
            ),
            temperature=_require_float_range(
                defaults.get("temperature", 0.2), "temperature", 0.0, 2.0
            ),
            max_tokens=_require_positive_int(
                defaults.get("max_tokens", 600), "max_tokens"
            ),
        )

    return QuizzerAIConfig(
        model=_require_string(
            raw_section.get("model", defaults["model"]), "model"
        ),
        api_base=_require_string(
            raw_section.get("api_base", defaults["api_base"]),
            "api_base",
        ),
        use_local=_require_bool(
            raw_section.get("use_local", defaults["use_local"]), "use_local"
        ),
        provider=_require_string(
            raw_section.get("provider", defaults["provider"]), "provider"
        ),
        temperature=_require_float_range(
            raw_section.get("temperature", defaults["temperature"]),
            "temperature",
            0.0,
            2.0,
        ),
        max_tokens=_require_positive_int(
            raw_section.get("max_tokens", defaults["max_tokens"]), "max_tokens"
        ),
    )


def _resolve_config_path(
    config_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    workspace_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve config path with CLI > env > workspace precedence."""
    env = env or os.environ

    # 1. CLI --config argument (highest priority)
    if config_path and config_path.exists():
        return config_path

    # 2. Environment variable override
    env_path = env.get(CONFIG_ENV)
    if env_path:
        resolved = Path(env_path).expanduser().resolve()
        if resolved.exists():
            return resolved

    # 3. Workspace config directory
    ws = workspace_path or ensure_workspace()
    config_dir = ws.path_for("config")
    candidate = config_dir / CONFIG_FILENAME
    if candidate.exists():
        return candidate

    return None


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    workspace_path: Optional[Path] = None,
) -> LoadResult:
    """Load and resolve quizzer configuration.

    Three-tier precedence (higher wins):
    1. CLI ``config_path`` argument
    2. ``STUDY_QUIZZER_CONFIG`` environment variable
    3. Workspace config directory (workspace/config/quizzer.toml)

    Args:
        config_path: Explicit path from CLI --config (highest priority).
        overrides: Additional key-value pairs to merge (currently unused,
            reserved for future per-command overrides).
        env: Environment mapping; defaults to ``os.environ``.
        workspace_path: Optional explicit workspace root path.

    Returns:
        LoadResult with resolved QuizzerAIConfig and config path.
    """
    global _cached_result

    env = env or os.environ
    resolved_path = _resolve_config_path(config_path, env, workspace_path)

    # Return cached result if no explicit args force a reload
    if config_path is None:
        if (
            _cached_result is not None
            and resolved_path == _cached_result.config_path
        ):
            return _cached_result
    elif _cached_result is not None:
        # An explicit config_path was passed, bypass cache.
        pass

    ai_section: Optional[Mapping[str, Any]] = None
    storage_out_dir = DEFAULTS.storage.get("out_dir", ".quizzer/<name>")

    if resolved_path and resolved_path.exists():
        try:
            raw = tomllib.load(resolved_path.open("rb"))
        except tomllib.TOMLDecodeError as exc:
            raise QuizzerConfigError(
                f"Failed to parse {resolved_path}: {exc}"
            ) from exc

        ai_section = raw.get("ai") if isinstance(raw.get("ai"), dict) else None
        storage_section = raw.get("storage")
        if isinstance(storage_section, dict):
            s_val = storage_section.get("out_dir")
            if isinstance(s_val, str):
                storage_out_dir = s_val

    ai_config = _build_ai(ai_section)
    _cached_result = LoadResult(
        ai=ai_config,
        config_path=resolved_path,
        storage_out_dir=storage_out_dir,
    )
    return _cached_result


def validate_config(
    path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Validate the resolved config file.

    Checks that the TOML file parses successfully and either has a valid
    ``[ai]`` section or falls back to defaults gracefully.

    Args:
        path: Explicit path to validate (resolves via config resolution).
        env: Environment mapping for resolution.

    Returns:
        The validated config path.

    Raises:
        QuizzerConfigError: If the file is missing, malformed, or invalid.
    """
    env = env or os.environ

    if path is not None:
        if not path.exists():
            raise QuizzerConfigError(f"Config file not found: {path}")
        try:
            with path.open("rb") as fh:
                tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise QuizzerConfigError(
                f"Failed to parse TOML in {path}: {exc}"
            ) from exc
        return path.resolve()

    # Resolve via standard precedence chain
    resolved = _resolve_config_path(config_path=path, env=env)
    if resolved is None:
        msg = "No config file found (CLI > env > workspace)."
        raise QuizzerConfigError(msg)

    try:
        with resolved.open("rb") as fh:
            tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise QuizzerConfigError(
            f"Failed to parse TOML in {resolved}: {exc}"
        ) from exc

    return resolved


def get_template_text() -> str:
    """Return the bundled template text."""
    return TEMPLATE_TEXT


def write_template(
    path: Path,
    *,
    overwrite: bool = False,
    mode: int = 0o600,
) -> Path:
    """Write the packaged template to *path*.

    Args:
        path: Destination file path.
        overwrite: If True, overwrite existing file.
        mode: File permission bits (default 0o600).

    Returns:
        The resolved destination path.
    """
    from ..core.config import write_toml_template

    template_text = get_template_text()
    return write_toml_template(
        path,
        template=template_text,
        overwrite=overwrite,
        mode=mode,
    )
