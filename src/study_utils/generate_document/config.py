"""Configuration helpers for the generate-document command."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from study_utils.core import workspace as workspace_mod
from study_utils.core.workspace import WorkspaceError


CONFIG_FILENAME = "documents.toml"


@dataclass(frozen=True)
class GenerateOptions:
    """CLI-facing options derived from argument parsing."""

    extensions: set[str]
    level_limit: int
    config_path: Path
    doc_type: str


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{field}' must be a boolean.")
    return value


def _require_string(
    value: Any, *, field: str, allow_whitespace: bool = False
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"'{field}' must be a string.")
    if allow_whitespace:
        if value == "":
            return value
        return value.strip()
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"'{field}' must be a non-empty string.")
    return trimmed


def _require_float_range(
    value: Any, *, field: str, min_value: float, max_value: float
) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{field}' must be a number.")
    number = float(value)
    if not (min_value <= number <= max_value):
        raise ValueError(
            f"'{field}' must be between {min_value} and {max_value}."
        )
    return number


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{field}' must be a positive integer.")
    return value


@dataclass(frozen=True)
class LLMConfig:
    """Local LLM connection configuration."""

    use_local: bool = True
    api_base: str = "http://localhost:8080/v1"
    provider: str = "local"
    temperature: float = 0.2
    max_tokens: int = 4096


_DEFAULT_LLM_CONFIG = LLMConfig()


def _build_llm(section: Mapping[str, Any]) -> LLMConfig:
    use_local = section.get("use_local", True)
    api_base = section.get("api_base")
    if not isinstance(api_base, str) or not api_base.strip():
        api_base = "http://localhost:8080/v1"
    provider = section.get("provider", "local")
    temperature = section.get("temperature", 0.2)
    max_tokens = section.get("max_tokens", 4096)
    return LLMConfig(
        use_local=_require_bool(use_local, field="llm.use_local"),
        api_base=_require_string(api_base, field="llm.api_base"),
        provider=_require_string(
            provider if isinstance(provider, str) else str(provider),
            field="llm.provider",
        ),
        temperature=_require_float_range(
            temperature, field="llm.temperature", min_value=0.0, max_value=2.0
        ),
        max_tokens=_require_positive_int(max_tokens, field="llm.max_tokens"),
    )


@dataclass(frozen=True)
class DocumentsConfig:
    """Wrapper for documents config with LLM connection settings."""

    llm: LLMConfig
    docs: Dict[str, Dict[str, str]]

    def __getitem__(self, key: str) -> Dict[str, str]:
        return self.docs[key]

    def __contains__(self, key: object) -> bool:
        return key in self.docs

    def keys(self):
        return self.docs.keys()

    def values(self):
        return self.docs.values()

    def items(self):
        return self.docs.items()


def _load_docs_section(raw: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Parse per-document-type entries from the raw TOML dictionary.

    Skips any key that doesn't have a ``prompt`` field (e.g. ``[llm]``).
    """
    data: Dict[str, Dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        prompt = value.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        entry = {
            "prompt": prompt.strip(),
            "description": str(value.get("description", "")).strip(),
            "model": str(
                value.get(
                    "model",
                    os.getenv("OPENAI_TITLE_MODEL", "gpt-4o-mini"),
                )
            ).strip(),
        }
        data[str(key).strip()] = entry
    return data


def load_documents_config(path: Path) -> DocumentsConfig:
    """Load a TOML file and return an enriched ``DocumentsConfig`` wrapper."""
    try:
        try:
            import tomllib  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for older Pythons
            import tomli as tomllib  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "Python 3.11+ required (tomllib) to read TOML config"
        ) from exc

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("documents.toml is empty or invalid")

    # Extract [llm] section BEFORE the per-doc-type loop so it is not skipped.
    llm_section: Dict[str, Any] = {}
    if "llm" in raw and isinstance(raw["llm"], dict):
        llm_section = raw.pop("llm")

    llm_cfg = _build_llm(llm_section)
    docs = _load_docs_section(raw)

    if not docs:
        raise ValueError(
            "documents.toml does not contain any valid document types"
        )

    return DocumentsConfig(llm=llm_cfg, docs=docs)


def load_documents_config_raw(path: Path) -> Dict[str, Dict[str, str]]:
    """Load a TOML file and return a plain dict of document types (legacy)."""
    try:
        try:
            import tomllib  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - fallback for older Pythons
            import tomli as tomllib  # type: ignore[import]
    except Exception as exc:
        raise RuntimeError(
            "Python 3.11+ required (tomllib) to read TOML config"
        ) from exc

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("documents.toml is empty or invalid")

    return _load_docs_section(raw)


def find_config_path(
    arg: Optional[str],
    *,
    workspace_path: Optional[Path] = None,
) -> Path:
    """Locate the documents config respecting CLI overrides and workspace."""

    if arg:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    cwd_cfg = Path.cwd() / CONFIG_FILENAME
    env_override = os.environ.get(workspace_mod.WORKSPACE_ENV, "")
    workspace_override = bool(workspace_path) or bool(env_override.strip())

    if not workspace_override and cwd_cfg.exists():
        return cwd_cfg.resolve()

    try:
        layout = workspace_mod.ensure_workspace(
            path=workspace_path,
            create=False,
        )
    except WorkspaceError as exc:
        raise FileNotFoundError(str(exc)) from exc

    workspace_candidate = layout.path_for("config") / CONFIG_FILENAME
    if workspace_candidate.exists():
        return workspace_candidate.resolve()

    if cwd_cfg.exists():
        return cwd_cfg.resolve()

    raise FileNotFoundError(
        "documents.toml not found. Run `study generate-document config init`."
    )
