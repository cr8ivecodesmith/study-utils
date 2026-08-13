"""Packaged configuration templates for study-utils commands."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

from .config import TomlConfigError, write_toml_template

__all__ = [
    "ConfigTemplate",
    "ConfigTemplateError",
    "get_template",
    "iter_templates",
]


class ConfigTemplateError(RuntimeError):
    """Raised when a requested configuration template is not available."""


@dataclass(frozen=True)
class ConfigTemplate:
    """Metadata and helpers for a packaged configuration template."""

    name: str
    filename: str
    description: str
    package: str

    def read_text(self) -> str:
        """Return the template contents as UTF-8 text."""

        try:
            resource = resources.files(self.package).joinpath(self.filename)
            return resource.read_text(encoding="utf-8")
        except FileNotFoundError as exc:  # pragma: no cover - package state
            raise ConfigTemplateError(
                f"Template '{self.name}' resource not found."
            ) from exc
        except ModuleNotFoundError as exc:  # pragma: no cover - import safety
            message = (
                f"Package '{self.package}' not found for template "
                f"'{self.name}'."
            )
            raise ConfigTemplateError(message) from exc

    def write(
        self, path: Path, *, overwrite: bool = False, mode: int = 0o600
    ) -> Path:
        """Write the template to ``path`` leveraging TOML helper semantics."""

        try:
            return write_toml_template(
                path,
                template=self.read_text(),
                overwrite=overwrite,
                mode=mode,
            )
        except TomlConfigError as exc:
            raise ConfigTemplateError(str(exc)) from exc


_TEMPLATES: dict[str, ConfigTemplate] = {
    "generate_document": ConfigTemplate(
        name="generate_document",
        filename="documents.toml",
        description=(
            "Configuration defaults for the generate-document workflow."
        ),
        package="study_utils.generate_document",
    ),
    "convert_markdown": ConfigTemplate(
        name="convert_markdown",
        filename="template.toml",
        description=(
            "Configuration defaults for the document-to-Markdown converter."
        ),
        package="study_utils.convert_markdown",
    ),
    "transcribe_video": ConfigTemplate(
        name="transcribe_video",
        filename="transform.toml",
        description=(
            "Configuration defaults for the transcribe-video workflow."
        ),
        package="study_utils.transcribe_video",
    ),
}


def get_template(name: str) -> ConfigTemplate:
    """Return the template named ``name`` or raise an error."""

    try:
        return _TEMPLATES[name]
    except KeyError as exc:
        raise ConfigTemplateError(f"Unknown config template '{name}'.") from exc


def iter_templates() -> Iterable[ConfigTemplate]:
    """Yield all registered templates."""

    return tuple(_TEMPLATES.values())
