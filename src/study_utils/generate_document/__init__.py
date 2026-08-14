"""Generate document package exposing the stable public surface."""

from .cli import build_arg_parser, main
from .config import (
    DocumentsConfig,
    LLMConfig,
    find_config_path,
    load_documents_config,
)
from .runner import build_messages, build_reference_block, generate_document

__all__ = [
    "build_arg_parser",
    "build_messages",
    "build_reference_block",
    "DocumentsConfig",
    "find_config_path",
    "generate_document",
    "LLMConfig",
    "load_documents_config",
    "main",
]
