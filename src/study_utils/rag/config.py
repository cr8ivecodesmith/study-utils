"""Configuration management for the Study RAG CLI.

Milestone 1 focuses on establishing a strongly validated configuration layer
backed by TOML. The config file groups related concerns so later milestones can
extend them without breaking compatibility.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from study_utils.core import config as core_config

from . import data_dir


CONFIG_PATH_ENV = "STUDY_RAG_CONFIG"


class ConfigError(RuntimeError):
    """Raised when configuration parsing or validation fails."""


@dataclass(frozen=True)
class PathsConfig:
    data_home_override: Optional[Path]


@dataclass(frozen=True)
class OpenAIConfig:
    chat_model: str
    embedding_model: str
    max_input_tokens: int
    max_output_tokens: int
    temperature: float
    api_base: Optional[str]
    request_timeout_seconds: int


@dataclass(frozen=True)
class ProvidersConfig:
    default: str
    openai: OpenAIConfig


@dataclass(frozen=True)
class ChunkingConfig:
    tokenizer: str
    encoding: str
    tokens_per_chunk: int
    token_overlap: int
    fallback_delimiter: str


@dataclass(frozen=True)
class DedupConfig:
    strategy: str
    checksum_algorithm: str


@dataclass(frozen=True)
class IngestionConfig:
    chunking: ChunkingConfig
    dedupe: DedupConfig
    max_workers: int
    file_batch_size: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    max_context_tokens: int
    score_threshold: float


@dataclass(frozen=True)
class ChatConfig:
    max_history_turns: int
    response_tokens: int
    stream: bool


@dataclass(frozen=True)
class ServiceAIConfig:
    use_local: bool
    api_base: str
    provider: str


@dataclass(frozen=True)
class ServicesAIConfig:
    chat: ServiceAIConfig
    embeddings: ServiceAIConfig


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    verbose: bool


@dataclass(frozen=True)
class RagConfig:
    paths: PathsConfig
    providers: ProvidersConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    chat: ChatConfig
    services: ServicesAIConfig
    logging: LoggingConfig

    @property
    def data_home(self) -> Path:
        """Return the effective data home considering overrides."""

        if self.paths.data_home_override is not None:
            return self.paths.data_home_override
        return data_dir.get_data_home(create=False)


def _require_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{field}' must be a positive integer.")
    return value


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{field}' must be a non-negative integer.")
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{field}' must be a boolean.")
    return value


def _require_float_range(
    value: Any, *, field: str, min_value: float, max_value: float
) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"'{field}' must be a number.")
    number = float(value)
    if not (min_value <= number <= max_value):
        raise ConfigError(
            f"'{field}' must be between {min_value} and {max_value}."
        )
    return number


def _require_string(
    value: Any, *, field: str, allow_whitespace: bool = False
) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"'{field}' must be a non-empty string.")
    if allow_whitespace:
        if value == "":
            raise ConfigError(f"'{field}' must be a non-empty string.")
        return value
    trimmed = value.strip()
    if not trimmed:
        raise ConfigError(f"'{field}' must be a non-empty string.")
    return trimmed


def _coerce_optional_string(value: Any, *, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field}' must be a non-empty string when set.")
    return value.strip()


def _coerce_optional_path(value: Any, *, field: str) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{field}' must be a non-empty string when set.")
    return Path(value).expanduser().resolve()


def _build_paths(section: Mapping[str, Any]) -> PathsConfig:
    raw_path = section.get("data_home")
    override = _coerce_optional_path(
        raw_path,
        field="paths.data_home",
    )
    return PathsConfig(data_home_override=override)


def _build_openai(section: Mapping[str, Any]) -> OpenAIConfig:
    chat_model = _require_string(
        section.get("chat_model"),
        field="providers.openai.chat_model",
    )
    embedding_model = _require_string(
        section.get("embedding_model"),
        field="providers.openai.embedding_model",
    )
    max_input_tokens = _require_positive_int(
        section.get("max_input_tokens"),
        field="providers.openai.max_input_tokens",
    )
    max_output_tokens = _require_positive_int(
        section.get("max_output_tokens"),
        field="providers.openai.max_output_tokens",
    )
    temperature = _require_float_range(
        section.get("temperature"),
        field="providers.openai.temperature",
        min_value=0.0,
        max_value=2.0,
    )
    api_base = _coerce_optional_string(
        section.get("api_base"), field="providers.openai.api_base"
    )
    request_timeout_seconds = _require_positive_int(
        section.get("request_timeout_seconds"),
        field="providers.openai.request_timeout_seconds",
    )
    return OpenAIConfig(
        chat_model=chat_model,
        embedding_model=embedding_model,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        api_base=api_base,
        request_timeout_seconds=request_timeout_seconds,
    )


def _build_providers(section: Mapping[str, Any]) -> ProvidersConfig:
    default_provider = _require_string(
        section.get("default"), field="providers.default"
    )
    openai_section = section.get("openai")
    if not isinstance(openai_section, Mapping):
        raise ConfigError("providers.openai table is required.")
    openai_config = _build_openai(openai_section)
    if default_provider != "openai":
        raise ConfigError(
            "Only the 'openai' provider is supported at launch; "
            "set providers.default to 'openai'."
        )
    return ProvidersConfig(default=default_provider, openai=openai_config)


def _build_chunking(section: Mapping[str, Any]) -> ChunkingConfig:
    tokenizer = _require_string(
        section.get("tokenizer"), field="ingestion.chunking.tokenizer"
    )
    encoding = _require_string(
        section.get("encoding"), field="ingestion.chunking.encoding"
    )
    tokens_per_chunk = _require_positive_int(
        section.get("tokens_per_chunk"),
        field="ingestion.chunking.tokens_per_chunk",
    )
    token_overlap = _require_non_negative_int(
        section.get("token_overlap"),
        field="ingestion.chunking.token_overlap",
    )
    if token_overlap >= tokens_per_chunk:
        raise ConfigError(
            "ingestion.chunking.token_overlap must be smaller than "
            "tokens_per_chunk."
        )
    fallback_delimiter = _require_string(
        section.get("fallback_delimiter"),
        field="ingestion.chunking.fallback_delimiter",
        allow_whitespace=True,
    )
    return ChunkingConfig(
        tokenizer=tokenizer,
        encoding=encoding,
        tokens_per_chunk=tokens_per_chunk,
        token_overlap=token_overlap,
        fallback_delimiter=fallback_delimiter,
    )


def _build_dedupe(section: Mapping[str, Any]) -> DedupConfig:
    strategy = _require_string(
        section.get("strategy"), field="ingestion.dedupe.strategy"
    )
    checksum_algorithm = _require_string(
        section.get("checksum_algorithm"),
        field="ingestion.dedupe.checksum_algorithm",
    )
    if strategy.lower() != "checksum":
        raise ConfigError(
            "ingestion.dedupe.strategy must be 'checksum' "
            "for the initial release."
        )
    return DedupConfig(
        strategy=strategy.lower(),
        checksum_algorithm=checksum_algorithm,
    )


def _build_ingestion(section: Mapping[str, Any]) -> IngestionConfig:
    chunking_section = section.get("chunking")
    if not isinstance(chunking_section, Mapping):
        raise ConfigError("ingestion.chunking table is required.")
    dedupe_section = section.get("dedupe")
    if not isinstance(dedupe_section, Mapping):
        raise ConfigError("ingestion.dedupe table is required.")
    chunking = _build_chunking(chunking_section)
    dedupe = _build_dedupe(dedupe_section)
    max_workers = _require_positive_int(
        section.get("max_workers"), field="ingestion.max_workers"
    )
    file_batch_size = _require_positive_int(
        section.get("file_batch_size"), field="ingestion.file_batch_size"
    )
    return IngestionConfig(
        chunking=chunking,
        dedupe=dedupe,
        max_workers=max_workers,
        file_batch_size=file_batch_size,
    )


def _build_retrieval(section: Mapping[str, Any]) -> RetrievalConfig:
    top_k = _require_positive_int(section.get("top_k"), field="retrieval.top_k")
    max_context_tokens = _require_positive_int(
        section.get("max_context_tokens"),
        field="retrieval.max_context_tokens",
    )
    score_threshold = _require_float_range(
        section.get("score_threshold"),
        field="retrieval.score_threshold",
        min_value=0.0,
        max_value=1.0,
    )
    return RetrievalConfig(
        top_k=top_k,
        max_context_tokens=max_context_tokens,
        score_threshold=score_threshold,
    )


def _build_chat(section: Mapping[str, Any]) -> ChatConfig:
    max_history_turns = _require_positive_int(
        section.get("max_history_turns"), field="chat.max_history_turns"
    )
    response_tokens = _require_positive_int(
        section.get("response_tokens"), field="chat.response_tokens"
    )
    stream = _require_bool(section.get("stream"), field="chat.stream")
    return ChatConfig(
        max_history_turns=max_history_turns,
        response_tokens=response_tokens,
        stream=stream,
    )


def _build_logging(section: Mapping[str, Any]) -> LoggingConfig:
    level = _require_string(section.get("level"), field="logging.level").upper()
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed:
        raise ConfigError(
            "logging.level must be one of "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    verbose = _require_bool(section.get("verbose"), field="logging.verbose")
    return LoggingConfig(level=level, verbose=verbose)


def _build_service_ai(section: Mapping[str, Any]) -> ServiceAIConfig:
    use_local = _require_bool(
        section.get("use_local", True), field="services.use_local"
    )
    api_base = _coerce_optional_string(
        section.get("api_base", "http://localhost:8080/v1"),
        field="services.api_base",
    )
    if api_base is None:
        api_base = "http://localhost:8080/v1"
    raw_provider = section.get("provider")
    if raw_provider is None:
        raw_provider = "local"
    provider = _require_string(
        raw_provider, field="services.provider"
    )
    return ServiceAIConfig(
        use_local=use_local,
        api_base=api_base,
        provider=provider,
    )


def _build_services(section: Mapping[str, Any]) -> ServicesAIConfig:
    chat_section = section.get("chat", {})
    if not isinstance(chat_section, Mapping):
        raise ConfigError("services.chat must be a mapping.")
    embeddings_section = section.get("embeddings", {})
    if not isinstance(embeddings_section, Mapping):
        raise ConfigError("services.embeddings must be a mapping.")
    chat_ai = _build_service_ai(chat_section)
    embeddings_ai = _build_service_ai(embeddings_section)
    return ServicesAIConfig(
        chat=chat_ai,
        embeddings=embeddings_ai,
    )


def _build_config(tree: Mapping[str, Any]) -> RagConfig:
    paths_section = tree.get("paths", {})
    providers_section = tree.get("providers", {})
    ingestion_section = tree.get("ingestion", {})
    retrieval_section = tree.get("retrieval", {})
    chat_section = tree.get("chat", {})
    services_section = tree.get("services", {})
    logging_section = tree.get("logging", {})

    if not isinstance(paths_section, Mapping):
        raise ConfigError("paths table must be a mapping if provided.")
    if not isinstance(providers_section, Mapping):
        raise ConfigError("providers table must be a mapping.")
    if not isinstance(ingestion_section, Mapping):
        raise ConfigError("ingestion table must be a mapping.")
    if not isinstance(retrieval_section, Mapping):
        raise ConfigError("retrieval table must be a mapping.")
    if not isinstance(chat_section, Mapping):
        raise ConfigError("chat table must be a mapping.")
    if not isinstance(services_section, Mapping):
        raise ConfigError("services table must be a mapping.")
    if not isinstance(logging_section, Mapping):
        raise ConfigError("logging table must be a mapping.")

    return RagConfig(
        paths=_build_paths(paths_section),
        providers=_build_providers(providers_section),
        ingestion=_build_ingestion(ingestion_section),
        retrieval=_build_retrieval(retrieval_section),
        chat=_build_chat(chat_section),
        services=_build_services(services_section),
        logging=_build_logging(logging_section),
    )


def resolve_config_path(
    *,
    explicit_path: Optional[Path] = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    env_map = os.environ if env is None else env
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    env_override = env_map.get(CONFIG_PATH_ENV)
    if env_override:
        return Path(env_override).expanduser().resolve()
    return data_dir.config_path(env=env_map, create_parent=True)


def load_config(
    *,
    explicit_path: Optional[Path] = None,
    env: Mapping[str, str] | None = None,
) -> RagConfig:
    """Load the TOML config, applying defaults and validation."""

    path = resolve_config_path(explicit_path=explicit_path, env=env)
    tree = copy.deepcopy(_DEFAULTS)
    try:
        toml_data = core_config.load_toml(path)
    except core_config.TomlConfigError as exc:
        raise ConfigError(str(exc)) from exc
    if not isinstance(toml_data, Mapping):
        raise ConfigError("Config TOML must contain a table at the root.")
    try:
        core_config.merge_defaults(tree, toml_data)
    except core_config.TomlConfigError as exc:
        raise ConfigError(str(exc)) from exc
    return _build_config(tree)


def default_tree() -> Dict[str, Any]:
    """Return a copy of the default configuration tree."""

    return copy.deepcopy(_DEFAULTS)


def config_template() -> str:
    """Return the TOML template recommended for new installs."""

    return _CONFIG_TEMPLATE.strip() + "\n"


def write_template(
    path: Path, *, overwrite: bool = False, mode: int = 0o600
) -> Path:
    """Write the default template to ``path``.

    Args:
        path: Destination TOML file.
        overwrite: When ``True`` existing files are replaced.
        mode: File permission bitmask to apply when supported.
    """

    try:
        return core_config.write_toml_template(
            path,
            template=config_template(),
            overwrite=overwrite,
            mode=mode,
        )
    except core_config.TomlConfigError as exc:
        raise ConfigError(str(exc)) from exc


_DEFAULTS: Dict[str, Any] = {
    "paths": {
        "data_home": None,
    },
    "providers": {
        "default": "openai",
        "openai": {
            "chat_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-large",
            "max_input_tokens": 6000,
            "max_output_tokens": 2000,
            "temperature": 0.2,
            "api_base": None,
            "request_timeout_seconds": 60,
        },
    },
    "ingestion": {
        "max_workers": 4,
        "file_batch_size": 32,
        "chunking": {
            "tokenizer": "tiktoken",
            "encoding": "cl100k_base",
            "tokens_per_chunk": 300,
            "token_overlap": 30,
            "fallback_delimiter": "\n\n",
        },
        "dedupe": {
            "strategy": "checksum",
            "checksum_algorithm": "sha256",
        },
    },
    "retrieval": {
        "top_k": 5,
        "max_context_tokens": 1800,
        "score_threshold": 0.2,
    },
    "chat": {
        "max_history_turns": 200,
        "response_tokens": 800,
        "stream": True,
    },
    "services": {
        "chat": {
            "use_local": True,
            "api_base": "http://localhost:8080/v1",
            "provider": "local",
        },
        "embeddings": {
            "use_local": True,
            "api_base": "http://localhost:8080/v1",
            "provider": "local",
        },
    },
    "logging": {
        "level": "INFO",
        "verbose": False,
    },
}


_CONFIG_TEMPLATE = """
# Study RAG configuration

[paths]
# Set to override the default data directory (~/.study-utils-data)
# data_home = "~/my-study-data"

[providers]
default = "openai"

[providers.openai]
# Primary chat completion model
chat_model = "gpt-4o-mini"
# Embedding model for vector stores
embedding_model = "text-embedding-3-large"
max_input_tokens = 6000
max_output_tokens = 2000
# Sampling temperature (0.0-2.0)
temperature = 0.2
# Optional API base override (leave blank for default)
# api_base = "https://api.openai.com/v1"
request_timeout_seconds = 60

[ingestion]
# Parallel workers when embedding documents
max_workers = 4
# Number of files to consider per batch before yielding control
file_batch_size = 32

[ingestion.chunking]
# Tokenizer implementation name
tokenizer = "tiktoken"
# tiktoken encoding to load
encoding = "cl100k_base"
# Target tokens per chunk and how many tokens to overlap
tokens_per_chunk = 300
token_overlap = 30
# Delimiter used when tokenizer fallback occurs
fallback_delimiter = "\\n\\n"

[ingestion.dedupe]
# Strategy for skipping previously ingested files
strategy = "checksum"
checksum_algorithm = "sha256"

[retrieval]
# Number of chunks to retrieve per query
top_k = 5
# Keep RAG prompts within this many tokens (including history)
max_context_tokens = 1800
# Drop candidates below this similarity score (0-1)
score_threshold = 0.2

[chat]
# Maximum turns to retain in session history
max_history_turns = 200
# Cap model responses to this many tokens
response_tokens = 800
# Stream responses to the terminal when supported
stream = true

[services]
# Per-service local LLM configuration. Each subsection controls settings for a
# specific AI service (chat, embeddings).  These are independently configurable
# so chat might use a local model while embeddings point to a cloud API or vice
# versa.

[services.chat]
# When true, prefer LOCAL_LLM_API_KEY over OPENAI_API_KEY.
use_local = true
# Base URL passed to the OpenAI SDK (base_url=...).
api_base = "http://localhost:8080/v1"
# Provider hint — "local" maps to key_source = "local".
provider = "local"

[services.embeddings]
use_local = true
api_base = "http://localhost:8080/v1"
provider = "local"

[logging]
level = "INFO"
verbose = false
"""
