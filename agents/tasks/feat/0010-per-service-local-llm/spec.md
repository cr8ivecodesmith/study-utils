# Per-Service Local LLM Config — Spec

## Summary

Add per-service configuration to control which API key (`OPENAI_API_KEY` vs `LOCAL_LLM_API_KEY`) and base URL each service type uses when calling `load_client()`. A new `[services]` section in the RAG config TOML gives each service an independent `use_local` flag (defaulting to `true`), a default `api_base` of `"http://localhost:8080/v1"` (matching the UAT setup), and a `provider` setting of `"local"`, so chat, embeddings, and title generation can each choose local or cloud independently.

## Goals

- Every service that calls `load_client()` can be configured via TOML to prefer LOCAL_LLM_API_KEY without changing the function's signature or behavior.
- New services added in the future only need a config entry — no code changes to ai.py required.
- Zero breaking changes: existing callers that do not read services config continue to behave exactly as before.

## Non-Goals

- No changes to `load_client()` in `src/study_utils/core/ai.py` — it remains a simple env-var-based factory with its current signature (`local: bool = False, api_base: str | None = None`).
- No changes to `load_llama_swap_upstream_client()` — llama-swap is unaffected by this work.
- No new top-level config file — `[services]` lives inside the existing RAG config TOML (or documents.toml for generate-document).
- CLI flag override is optional in this first pass; the core deliverable is TOML-based per-service config read at runtime.

## Behavior (BDD-ish)

### Default behavior

- Given no `[services]` section exists in the TOML, when a service reads the section, then all fields default to `use_local = true`, `api_base = "http://localhost:8080/v1"`, and `provider = "local"`.
- Given a services section defines `use_local = true` for `chat`, when `load_client(use_local=True, api_base="http://localhost:8080/v1")` is called, then the client uses `LOCAL_LLM_API_KEY` from environment.

### Key selection

- Given `[services.chat]` has `use_local = true` and both env vars are set, when a chat caller invokes `load_client()`, then `LOCAL_LLM_API_KEY` is used (not `OPENAI_API_KEY`).
- Given `[services.embeddings]` has `use_local = false`, when the embedding service calls `load_client(use_local=False)`, then it uses `OPENAI_API_KEY`.
- The default `provider = "local"` maps to UAT's `key_source = "local"` pattern, meaning local LLM is preferred by default across all services.

### API base override

- The default `api_base` for all services is `"http://localhost:8080/v1"` (matching the UAT setup). A service-specific `api_base = null` means "use the default".
- Given `[services.chat]` has a custom value like `api_base = "http://example.com/v2"`, when the chat client is created, then `base_url` is set to that overridden value.
- Given both keys are set in `.env`, then `LOCAL_LLM_API_KEY` takes precedence for services where `use_local = true`.

### Granular independence

- Given `[services.chat.use_local = true]` but `[services.embeddings.use_local = false]`, when chat and embeddings both call `load_client()` concurrently, they each independently choose their configured API key.
- Changes to one service's config do not affect others — no shared global "local mode" toggle.

### Load client backward compatibility

- Given an existing caller invokes `load_client()` with positional or keyword arguments (e.g., `load_client(local=True, api_base="http://...")`), when new `[services]` config is added to the TOML, then the function continues to work identically — callers that also pass service-specific values override the defaults.
- If a caller reads services config but both are absent for a field, the resolved default (`use_local=true`, `api_base="http://localhost:8080/v1"`, `provider="local"`) still works correctly: `load_client()` simply prefers LOCAL_LLM_API_KEY which is either set or will fallback to OPENAI_API_KEY.

## Design

### New data structures in `rag/config.py`

Three new frozen dataclasses and one addition to `RagConfig`:

```python
@dataclass(frozen=True)
class ServiceAIConfig:
    use_local: bool = True           # prefer LOCAL_LLM_API_KEY when true
    api_base: Optional[str] = "http://localhost:8080/v1"  # passed as base_url to OpenAI client
    provider: Literal["local", "openai"] = "local"        # maps to key_source

@dataclass(frozen=True)
class ServicesAIConfig:
    chat: ServiceAIConfig      # chat completions (gemma4-e4b by default)
    embeddings: ServiceAIConfig  # vector embedding generation
```

The `RagConfig` dataclass grows one new field:

```python
@dataclass(frozen=True)
class RagConfig:
    paths: PathsConfig
    providers: ProvidersConfig         # existing OpenAIConfig unchanged
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    chat: ChatConfig
    logging: LoggingConfig
    
    services: ServicesAIConfig          # NEW — per-service granular config
```

### TOML layout

Added as a new section at the end of the existing config template:

```toml
[paths]
data_home = null

[providers]
default = "openai"

[providers.openai]
chat_model = "gemma4-e4b"
embedding_model = "qwen3-embedding"
max_input_tokens = 6000
max_output_tokens = 2000
temperature = 0.2
api_base = null
request_timeout_seconds = 60

[ingestion]
# — unchanged —

[retrieval]
# — unchanged —

[chat]
# — unchanged —

[logging]
level = "INFO"
verbose = false

# NEW SECTION — per-service granular AI settings
[services]
[services.chat]
use_local = true
api_base = "http://localhost:8080/v1"  # matches UAT (local_client_config.toml)
provider = "local"                        # local | openai

[services.embeddings]
use_local = true
api_base = "http://localhost:8080/v1"
provider = "local"

[services.transcription]
use_local = true
api_base = "http://localhost:8080/v1"
model = "whisper-3"
```

### Defaults (`_DEFAULTS` dict in `rag/config.py`)

```python
"services": {
    "chat": {
        "use_local": True,
        "api_base": "http://localhost:8080/v1",
        "provider": "local",   # maps to key_source pattern
    },
    "embeddings": {
        "use_local": True,
        "api_base": "http://localhost:8080/v1",
        "provider": "local",
    },
    "transcription": {
        "use_local": True,
        "api_base": "http://localhost:8080/v1",
        "model": "whisper-3",   # configurable via TRANSCRIPTION_MODEL env var
    },
}
```

### Config resolution pipeline

Config values flow through the same precedence as existing RAG config:

```python
explicit_path arg (CLI --config)
  → STUDY_RAG_CONFIG env var
  → workspace config path
  → _DEFAULTS built-in defaults
  → merge_defaults() -> _build_config() -> RagConfig.services
```

### Caller wiring

Each service reads its own config field and passes the resolved values to `load_client()`:

| Service | Caller module(s) | Resolution | Call site |
|---------|-----------------|------------|-----------|
| **embeddings** | `rag.ingest._build_openai_client()` | reads from cfg.services.embeddings | `_build_embedder()` in cli.py |
| **chat (RAG)** | `rag.chat.OpenAIChatClient` | reads from cfg.services.chat | `_build_chat_client()` in cli.py |
| **chat (generate-document)** | `runner.generate_document()` | reads from document config or defaults | runner.py line ~78 |
| **chat (quizzer)** | `quiz._main`, `manager.quiz` | reads from quizzer.toml [ai] section | _main.py line ~164, quiz.py line ~262 |
| **transcription** | `transcribe_video.py` | reads from `_TRANSCRIBE_LLM["MODEL"]`, env `TRANSCRIPTION_MODEL` | `transcribe_audio_file()`, `main()` |

### generate-document extension

The `documents.toml` template can optionally define a top-level `[ai]` table:

```toml
[ai]
use_local = true
api_base = "http://localhost:8080/v1"
provider = "local"
chat_model = "gemma4-e4b"

[keywords]
model = "gemma4-e4b"
prompt = """..."""

[reading_assignment]
model = "gemma4-e4b"
use_local = true          # per-section override

[book]
model = "gemma4-e4b"
```

The `generate_document/config.py` `load_documents_config()` parser will handle the optional `[ai]` section. Per-document-type overrides (e.g., `reading_assignment.use_local`) coexist with a top-level `[ai]` default via the same pattern as existing model/prompt resolution.

### quizzer extension

The existing `[ai]` section in `quizzer.toml` grows three new fields:

```toml
[ai]
model = "gemma4-e4b"
use_local = true          # NEW: force LOCAL_LLM_API_KEY
local_api_base = null     # NEW (same semantics as api_base under the hood)  
provider = "local"        # NEW: matches UAT key_source pattern
temperature = 0.2
max_tokens = 600
```

The `quizzer._main` and `manager.quiz` modules pass these to `load_client()` at their respective call sites.

## Constraints & Dependencies

- Zero new dependencies — all code builds on existing `openai`, `tomllib`, `python-dotenv`, and `dataclasses`.
- RAG config is the primary home for `[services]`; generate-document and quizzer use lightweight copies of the same pattern in their respective TOML files.
- Python 3.10+ (already a project requirement due to existing type hints).
- Default `api_base` `"http://localhost:8080/v1"` matches `uat/local_client_config.toml` pattern (same port structure, localhost IP instead of the LAN address used in UAT).

## Security & Privacy

- No change to how API keys are stored or transmitted — `LOCAL_LLM_API_KEY` and `OPENAI_API_KEY` continue to be read from `.env` or environment, identical to current behavior.
- No secrets written to disk beyond the existing config TOML files (which already have `0o600` permissions in templates).

## Telemetry & Operability

- No new logs or metrics required — the change is purely configuration-driven and transparent at runtime.
- Existing RAG logging (`logging.log`, console output) will automatically reflect which API key was chosen, since the OpenAI client prints its own connection details.

## Rollout / Revert

- **Rollout**: add `[services]` section to config template; existing TOML files merge in via `merge_defaults()` without requiring explicit new fields.
- **Revert**: remove `[services]` — callers fall back to built-in `_DEFAULTS` (`use_local=True`, `api_base="http://localhost:8080/v1"`, `provider="local"`), which is backward-compatible with current behavior.

## Definition of Done

- [ ] `[services]` section added to `rag/config.py` dataclasses and defaults
- [ ] RAG embedder passes service config through to `load_client()`
- [ ] RAG chat client passes service config through to `load_client()`
- [ ] generate-document runner reads `[ai]` from documents.toml and passes to `load_client()`
- [ ] quizzer passes `[ai.use_local]` to `load_client()`
- [ ] transcription uses configurable model via `_TRANSCRIBE_LLM["MODEL"]`, default `"whisper-3"`
- [ ] All callers verified: 10 call sites across RAG, generate_document, quizzer, transcribe_video fully wired
- [ ] Unit tests cover: empty section defaults, custom values per service, key resolution by service, api_base passthrough
- [ ] Existing ~530 tests remain passing; zero new failures

## Ownership

- Owner: @matt
- Stakeholders: Study RAG users, document generation pipeline

## Links

- Related: feat/0009 (AI Client Enhancement — initial LOCAL_LLM_API_KEY support)
- References: `src/study_utils/core/ai.py` (`load_client`, `load_llama_swap_upstream_client`)
- UAT: `uat/test_client.py`, `uat/local_client_config.toml`

## History

### 2026-08-11 initial draft
**Summary** — Draft spec for per-service local LLM config integration.
**Changes**
- Defined `[services]` TOML section with `use_local`, `api_base`, and `provider` per service.
- Identified all 6 call sites across RAG, generate-document, quizzer.
- Decided on no changes to `load_client()` signature — pure env-var/config delegation.
- Set default of `use_local = true` for all services (local-first by default).
- Confirmed llama-swap upstream client is excluded from scope.

### 2026-08-11 update based on UAT
**Summary** — Updated defaults to match existing local_client_config.toml.
**Changes**
- Changed default `api_base` from `null` to `"http://localhost:8080/v1"` (matches UAT pattern).
- Added `provider = "local"` field mapping from UAT's `key_source`.
- Default chat model changed from `"gpt-4o-mini"` to `"gemma4-e4b"` (UAT default).
- Default embedding model changed from `"text-embedding-3-large"` to `"qwen3-embedding"`.

### 2026-08-11 final update
**Summary** — Finalized spec with all UAT-derived defaults confirmed.
**Changes**
- Updated `ServiceAIConfig` dataclass signature: Added `provider: Literal["local", "openai"]`, changed `api_base` default to `"http://localhost:8080/v1"`.
- Updated TOML template layout with correct model names (`gemma4-e4b` for chat, `qwen3-embedding` for embeddings), provider field, and api_base values.
- Updated `_DEFAULTS` dict entries with all new fields.
- Updated generate-document `[ai]` section: model to `"gemma4-e4b"`, added `provider`, api_base default.
- Updated quizzer extension: model to `"gemma4-e4b"`, added two new fields (`local_api_base`, `provider`).
- Moved History section inline to document evolution of decisions.

### 2026-08-11 transcription model upgrade
**Summary** — Made the transcription model configurable, changing default from `"whisper-1"` to `"whisper-3"`.
**Changes**
- Added `MODEL` key to `_TRANSCRIBE_LLM` dict in `transcribe_video.py`, with env var fallback `TRANSCRIPTION_MODEL`.
- Added `model = "whisper-3"` field to `[services.transcription]` subsection in the TOML template layout.
- Added transcription row to the caller wiring table.
- Updated `Definition of Done` checklist to include configurable model for transcription.
