"""Chat runtime orchestration for the Study RAG tool (Milestone 3)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

from rich.console import Console
from rich.panel import Panel

from . import config as config_mod
from . import ingest as ingest_mod
from . import session
from . import vector_store
from study_utils.core.ai import load_client as load_openai_client

__all__ = [
    "ChatError",
    "RetrievalResult",
    "ChatAnswer",
    "ChatClient",
    "OpenAIChatClient",
    "ChatRuntime",
]


_SYSTEM_PROMPT = (
    "You are Study, a retrieval-augmented tutor. Answer using the supplied "
    "context. If the context is insufficient, say so explicitly."
)


class ChatError(RuntimeError):
    """Raised when chat orchestration fails."""


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved context chunk supplied to the language model."""

    db_name: str
    text: str
    score: float
    metadata: Mapping[str, Any]

    def to_dict(self) -> MutableMapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "db_name": self.db_name,
            "text": self.text,
            "score": self.score,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ChatAnswer:
    """RAG response plus supporting contexts."""

    session_id: str
    prompt: str
    response: str
    contexts: Sequence[RetrievalResult]


class ChatClient(Protocol):
    """Protocol satisfied by model adapters."""

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        timeout: int,
    ) -> str:
        """Return an assistant response given the chat messages."""


class OpenAIChatClient:
    """Adapter for OpenAI chat completions."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        request_timeout: int,
        api_base: str | None,
        use_local: bool = False,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._timeout = request_timeout
        if client is not None:
            self._client = client
        else:
            self._client = load_openai_client(
                local=use_local,
                api_base=api_base,
            )

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        timeout: int,
    ) -> str:
        del stream  # Streaming handled at CLI layer for now.
        if model != self._model:
            # Reconfigure on the fly to honour per-session overrides.
            self._model = model
        if timeout != self._timeout:
            self._timeout = timeout
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[dict(msg) for msg in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        return content.strip()


class _EmbeddingRecord:
    __slots__ = ("text", "metadata", "embedding")

    def __init__(
        self,
        *,
        text: str,
        metadata: Mapping[str, Any],
        embedding: Sequence[float] | None,
    ) -> None:
        self.text = text
        self.metadata = dict(metadata)
        self.embedding = list(embedding) if embedding is not None else None


class _RetrievalIndex:
    def __init__(
        self,
        *,
        name: str,
        manifest: vector_store.VectorStoreManifest,
        records: Sequence[_EmbeddingRecord],
        store_path: Path,
    ) -> None:
        self._name = name
        self._manifest = manifest
        self._records = list(records)
        self._faiss_index = None
        self._numpy = None
        self._store_path = store_path
        self._load_backend_if_needed()

    @property
    def embedding_dimension(self) -> int:
        return self._manifest.embedding.dimension

    def _load_backend_if_needed(self) -> None:
        if any(record.embedding is not None for record in self._records):
            return
        faiss_path = self._store_path / "index.faiss"
        if not faiss_path.is_file():
            raise ChatError(
                f"Vector store '{self._manifest.name}' missing embeddings "
                "and index."
            )
        try:
            import faiss  # type: ignore
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional dep
            raise ChatError(
                "FAISS runtime required to query this vector store. Install "
                "faiss-cpu and numpy."
            ) from exc
        self._faiss_index = faiss.read_index(str(faiss_path))
        self._numpy = np

    def search(
        self,
        query: Sequence[float],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        if self._faiss_index is not None:
            return self._search_faiss(query, top_k=top_k)
        return self._search_memory(query, top_k=top_k)

    def _search_memory(
        self,
        query: Sequence[float],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        if not self._records:
            return []
        scores: list[tuple[float, _EmbeddingRecord]] = []
        for record in self._records:
            if record.embedding is None:
                continue
            score = _cosine_similarity(query, record.embedding)
            scores.append((score, record))
        scores.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for score, record in scores[:top_k]:
            results.append(
                RetrievalResult(
                    db_name=self._name,
                    text=record.text,
                    score=score,
                    metadata=record.metadata,
                )
            )
        return results

    def _search_faiss(
        self,
        query: Sequence[float],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:  # pragma: no cover - requires faiss at runtime
        np = self._numpy
        if np is None:  # pragma: no cover - defensive
            raise ChatError("NumPy runtime missing for FAISS search.")
        assert self._faiss_index is not None
        array = np.array(query, dtype="float32").reshape(1, -1)
        distances, indices = self._faiss_index.search(array, top_k)
        results: list[RetrievalResult] = []
        for idx, distance in zip(indices[0], distances[0], strict=True):
            if idx < 0 or idx >= len(self._records):
                continue
            record = self._records[idx]
            score = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievalResult(
                    db_name=self._name,
                    text=record.text,
                    score=score,
                    metadata=record.metadata,
                )
            )
        return results


class _Retriever:
    """Aggregate retrieval across one or more vector stores."""

    def __init__(
        self,
        *,
        repository: vector_store.VectorStoreRepository,
        embedder: ingest_mod.EmbeddingClient,
        retrieval_config: config_mod.RetrievalConfig,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._retrieval = retrieval_config

    def load_indexes(self, names: Sequence[str]) -> list[_RetrievalIndex]:
        indexes: list[_RetrievalIndex] = []
        for name in names:
            manifest = self._repository.load_manifest(name)
            store_path = self._repository.store_path(name)
            records = _load_records(store_path)
            indexes.append(
                _RetrievalIndex(
                    name=name,
                    manifest=manifest,
                    records=records,
                    store_path=store_path,
                )
            )
        return indexes

    def retrieve(
        self,
        *,
        question: str,
        indexes: Sequence[_RetrievalIndex],
    ) -> tuple[list[RetrievalResult], Sequence[float]]:
        embeddings = self._embedder.embed_documents([question])
        if not embeddings:
            raise ChatError("Embedding provider returned no vector for query.")
        query_vector = list(embeddings[0])
        dimension = len(query_vector)
        for index in indexes:
            if dimension != index.embedding_dimension:
                raise ChatError(
                    f"Embedding dimension mismatch for vector store "
                    f"'{index._name}'."
                )
        top_k = self._retrieval.top_k
        aggregated: list[RetrievalResult] = []
        for index in indexes:
            aggregated.extend(index.search(query_vector, top_k=top_k))
        aggregated.sort(key=lambda item: item.score, reverse=True)
        filtered: list[RetrievalResult] = []
        threshold = self._retrieval.score_threshold
        for item in aggregated:
            if item.score < threshold:
                continue
            filtered.append(item)
            if len(filtered) >= top_k:
                break
        return filtered, query_vector


def _load_records(store_path: Path) -> list[_EmbeddingRecord]:
    records_path = store_path / "records.json"
    if not records_path.is_file():
        raise ChatError(f"Vector store records missing: {records_path}")
    raw = records_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ChatError(
            f"Failed to parse records file: {records_path}"
        ) from exc
    if not isinstance(payload, list):
        raise ChatError(
            f"Unexpected records structure in {records_path}; expected a list."
        )
    records: list[_EmbeddingRecord] = []
    for entry in payload:
        if not isinstance(entry, Mapping):  # pragma: no cover - defensive
            continue
        text = str(entry.get("text", ""))
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        embedding_raw = entry.get("embedding")
        embedding: Sequence[float] | None
        if embedding_raw is None:
            embedding = None
        else:
            embedding = [float(value) for value in embedding_raw]
        records.append(
            _EmbeddingRecord(
                text=text,
                metadata=metadata,
                embedding=embedding,
            )
        )
    return records


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):  # pragma: no cover - defensive
        raise ChatError("Embedding vectors must share the same dimension.")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class ChatRuntime:
    """High-level chat orchestration used by the CLI."""

    def __init__(
        self,
        *,
        config: config_mod.RagConfig,
        repository: vector_store.VectorStoreRepository,
        session_store: session.SessionStore,
        embedder: ingest_mod.EmbeddingClient,
        chat_client: ChatClient,
    ) -> None:
        self._config = config
        self._repository = repository
        self._session_store = session_store
        self._retriever = _Retriever(
            repository=repository,
            embedder=embedder,
            retrieval_config=config.retrieval,
        )
        self._chat_client = chat_client

    def prepare_session(
        self,
        *,
        resume_id: str | None,
        vector_dbs: Sequence[str],
    ) -> session.ChatSession:
        if resume_id:
            sess = self._session_store.load(resume_id)
            self._ensure_vector_dbs(sess, additional=tuple(vector_dbs))
            if vector_dbs:
                sess.merge_vector_dbs(tuple(vector_dbs))
                self._session_store.save(sess)
            return sess
        if not vector_dbs:
            raise ChatError(
                "At least one --db must be provided when starting a session."
            )
        manifests = [
            self._repository.load_manifest(name) for name in vector_dbs
        ]
        embedding_provider = manifests[0].embedding.provider
        embedding_model = manifests[0].embedding.model
        embedding_dimension = manifests[0].embedding.dimension
        if embedding_provider.lower() != self._config.providers.default.lower():
            raise ChatError(
                "Selected vector stores were built with a different provider "
                "than the configured default."
            )
        for manifest in manifests[1:]:
            if manifest.embedding.provider != embedding_provider:
                raise ChatError(
                    "Selected vector stores use different embedding providers."
                )
            if manifest.embedding.model != embedding_model:
                raise ChatError(
                    "Selected vector stores use different embedding models."
                )
            if manifest.embedding.dimension != embedding_dimension:
                raise ChatError(
                    "Selected vector stores have mismatched dimensions."
                )
        sess = self._session_store.create(
            vector_dbs=tuple(vector_dbs),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chat_model=self._config.providers.openai.chat_model,
        )
        return sess

    def ask(
        self,
        sess: session.ChatSession,
        prompt: str,
    ) -> ChatAnswer:
        question = prompt.strip()
        if not question:
            raise ChatError("Prompt cannot be empty.")
        self._ensure_vector_dbs(sess)
        indexes = self._retriever.load_indexes(sess.vector_dbs)
        contexts, query_vector = self._retriever.retrieve(
            question=question,
            indexes=indexes,
        )
        assembled_messages = self._assemble_messages(sess, question, contexts)
        provider = self._config.providers.openai
        response_text = self._chat_client.complete(
            model=sess.chat_model,
            messages=assembled_messages,
            temperature=provider.temperature,
            max_tokens=self._config.chat.response_tokens,
            stream=self._config.chat.stream,
            timeout=provider.request_timeout_seconds,
        )
        sess.add_message(
            "user",
            question,
            metadata={
                "retrieval": [item.to_dict() for item in contexts],
                "query_vector_dim": len(query_vector),
            },
        )
        sess.add_message(
            "assistant",
            response_text,
            metadata={"retrieval": [item.to_dict() for item in contexts]},
        )
        max_messages = max(1, self._config.chat.max_history_turns * 2)
        sess.enforce_history_limit(max_messages)
        self._session_store.save(sess)
        return ChatAnswer(
            session_id=sess.session_id,
            prompt=question,
            response=response_text,
            contexts=contexts,
        )

    def interactive_loop(
        self,
        sess: session.ChatSession,
        *,
        console: Console,
    ) -> None:
        console.print(
            Panel(
                (
                    f"Session [bold]{sess.session_id}[/] ready. Connected to "
                    f"{', '.join(sess.vector_dbs)}."
                ),
                title="Study RAG Chat",
            )
        )
        while True:
            try:
                prompt = console.input("[bold green]You[/]> ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting session.")
                break
            prompt = prompt.strip()
            if prompt in {":quit", ":q", "exit"}:
                console.print("Goodbye!")
                break
            if not prompt:
                continue
            try:
                answer = self.ask(sess, prompt)
            except ChatError as exc:
                console.print(f"[red]Error:[/] {exc}")
                continue
            self._render_answer(console, answer)

    def _render_answer(self, console: Console, answer: ChatAnswer) -> None:
        console.print(Panel(answer.response, title="Assistant"))
        if not answer.contexts:
            console.print(
                "[dim]No retrieval context met the score threshold.[/]"
            )
            return
        for idx, item in enumerate(answer.contexts, start=1):
            source = item.metadata.get("source_path", "unknown")
            console.print(
                Panel(
                    f"Score {item.score:.3f} — {source}\n\n{item.text}",
                    title=f"Context {idx} ({item.db_name})",
                    subtitle=str(item.metadata.get("chunk_index", "")),
                )
            )

    def _assemble_messages(
        self,
        sess: session.ChatSession,
        question: str,
        contexts: Sequence[RetrievalResult],
    ) -> list[Mapping[str, str]]:
        history: list[Mapping[str, str]] = []
        for message in sess.messages:
            history.append({"role": message.role, "content": message.content})
        max_chars = max(0, self._config.retrieval.max_context_tokens * 4)
        remaining = max_chars if max_chars else None
        formatted: list[str] = []
        for idx, item in enumerate(contexts, start=1):
            chunk = _format_context(idx, item)
            if remaining is not None:
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                remaining -= len(chunk)
            formatted.append(chunk)
        context_block = "\n\n".join(formatted)
        prompt_content = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_block or 'None'}\n\n"
            f"Question:\n{question}"
        )
        history.append({"role": "user", "content": prompt_content})
        return history

    def _ensure_vector_dbs(
        self,
        sess: session.ChatSession,
        *,
        additional: tuple[str, ...] | None = None,
    ) -> None:
        names = set(sess.vector_dbs)
        if additional:
            names.update(additional)
        for name in names:
            manifest = self._repository.load_manifest(name)
            if manifest.embedding.provider != sess.embedding_provider:
                raise ChatError(
                    f"Vector store '{name}' provider mismatch with session."
                )
            if manifest.embedding.model != sess.embedding_model:
                raise ChatError(
                    f"Vector store '{name}' embedding model mismatch "
                    "with session."
                )
            if manifest.embedding.dimension != sess.embedding_dimension:
                raise ChatError(
                    f"Vector store '{name}' dimension differs from session."
                )


def _format_context(idx: int, result: RetrievalResult) -> str:
    source = result.metadata.get("source_path", "unknown")
    chunk = result.metadata.get("chunk_index")
    header = f"Context {idx} — DB: {result.db_name} — Score: {result.score:.3f}"
    tail = f"Source: {source}"
    if chunk is not None:
        tail += f" (chunk {chunk})"
    return f"{header}\n{tail}\n{result.text}"
