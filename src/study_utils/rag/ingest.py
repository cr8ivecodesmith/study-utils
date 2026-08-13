"""Ingestion pipeline for Study RAG vector databases (Milestone 2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

from . import vector_store

try:  # Optional dependency resolved lazily for chunking.
    import tiktoken
except Exception:  # pragma: no cover - fallback when tiktoken missing
    tiktoken = None  # type: ignore

try:  # Optional dependency for OpenAI embeddings client.
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    OpenAI = None  # type: ignore

from study_utils.core.ai import load_client as load_openai_client

__all__ = [
    "EmbeddingClient",
    "OpenAIEmbeddingClient",
    "TextChunker",
    "IngestionReport",
    "ingest_sources",
]


class EmbeddingClient(Protocol):
    """Protocol satisfied by embedding providers."""

    def embed_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        """Return embedding vectors for each text chunk."""


class OpenAIEmbeddingClient:
    """Adapter around the OpenAI embeddings API."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None,
        request_timeout: int,
        use_local: bool = False,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = _build_openai_client(
                api_base=api_base,
                use_local=use_local,
            )
        self._model = model
        self._timeout = request_timeout

    def embed_documents(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        if OpenAI is None:
            raise RuntimeError(
                "The openai package is required to create embeddings."
            )
        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            timeout=self._timeout,
        )
        return [list(item.embedding) for item in response.data]


def _build_openai_client(
    *,
    api_base: str | None,
    use_local: bool = False,
) -> Any:
    client = load_openai_client(local=use_local, api_base=api_base)
    return client


@dataclass(frozen=True)
class IngestionReport:
    name: str
    documents_ingested: int
    documents_skipped: int
    chunks_ingested: int
    manifest: vector_store.VectorStoreManifest


class TextChunker:
    """Token-aware text chunker with tiktoken + fallback strategy."""

    def __init__(
        self,
        *,
        tokenizer: str,
        encoding: str,
        tokens_per_chunk: int,
        token_overlap: int,
        fallback_delimiter: str,
    ) -> None:
        if tokens_per_chunk <= 0:
            raise ValueError("tokens_per_chunk must be positive")
        if token_overlap < 0:
            raise ValueError("token_overlap must be non-negative")
        if token_overlap >= tokens_per_chunk:
            raise ValueError(
                "token_overlap must be smaller than tokens_per_chunk"
            )
        self._tokens_per_chunk = tokens_per_chunk
        self._token_overlap = token_overlap
        self._fallback_delimiter = fallback_delimiter or "\n\n"
        self._encoder = _load_encoder(tokenizer=tokenizer, encoding=encoding)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        if self._encoder is not None:
            return self._chunk_with_encoder(text)
        return self._chunk_with_fallback(text)

    def _chunk_with_encoder(self, text: str) -> list[str]:
        encoder = self._encoder
        assert encoder is not None
        tokens = encoder.encode(text)
        if not tokens:
            return []
        chunks: list[str] = []
        stride = self._tokens_per_chunk - self._token_overlap
        for start in range(0, len(tokens), stride):
            window = tokens[start : start + self._tokens_per_chunk]
            if not window:
                continue
            chunk_text = encoder.decode(window)
            if chunk_text.strip():
                chunks.append(chunk_text)
        return chunks

    def _chunk_with_fallback(self, text: str) -> list[str]:
        chunk_chars = max(self._tokens_per_chunk * 4, 1)
        overlap_chars = max(self._token_overlap * 4, 0)
        clean = text.strip()
        if not clean:
            return []
        chunks: list[str] = []
        step = max(chunk_chars - overlap_chars, 1)
        for start in range(0, len(clean), step):
            window = clean[start : start + chunk_chars]
            if window.strip():
                chunks.append(window)
        return chunks


def _load_encoder(*, tokenizer: str, encoding: str):
    if tokenizer.lower() != "tiktoken":
        return None
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(encoding)
    except Exception:  # pragma: no cover - dependent on runtime assets
        return None


def ingest_sources(
    name: str,
    *,
    inputs: Sequence[Path],
    repository: vector_store.VectorStoreRepository,
    backend: vector_store.VectorStoreBackend,
    embedder: EmbeddingClient,
    chunker: TextChunker,
    embedding_provider: str,
    embedding_model: str,
    dedupe: vector_store.DedupMetadata,
    chunking: vector_store.ChunkingMetadata,
    overwrite: bool = False,
) -> IngestionReport:
    if not inputs:
        raise vector_store.VectorStoreError("No input paths supplied.")

    repository.ensure_root()
    store_dir = repository.prepare_store(name, overwrite=overwrite)

    files = list(_discover_files(inputs))
    if not files:
        raise vector_store.VectorStoreError(
            "No files discovered for ingestion."
        )

    texts, metadatas, documents, skipped = _collect_chunks(
        files,
        chunker,
        dedupe,
    )
    if not texts:
        raise vector_store.VectorStoreError(
            "No textual chunks produced; ensure sources contain text."
        )

    embeddings = list(embedder.embed_documents(texts))
    dimension = _validate_embeddings(texts, embeddings)

    backend.create(
        store_dir,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    manifest = vector_store.build_manifest(
        name=name,
        embedding=vector_store.EmbeddingMetadata(
            provider=embedding_provider,
            model=embedding_model,
            dimension=dimension,
        ),
        chunking=chunking,
        dedupe=dedupe,
        documents=documents,
    )
    repository.write_manifest(manifest)

    return IngestionReport(
        name=name,
        documents_ingested=len(documents),
        documents_skipped=skipped,
        chunks_ingested=len(texts),
        manifest=manifest,
    )


def _collect_chunks(
    files: Sequence[Path],
    chunker: TextChunker,
    dedupe: vector_store.DedupMetadata,
) -> tuple[
    list[str],
    list[dict[str, Any]],
    list[vector_store.SourceDocument],
    int,
]:
    checksum_seen: set[str] = set()
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    documents: list[vector_store.SourceDocument] = []
    skipped = 0

    for file_path in files:
        checksum = _compute_checksum(file_path, dedupe.checksum_algorithm)
        if checksum in checksum_seen:
            skipped += 1
            continue
        checksum_seen.add(checksum)
        text = _read_text(file_path)
        chunks = chunker.chunk(text)
        if not chunks:
            skipped += 1
            continue
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source_path": str(file_path),
                    "chunk_index": idx,
                    "checksum": checksum,
                }
            )
        size_bytes = file_path.stat().st_size
        documents.append(
            vector_store.SourceDocument(
                source_path=str(file_path),
                checksum=checksum,
                size_bytes=int(size_bytes),
                chunk_count=len(chunks),
            )
        )

    return texts, metadatas, documents, skipped


def _validate_embeddings(
    texts: Sequence[str], embeddings: Sequence[Sequence[float]]
) -> int:
    if len(embeddings) != len(texts):
        raise vector_store.VectorStoreError(
            "Embedding provider returned unexpected vector count."
        )
    if not embeddings:
        raise vector_store.VectorStoreError(
            "Embedding provider returned nothing."
        )
    dimension = len(embeddings[0])
    if dimension <= 0:
        raise vector_store.VectorStoreError(
            "Embedding provider returned vectors with invalid dimension."
        )
    for vector in embeddings:
        if len(vector) != dimension:
            raise vector_store.VectorStoreError(
                "Embedding provider returned mixed vector dimensions."
            )
    return dimension


def _discover_files(inputs: Sequence[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path in seen:
            continue
        if not path.exists():
            raise vector_store.VectorStoreError(f"Input path not found: {path}")
        if path.is_file():
            seen.add(path)
            yield path
        else:
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved


def _compute_checksum(path: Path, algorithm: str) -> str:
    algo = algorithm.lower()
    if algo != "sha256":
        raise vector_store.VectorStoreError(
            f"Unsupported checksum algorithm '{algorithm}'."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
