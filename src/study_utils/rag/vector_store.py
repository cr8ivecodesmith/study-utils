"""Vector store management for the Study RAG workspace.

This module focuses on persistence concerns for retrieval databases during
Milestone 2. The ingestion pipeline orchestrates chunking/embedding and
ultimately relies on the repository + backend abstractions defined here to
persist data, emit manifests, and support lifecycle operations like export or
import.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Protocol

__all__ = [
    "VectorStoreError",
    "EmbeddingMetadata",
    "ChunkingMetadata",
    "DedupMetadata",
    "SourceDocument",
    "VectorStoreManifest",
    "VectorStoreBackend",
    "FaissVectorStoreBackend",
    "InMemoryVectorStoreBackend",
    "VectorStoreRepository",
]


_MANIFEST_FILENAME = "manifest.json"
_RECORDS_FILENAME = "records.json"
_FAISS_FILENAME = "index.faiss"

_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_MANIFEST_SCHEMA_VERSION = "1.0"


class VectorStoreError(RuntimeError):
    """Raised when vector store lifecycle operations fail."""


@dataclass(frozen=True)
class EmbeddingMetadata:
    provider: str
    model: str
    dimension: int


@dataclass(frozen=True)
class ChunkingMetadata:
    tokenizer: str
    encoding: str
    tokens_per_chunk: int
    token_overlap: int
    fallback_delimiter: str


@dataclass(frozen=True)
class DedupMetadata:
    strategy: str
    checksum_algorithm: str


@dataclass(frozen=True)
class SourceDocument:
    source_path: str
    checksum: str
    size_bytes: int
    chunk_count: int


@dataclass(frozen=True)
class VectorStoreManifest:
    name: str
    created_at: str
    updated_at: str
    schema_version: str
    embedding: EmbeddingMetadata
    chunking: ChunkingMetadata
    dedupe: DedupMetadata
    total_chunks: int
    documents: tuple[SourceDocument, ...]

    def to_dict(self) -> MutableMapping[str, Any]:
        data: MutableMapping[str, Any] = asdict(self)
        data["documents"] = [asdict(doc) for doc in self.documents]
        data["embedding"] = asdict(self.embedding)
        data["chunking"] = asdict(self.chunking)
        data["dedupe"] = asdict(self.dedupe)
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VectorStoreManifest":
        try:
            documents = tuple(
                SourceDocument(
                    source_path=str(item["source_path"]),
                    checksum=str(item["checksum"]),
                    size_bytes=int(item["size_bytes"]),
                    chunk_count=int(item["chunk_count"]),
                )
                for item in payload["documents"]
            )
            manifest = cls(
                name=str(payload["name"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                schema_version=str(payload.get("schema_version", "1.0")),
                embedding=EmbeddingMetadata(
                    provider=str(payload["embedding"]["provider"]),
                    model=str(payload["embedding"]["model"]),
                    dimension=int(payload["embedding"]["dimension"]),
                ),
                chunking=ChunkingMetadata(
                    tokenizer=str(payload["chunking"]["tokenizer"]),
                    encoding=str(payload["chunking"]["encoding"]),
                    tokens_per_chunk=int(
                        payload["chunking"]["tokens_per_chunk"]
                    ),
                    token_overlap=int(payload["chunking"]["token_overlap"]),
                    fallback_delimiter=str(
                        payload["chunking"].get("fallback_delimiter", "")
                    ),
                ),
                dedupe=DedupMetadata(
                    strategy=str(payload["dedupe"]["strategy"]),
                    checksum_algorithm=str(
                        payload["dedupe"]["checksum_algorithm"]
                    ),
                ),
                total_chunks=int(payload["total_chunks"]),
                documents=documents,
            )
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise VectorStoreError(
                "Manifest missing expected field: {0}".format(exc)
            ) from exc
        return manifest


class VectorStoreBackend(Protocol):
    """Persistence strategy for embedding vectors."""

    def create(
        self,
        store_path: Path,
        *,
        texts: Iterable[str],
        embeddings: Iterable[Iterable[float]],
        metadatas: Iterable[Mapping[str, Any]],
    ) -> None:
        """Persist embeddings and metadata to ``store_path``."""


class FaissVectorStoreBackend:
    """FAISS-backed persistence.

    The dependency is imported lazily so environments missing FAISS/numpy can
    still import the module and display actionable errors.
    """

    def create(
        self,
        store_path: Path,
        *,
        texts: Iterable[str],
        embeddings: Iterable[Iterable[float]],
        metadatas: Iterable[Mapping[str, Any]],
    ) -> None:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional dependency
            raise VectorStoreError(
                "The 'numpy' package is required for FAISS."
            ) from exc
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise VectorStoreError(
                "The 'faiss' package is required for FAISS."
            ) from exc

        vectors = list(embeddings)
        if not vectors:
            raise VectorStoreError("No embeddings provided for persistence.")
        texts_list = list(texts)
        metadata_list = list(metadatas)
        if not (len(texts_list) == len(vectors) == len(metadata_list)):
            raise VectorStoreError(
                "Embeddings, texts, and metadata collections must align."
            )

        dim = len(vectors[0])
        if dim <= 0:
            raise VectorStoreError("Embedding dimension must be positive.")
        for vector in vectors:
            if len(vector) != dim:
                raise VectorStoreError("Mixed embedding dimensions detected.")

        array = np.array(vectors, dtype="float32")
        index = faiss.IndexFlatL2(dim)
        index.add(array)

        store_path.mkdir(parents=True, exist_ok=True)
        _chmod_safe(store_path, 0o700)

        faiss.write_index(index, str(store_path / _FAISS_FILENAME))
        _atomic_write_json(
            store_path / _RECORDS_FILENAME,
            [
                {"text": text, "metadata": dict(meta)}
                for text, meta in zip(texts_list, metadata_list, strict=True)
            ],
        )


class InMemoryVectorStoreBackend:
    """Testing-oriented backend that writes embeddings to JSON only."""

    def create(
        self,
        store_path: Path,
        *,
        texts: Iterable[str],
        embeddings: Iterable[Iterable[float]],
        metadatas: Iterable[Mapping[str, Any]],
    ) -> None:
        texts_list = list(texts)
        embeddings_list = [list(vec) for vec in embeddings]
        metadata_list = [dict(meta) for meta in metadatas]
        if not (len(texts_list) == len(embeddings_list) == len(metadata_list)):
            raise VectorStoreError(
                "Embeddings, texts, and metadata collections must align."
            )
        store_path.mkdir(parents=True, exist_ok=True)
        _chmod_safe(store_path, 0o700)
        records = []
        for text, vector, meta in zip(
            texts_list, embeddings_list, metadata_list, strict=True
        ):
            records.append(
                {
                    "text": text,
                    "embedding": vector,
                    "metadata": meta,
                }
            )
        _atomic_write_json(store_path / _RECORDS_FILENAME, records)


class VectorStoreRepository:
    """Coordinate storage layout, manifests, and archival operations."""

    def __init__(self, root: Path):
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        _chmod_safe(self._root, 0o700)

    def store_path(self, name: str) -> Path:
        self._validate_name(name)
        return self._root / name

    def manifest_path(self, name: str) -> Path:
        return self.store_path(name) / _MANIFEST_FILENAME

    def list_manifests(self) -> list[VectorStoreManifest]:
        manifests: list[VectorStoreManifest] = []
        if not self._root.exists():
            return manifests
        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / _MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            try:
                manifests.append(self._load_manifest_path(manifest_path))
            except VectorStoreError:
                continue
        return manifests

    def prepare_store(self, name: str, *, overwrite: bool = False) -> Path:
        path = self.store_path(name)
        if path.exists():
            if not overwrite:
                raise VectorStoreError(
                    "Vector store '{0}' already exists. Use --force to "
                    "overwrite.".format(name)
                )
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        _chmod_safe(path, 0o700)
        return path

    def delete(self, name: str) -> None:
        path = self.store_path(name)
        if not path.exists():
            raise VectorStoreError(f"Vector store '{name}' does not exist.")
        shutil.rmtree(path)

    def load_manifest(self, name: str) -> VectorStoreManifest:
        path = self.manifest_path(name)
        if not path.exists():
            raise VectorStoreError(
                f"Manifest missing for vector store '{name}'."
            )
        return self._load_manifest_path(path)

    def write_manifest(self, manifest: VectorStoreManifest) -> Path:
        path = self.manifest_path(manifest.name)
        _atomic_write_json(path, manifest.to_dict())
        return path

    def export_store(self, name: str, destination: Path) -> Path:
        source_dir = self.store_path(name)
        if not source_dir.is_dir():
            raise VectorStoreError(
                f"Vector store '{name}' does not exist or is incomplete."
            )
        manifest_path = source_dir / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise VectorStoreError(
                f"Vector store '{name}' is missing its manifest."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for filename in (
                _MANIFEST_FILENAME,
                _RECORDS_FILENAME,
                _FAISS_FILENAME,
            ):
                file_path = source_dir / filename
                if file_path.exists():
                    zf.write(file_path, arcname=filename)
        return destination

    def import_store(
        self,
        name: str,
        archive: Path,
        *,
        overwrite: bool = False,
    ) -> VectorStoreManifest:
        if not archive.is_file():
            raise VectorStoreError(f"Archive not found: {archive}")
        destination = self.prepare_store(name, overwrite=overwrite)
        with zipfile.ZipFile(archive, "r") as zf:
            _safe_extract(zf, destination)
        manifest_path = destination / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise VectorStoreError(
                "Imported archive missing manifest.json; aborting load."
            )
        return self._load_manifest_path(manifest_path)

    @classmethod
    def _validate_name(cls, name: str) -> None:
        if not _NAME_PATTERN.fullmatch(name):
            raise VectorStoreError(
                "Vector store names must match [a-z0-9_-] and be 1-64 "
                "characters."
            )

    def _load_manifest_path(self, path: Path) -> VectorStoreManifest:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(
                f"Failed to parse manifest at {path}: {exc}"
            ) from exc
        manifest = VectorStoreManifest.from_dict(raw)
        if manifest.name != path.parent.name:
            raise VectorStoreError(
                "Manifest name mismatch with directory: "
                f"{manifest.name} vs {path.parent.name}"
            )
        return manifest


def build_manifest(
    *,
    name: str,
    embedding: EmbeddingMetadata,
    chunking: ChunkingMetadata,
    dedupe: DedupMetadata,
    documents: Iterable[SourceDocument],
) -> VectorStoreManifest:
    """Helper to construct a manifest with timestamps."""

    now = datetime.now(timezone.utc).isoformat()
    docs_tuple = tuple(documents)
    return VectorStoreManifest(
        name=name,
        created_at=now,
        updated_at=now,
        schema_version=_MANIFEST_SCHEMA_VERSION,
        embedding=embedding,
        chunking=chunking,
        dedupe=dedupe,
        total_chunks=sum(doc.chunk_count for doc in docs_tuple),
        documents=docs_tuple,
    )


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any] | list[Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=str(path.parent)
    )
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    _chmod_safe(path, 0o600)


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    for member in archive.infolist():
        member_name = member.filename
        if member_name.endswith("/"):
            continue
        target_path = (destination / member_name).resolve()
        if (
            destination not in target_path.parents
            and target_path != destination
        ):
            raise VectorStoreError(
                f"Archive member escapes destination: {member_name}"
            )
    archive.extractall(destination)
    for root, dirs, files in os.walk(destination):
        for dirname in dirs:
            _chmod_safe(Path(root) / dirname, 0o700)
        for filename in files:
            _chmod_safe(Path(root) / filename, 0o600)


def _chmod_safe(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except PermissionError:  # pragma: no cover - depends on filesystem
        pass
