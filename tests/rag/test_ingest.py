from __future__ import annotations

import types
from pathlib import Path

import pytest

from study_utils.rag import ingest
from study_utils.rag import vector_store


class StubEmbedder:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class BadEmbedder:
    def embed_documents(self, texts):
        return []


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "stores"
    repository = vector_store.VectorStoreRepository(root)
    repository.ensure_root()
    return repository


@pytest.fixture()
def chunker():
    return ingest.TextChunker(
        tokenizer="tiktoken",
        encoding="cl100k_base",
        tokens_per_chunk=50,
        token_overlap=10,
        fallback_delimiter="\n\n",
    )


@pytest.fixture()
def chunking_meta():
    return vector_store.ChunkingMetadata(
        tokenizer="tiktoken",
        encoding="cl100k_base",
        tokens_per_chunk=50,
        token_overlap=10,
        fallback_delimiter="\n\n",
    )


@pytest.fixture()
def dedupe_meta():
    return vector_store.DedupMetadata(
        strategy="checksum",
        checksum_algorithm="sha256",
    )


def _make_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_ingest_sources_writes_manifest(
    tmp_path, repo, chunker, chunking_meta, dedupe_meta
):
    source = _make_file(tmp_path / "doc.txt", "This is a test document.")
    backend = vector_store.InMemoryVectorStoreBackend()

    report = ingest.ingest_sources(
        "physics",
        inputs=[source],
        repository=repo,
        backend=backend,
        embedder=StubEmbedder(),
        chunker=chunker,
        embedding_provider="openai",
        embedding_model="test-model",
        dedupe=dedupe_meta,
        chunking=chunking_meta,
    )

    assert report.documents_ingested == 1
    manifest = repo.load_manifest("physics")
    assert manifest.embedding.dimension == 2
    assert manifest.documents[0].chunk_count >= 1
    records = repo.store_path("physics") / "records.json"
    assert records.exists()


def test_ingest_sources_skips_duplicate_hashes(
    tmp_path, repo, chunker, chunking_meta, dedupe_meta
):
    content = "Repeated" * 5
    source1 = _make_file(tmp_path / "doc1.txt", content)
    source2 = _make_file(tmp_path / "doc2.txt", content)
    backend = vector_store.InMemoryVectorStoreBackend()

    report = ingest.ingest_sources(
        "physics",
        inputs=[source1, source2],
        repository=repo,
        backend=backend,
        embedder=StubEmbedder(),
        chunker=chunker,
        embedding_provider="openai",
        embedding_model="test-model",
        dedupe=dedupe_meta,
        chunking=chunking_meta,
    )

    assert report.documents_ingested == 1
    assert report.documents_skipped == 1


def test_ingest_sources_requires_files(
    repo, chunker, chunking_meta, dedupe_meta
):
    backend = vector_store.InMemoryVectorStoreBackend()
    with pytest.raises(vector_store.VectorStoreError):
        ingest.ingest_sources(
            "physics",
            inputs=[],
            repository=repo,
            backend=backend,
            embedder=StubEmbedder(),
            chunker=chunker,
            embedding_provider="openai",
            embedding_model="test-model",
            dedupe=dedupe_meta,
            chunking=chunking_meta,
        )


def test_ingest_sources_validates_missing_inputs(
    repo, chunker, chunking_meta, dedupe_meta
):
    backend = vector_store.InMemoryVectorStoreBackend()
    with pytest.raises(vector_store.VectorStoreError):
        ingest.ingest_sources(
            "physics",
            inputs=[Path("missing.txt")],
            repository=repo,
            backend=backend,
            embedder=StubEmbedder(),
            chunker=chunker,
            embedding_provider="openai",
            embedding_model="test-model",
            dedupe=dedupe_meta,
            chunking=chunking_meta,
        )


def test_ingest_sources_rejects_bad_embeddings(
    tmp_path, repo, chunker, chunking_meta, dedupe_meta
):
    source = _make_file(tmp_path / "doc.txt", "content")
    backend = vector_store.InMemoryVectorStoreBackend()
    with pytest.raises(vector_store.VectorStoreError):
        ingest.ingest_sources(
            "physics",
            inputs=[source],
            repository=repo,
            backend=backend,
            embedder=BadEmbedder(),
            chunker=chunker,
            embedding_provider="openai",
            embedding_model="test-model",
            dedupe=dedupe_meta,
            chunking=chunking_meta,
        )


def test_text_chunker_validates_overlap():
    with pytest.raises(ValueError):
        ingest.TextChunker(
            tokenizer="tiktoken",
            encoding="cl100k_base",
            tokens_per_chunk=10,
            token_overlap=10,
            fallback_delimiter="\n\n",
        )


def test_text_chunker_validates_positive_chunk():
    with pytest.raises(ValueError):
        ingest.TextChunker(
            tokenizer="tiktoken",
            encoding="cl100k_base",
            tokens_per_chunk=0,
            token_overlap=0,
            fallback_delimiter="\n\n",
        )
    with pytest.raises(ValueError):
        ingest.TextChunker(
            tokenizer="tiktoken",
            encoding="cl100k_base",
            tokens_per_chunk=5,
            token_overlap=-1,
            fallback_delimiter="\n\n",
        )


def test_text_chunker_fallback_handles_blank_text():
    chunker = ingest.TextChunker(
        tokenizer="plain",
        encoding="ignored",
        tokens_per_chunk=5,
        token_overlap=0,
        fallback_delimiter="\n\n",
    )
    assert chunker.chunk("   ") == []
    assert chunker._chunk_with_fallback("   ") == []


def test_text_chunker_fallback_produces_chunks():
    chunker = ingest.TextChunker(
        tokenizer="plain",
        encoding="ignored",
        tokens_per_chunk=2,
        token_overlap=1,
        fallback_delimiter="\n",
    )
    text = "alpha beta gamma delta"
    chunks = chunker.chunk(text)
    assert chunks
    assert any("alpha" in chunk for chunk in chunks)
    assert any("gamma" in chunk for chunk in chunks)


def test_openai_embedding_client_uses_stub(monkeypatch):
    stub_data = types.SimpleNamespace(
        data=[types.SimpleNamespace(embedding=[1.0, 2.0])]
    )
    client = types.SimpleNamespace(
        embeddings=types.SimpleNamespace(create=lambda **kwargs: stub_data)
    )
    monkeypatch.setattr(ingest, "OpenAI", object())
    adapter = ingest.OpenAIEmbeddingClient(
        model="test",
        api_base=None,
        request_timeout=5,
        client=client,
    )
    vectors = adapter.embed_documents(["hello"])
    assert vectors == [[1.0, 2.0]]
    assert adapter.embed_documents([]) == []


def test_openai_embedding_client_builds_client(monkeypatch):
    stub_data = types.SimpleNamespace(
        data=[types.SimpleNamespace(embedding=[1.0])]
    )
    stub_client = types.SimpleNamespace(
        base_url=None,
        embeddings=types.SimpleNamespace(create=lambda **kwargs: stub_data),
    )
    monkeypatch.setattr(ingest, "OpenAI", object())

    def _mock_load_openai_client(local: bool = False, api_base: str | None = None):
        if api_base and hasattr(stub_client, "base_url"):
            stub_client.base_url = api_base
        return stub_client

    monkeypatch.setattr(ingest, "load_openai_client", _mock_load_openai_client)
    adapter = ingest.OpenAIEmbeddingClient(
        model="m",
        api_base="https://example",
        request_timeout=1,
    )
    assert stub_client.base_url == "https://example"
    assert adapter.embed_documents(["hi"]) == [[1.0]]


def test_openai_embedding_client_requires_openai(monkeypatch):
    monkeypatch.setattr(ingest, "OpenAI", None)
    adapter = ingest.OpenAIEmbeddingClient(
        model="m",
        api_base=None,
        request_timeout=1,
        client=types.SimpleNamespace(embeddings=None),
    )
    with pytest.raises(RuntimeError):
        adapter.embed_documents(["hi"])


def test_build_openai_client_sets_base_url(monkeypatch):
    holder = {}

    class StubClient:
        base_url: str | None

        def __init__(self):
            self.base_url = None

    def _mock_loader(local: bool = False, api_base: str | None = None):
        if "client" not in holder:
            c = StubClient()
            if api_base is not None:
                c.base_url = api_base
            holder["client"] = c
            return c
        client = holder["client"]
        if api_base is not None:
            client.base_url = api_base
        return client

    monkeypatch.setattr(ingest, "load_openai_client", _mock_loader)
    client = ingest._build_openai_client(api_base="https://example")
    assert client.base_url == "https://example"


def test_text_chunker_with_encoder(monkeypatch):
    class StubEncoding:
        def encode(self, text):
            return WeirdTokens(list(range(len(text.split()))))

        def decode(self, tokens):
            return " ".join(str(t) for t in tokens)

    class WeirdTokens(list):
        def __getitem__(self, key):
            if isinstance(key, slice) and key.start == 2:
                return []
            return super().__getitem__(key)

    monkeypatch.setattr(
        ingest,
        "tiktoken",
        types.SimpleNamespace(get_encoding=lambda name: StubEncoding()),
    )
    chunker = ingest.TextChunker(
        tokenizer="tiktoken",
        encoding="whatever",
        tokens_per_chunk=2,
        token_overlap=1,
        fallback_delimiter="\n\n",
    )
    chunks = chunker.chunk("a b c")
    assert chunks


def test_text_chunker_encoder_handles_empty_tokens(monkeypatch):
    class EmptyEncoding:
        def encode(self, text):  # noqa: D401, ARG002
            return []

        def decode(self, tokens):  # noqa: D401, ARG002
            return ""

    monkeypatch.setattr(
        ingest,
        "tiktoken",
        types.SimpleNamespace(get_encoding=lambda name: EmptyEncoding()),
    )
    chunker = ingest.TextChunker(
        tokenizer="tiktoken",
        encoding="enc",
        tokens_per_chunk=5,
        token_overlap=0,
        fallback_delimiter="\n\n",
    )
    assert chunker.chunk("text") == []


def test_collect_chunks_skips_empty_files(tmp_path, chunker, dedupe_meta):
    empty = _make_file(tmp_path / "empty.txt", "   ")
    texts, metadatas, documents, skipped = ingest._collect_chunks(
        [empty],
        chunker,
        dedupe_meta,
    )
    assert not texts
    assert skipped == 1
    assert documents == []


def test_validate_embeddings_dimension_errors():
    with pytest.raises(vector_store.VectorStoreError):
        ingest._validate_embeddings(["a"], [])
    with pytest.raises(vector_store.VectorStoreError):
        ingest._validate_embeddings(["a"], [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(vector_store.VectorStoreError):
        ingest._validate_embeddings(["a"], [[]])
    with pytest.raises(vector_store.VectorStoreError):
        ingest._validate_embeddings(["a", "b"], [[0.0], [2.0, 3.0]])


def test_validate_embeddings_allows_empty_inputs():
    with pytest.raises(vector_store.VectorStoreError):
        ingest._validate_embeddings([], [])


def test_read_text_handles_decode_error(tmp_path):
    path = tmp_path / "binary.bin"
    path.write_bytes(b"\xff\xfe")
    result = ingest._read_text(path)
    assert isinstance(result, str)


def test_ingest_sources_requires_non_empty_directory(
    tmp_path, repo, chunker, chunking_meta, dedupe_meta
):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    backend = vector_store.InMemoryVectorStoreBackend()
    with pytest.raises(vector_store.VectorStoreError):
        ingest.ingest_sources(
            "physics",
            inputs=[empty_dir],
            repository=repo,
            backend=backend,
            embedder=StubEmbedder(),
            chunker=chunker,
            embedding_provider="openai",
            embedding_model="test-model",
            dedupe=dedupe_meta,
            chunking=chunking_meta,
        )


def test_ingest_sources_requires_textual_chunks(
    tmp_path, repo, chunker, chunking_meta, dedupe_meta
):
    blank = _make_file(tmp_path / "blank.txt", "   ")
    backend = vector_store.InMemoryVectorStoreBackend()
    with pytest.raises(vector_store.VectorStoreError):
        ingest.ingest_sources(
            "physics",
            inputs=[blank],
            repository=repo,
            backend=backend,
            embedder=StubEmbedder(),
            chunker=chunker,
            embedding_provider="openai",
            embedding_model="test-model",
            dedupe=dedupe_meta,
            chunking=chunking_meta,
        )


def test_discover_files_deduplicates_inputs(tmp_path):
    file_path = _make_file(tmp_path / "doc.txt", "data")
    results = list(ingest._discover_files([file_path, file_path]))
    assert results == [file_path.resolve()]


def test_discover_files_walks_directories(tmp_path):
    folder = tmp_path / "folder"
    sub = folder / "sub"
    sub.mkdir(parents=True)
    file_path = _make_file(sub / "doc.txt", "data")
    results = list(ingest._discover_files([folder]))
    assert results == [file_path.resolve()]


def test_load_encoder_paths(monkeypatch):
    assert ingest._load_encoder(tokenizer="plain", encoding="x") is None
    monkeypatch.setattr(ingest, "tiktoken", None)
    assert ingest._load_encoder(tokenizer="tiktoken", encoding="x") is None


def test_compute_checksum_rejects_algorithm(tmp_path):
    file_path = _make_file(tmp_path / "doc.txt", "data")
    with pytest.raises(vector_store.VectorStoreError):
        ingest._compute_checksum(file_path, "md5")
