from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest

from study_utils.rag import chat as chat_mod
from study_utils.rag import config as config_mod
from study_utils.rag import data_dir
from study_utils.rag import ingest as ingest_mod
from study_utils.rag import session as session_mod
from study_utils.rag import vector_store
from study_utils.rag import cli as rag_cli


class StubEmbeddingClient:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class StubChatClient:
    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        *,
        model,
        messages,
        temperature,
        max_tokens,
        stream,
        timeout,
    ) -> str:
        self.calls.append([dict(item) for item in messages])
        last = messages[-1]["content"].splitlines()[-1]
        return f"Stub answer: {last}"


class StubConsole:
    def __init__(self, prompts):
        self._prompts = iter(prompts)
        self.outputs: list[object] = []

    def print(self, message):  # noqa: D401
        self.outputs.append(message)

    def input(self, prompt):  # noqa: D401
        self.outputs.append(prompt)
        return next(self._prompts)


def make_manifest(
    *,
    name: str = "notes",
    provider: str = "openai",
    model: str = "text-embedding-3-large",
    dimension: int = 2,
) -> vector_store.VectorStoreManifest:
    timestamp = "2024-01-01T00:00:00Z"
    embedding = vector_store.EmbeddingMetadata(
        provider=provider,
        model=model,
        dimension=dimension,
    )
    chunking = vector_store.ChunkingMetadata(
        tokenizer="cl100k",
        encoding="utf-8",
        tokens_per_chunk=128,
        token_overlap=16,
        fallback_delimiter="\n",
    )
    dedupe = vector_store.DedupMetadata(
        strategy="hash",
        checksum_algorithm="sha256",
    )
    document = vector_store.SourceDocument(
        source_path="doc.md",
        checksum="abc",
        size_bytes=123,
        chunk_count=1,
    )
    return vector_store.VectorStoreManifest(
        name=name,
        created_at=timestamp,
        updated_at=timestamp,
        schema_version="1.0",
        embedding=embedding,
        chunking=chunking,
        dedupe=dedupe,
        total_chunks=1,
        documents=(document,),
    )


@pytest.fixture()
def runtime_env(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    monkeypatch.setenv("STUDY_UTILS_DATA_HOME", str(data_home))
    config_mod.write_template(data_home / "config" / "rag.toml")
    cfg = config_mod.load_config()
    repo = vector_store.VectorStoreRepository(data_dir.vector_db_dir())
    backend = vector_store.InMemoryVectorStoreBackend()
    embedder = StubEmbeddingClient()
    chunker = rag_cli._build_chunker(cfg)
    dedupe = rag_cli._build_dedupe(cfg)
    chunking_meta = rag_cli._build_chunking(cfg)
    doc = tmp_path / "notes.txt"
    doc.write_text("Fundamental theorem of calculus.", encoding="utf-8")
    ingest_mod.ingest_sources(
        "notes",
        inputs=[doc],
        repository=repo,
        backend=backend,
        embedder=embedder,
        chunker=chunker,
        embedding_provider=cfg.providers.default,
        embedding_model=cfg.providers.openai.embedding_model,
        dedupe=dedupe,
        chunking=chunking_meta,
        overwrite=True,
    )
    custom_retrieval = config_mod.RetrievalConfig(
        top_k=3,
        max_context_tokens=4,
        score_threshold=0.0,
    )
    custom_chat = config_mod.ChatConfig(
        max_history_turns=1,
        response_tokens=64,
        stream=False,
    )
    cfg = replace(cfg, retrieval=custom_retrieval, chat=custom_chat)
    store = session_mod.SessionStore(data_dir.sessions_dir())
    chat_client = StubChatClient()
    runtime = chat_mod.ChatRuntime(
        config=cfg,
        repository=repo,
        session_store=store,
        embedder=embedder,
        chat_client=chat_client,
    )
    return cfg, runtime, chat_client, store


def test_chat_runtime_answers_and_trims_history(runtime_env):
    cfg, runtime, chat_client, store = runtime_env
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))

    answer1 = runtime.ask(session, "What is described?")
    assert "Stub answer" in answer1.response
    assert answer1.contexts

    answer2 = runtime.ask(session, "Second question?")
    assert answer2.session_id == session.session_id
    assert answer2.contexts

    stored = store.load(session.session_id)
    assert len(stored.messages) == 2
    assert stored.messages[0].role == "user"
    assert stored.messages[0].content == "Second question?"
    assert len(chat_client.calls) == 2


def test_chat_runtime_resume_adds_new_db(runtime_env, tmp_path):
    cfg, runtime, chat_client, _ = runtime_env
    repo = vector_store.VectorStoreRepository(data_dir.vector_db_dir())

    other_doc = tmp_path / "algebra.txt"
    other_doc.write_text("Linear algebra primer", encoding="utf-8")
    ingest_mod.ingest_sources(
        "algebra",
        inputs=[other_doc],
        repository=repo,
        backend=vector_store.InMemoryVectorStoreBackend(),
        embedder=StubEmbeddingClient(),
        chunker=rag_cli._build_chunker(cfg),
        embedding_provider=cfg.providers.default,
        embedding_model=cfg.providers.openai.embedding_model,
        dedupe=rag_cli._build_dedupe(cfg),
        chunking=rag_cli._build_chunking(cfg),
        overwrite=True,
    )

    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))
    session = runtime.prepare_session(
        resume_id=session.session_id,
        vector_dbs=("algebra",),
    )
    assert session.vector_dbs == ("algebra", "notes")

    new_retrieval = config_mod.RetrievalConfig(
        top_k=5,
        max_context_tokens=10,
        score_threshold=0.0,
    )
    runtime._retriever._retrieval = new_retrieval
    runtime._config = replace(runtime._config, retrieval=new_retrieval)
    runtime.ask(session, "Explain both subjects")
    assert any(
        call for call in chat_client.calls if "Context 1" in call[-1]["content"]
    )


def test_chat_runtime_interactive_loop_handles_exit(runtime_env):
    _, runtime, chat_client, _ = runtime_env
    new_retrieval = config_mod.RetrievalConfig(
        top_k=3,
        max_context_tokens=4,
        score_threshold=1.1,
    )
    runtime._retriever._retrieval = new_retrieval
    runtime._config = replace(runtime._config, retrieval=new_retrieval)
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))
    console = StubConsole(["What is this?", ":q"])
    runtime.interactive_loop(session, console=console)
    assert any("Goodbye!" in str(item) for item in console.outputs)
    assert len(chat_client.calls) == 1


def test_chat_runtime_requires_vector_db(runtime_env):
    _, runtime, _, _ = runtime_env
    with pytest.raises(chat_mod.ChatError):
        runtime.prepare_session(resume_id=None, vector_dbs=())


def test_openai_chat_client_complete():
    class DummyResponse:
        def __init__(self):
            self.choices = [
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=" hello ")
                )
            ]

    class DummyClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):  # noqa: D401
            self.kwargs = kwargs
            return DummyResponse()

    client = DummyClient()
    chat_client = chat_mod.OpenAIChatClient(
        model="gpt-4o-mini",
        temperature=0.2,
        max_output_tokens=32,
        request_timeout=30,
        api_base=None,
        client=client,
    )
    result = chat_client.complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=16,
        stream=False,
        timeout=45,
    )
    assert result == "hello"
    assert client.kwargs["model"] == "gpt-4o-mini"


def test_openai_chat_client_initializes_from_loader(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.choices = [
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hi")
                )
            ]

    class DummyClient:
        base_url: str | None

        def __init__(self):
            self.base_url = None
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):  # noqa: D401
            self.kwargs = kwargs
            return DummyResponse()

    dummy_client = DummyClient()

    def _mock_load_openai_client(
        local: bool = False,
        api_base: str | None = None,
    ):
        if api_base and hasattr(dummy_client, "base_url"):
            dummy_client.base_url = api_base
        return dummy_client

    monkeypatch.setattr(
        chat_mod,
        "load_openai_client",
        _mock_load_openai_client,
    )

    chat_client = chat_mod.OpenAIChatClient(
        model="gpt-4o-mini",
        temperature=0.7,
        max_output_tokens=64,
        request_timeout=20,
        api_base="https://example.test",
    )
    assert dummy_client.base_url == "https://example.test"

    response = chat_client.complete(
        model="gpt-4o-mini-second",
        messages=[{"role": "user", "content": "Question"}],
        temperature=0.1,
        max_tokens=8,
        stream=True,
        timeout=5,
    )
    assert response == "hi"
    assert dummy_client.kwargs["model"] == "gpt-4o-mini-second"
    assert dummy_client.kwargs["timeout"] == 5


def test_retrieval_index_requires_embeddings(tmp_path):
    manifest = make_manifest()
    records = [
        chat_mod._EmbeddingRecord(text="chunk", metadata={}, embedding=None)
    ]
    with pytest.raises(chat_mod.ChatError) as excinfo:
        chat_mod._RetrievalIndex(
            name="db",
            manifest=manifest,
            records=records,
            store_path=tmp_path,
        )
    assert "missing embeddings" in str(excinfo.value)


def test_retrieval_index_loads_faiss_backend(tmp_path, monkeypatch):
    manifest = make_manifest(dimension=3)
    records = [
        chat_mod._EmbeddingRecord(
            text="first",
            metadata={"source_path": "first.md"},
            embedding=None,
        ),
        chat_mod._EmbeddingRecord(
            text="second",
            metadata={"source_path": "second.md"},
            embedding=None,
        ),
    ]
    faiss_path = tmp_path / "index.faiss"
    faiss_path.write_bytes(b"stub")

    class FakeArray:
        def __init__(self, data):
            self.data = data

        def reshape(self, *_):  # noqa: D401
            return self

    class FakeNumpy:
        def array(self, data, dtype=None):  # noqa: D401, ANN001
            return FakeArray(data)

    fake_index = types.SimpleNamespace(
        search=lambda array, top_k: ([[0.0, 1.0]], [[0, 1]])
    )
    fake_faiss = types.SimpleNamespace(read_index=lambda path: fake_index)

    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())

    retrieval_index = chat_mod._RetrievalIndex(
        name="db",
        manifest=manifest,
        records=records,
        store_path=tmp_path,
    )
    results = retrieval_index.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_retrieval_index_search_memory_handles_empty(tmp_path):
    manifest = make_manifest()
    records = [
        chat_mod._EmbeddingRecord(
            text="kept", metadata={}, embedding=[1.0, 0.0]
        )
    ]
    retrieval_index = chat_mod._RetrievalIndex(
        name="db",
        manifest=manifest,
        records=records,
        store_path=tmp_path,
    )
    retrieval_index._records = []
    assert retrieval_index.search([0.0, 1.0], top_k=3) == []


def test_retrieval_index_skips_missing_embeddings(tmp_path):
    manifest = make_manifest()
    records = [
        chat_mod._EmbeddingRecord(text="skip", metadata={}, embedding=None),
        chat_mod._EmbeddingRecord(
            text="keep", metadata={}, embedding=[1.0, 1.0]
        ),
    ]
    retrieval_index = chat_mod._RetrievalIndex(
        name="db",
        manifest=manifest,
        records=records,
        store_path=tmp_path,
    )
    results = retrieval_index.search([1.0, 1.0], top_k=1)
    assert len(results) == 1
    assert results[0].text == "keep"


def test_retriever_errors_on_missing_embeddings():
    config = config_mod.RetrievalConfig(
        top_k=1,
        max_context_tokens=128,
        score_threshold=0.0,
    )

    class MissingEmbeddingClient:
        def embed_documents(self, texts):  # noqa: D401, ANN001
            return []

    retriever = chat_mod._Retriever(
        repository=types.SimpleNamespace(),
        embedder=MissingEmbeddingClient(),
        retrieval_config=config,
    )
    with pytest.raises(chat_mod.ChatError):
        retriever.retrieve(question="q", indexes=[])


def test_retriever_validates_embedding_dimension():
    config = config_mod.RetrievalConfig(
        top_k=1,
        max_context_tokens=128,
        score_threshold=0.0,
    )

    class Embedder:
        def embed_documents(self, texts):  # noqa: D401, ANN001
            return [[1.0, 2.0]]

    class WrongDimensionIndex:
        _name = "db"
        embedding_dimension = 3

        def search(self, query, top_k):  # noqa: D401, ANN001
            return []

    retriever = chat_mod._Retriever(
        repository=types.SimpleNamespace(),
        embedder=Embedder(),
        retrieval_config=config,
    )
    with pytest.raises(chat_mod.ChatError) as excinfo:
        retriever.retrieve(question="q", indexes=[WrongDimensionIndex()])
    assert "dimension" in str(excinfo.value)


def test_retriever_honours_top_k_break():
    config = config_mod.RetrievalConfig(
        top_k=1,
        max_context_tokens=128,
        score_threshold=0.0,
    )

    class Embedder:
        def embed_documents(self, texts):  # noqa: D401, ANN001
            return [[1.0]]

    result_one = chat_mod.RetrievalResult(
        db_name="db",
        text="first",
        score=0.9,
        metadata={},
    )
    result_two = chat_mod.RetrievalResult(
        db_name="db",
        text="second",
        score=0.1,
        metadata={},
    )

    class Index:
        embedding_dimension = 1

        def search(self, query, top_k):  # noqa: D401, ANN001
            return [result_one, result_two]

    retriever = chat_mod._Retriever(
        repository=types.SimpleNamespace(),
        embedder=Embedder(),
        retrieval_config=config,
    )
    contexts, _ = retriever.retrieve(question="q", indexes=[Index()])
    assert contexts == [result_one]


def test_load_records_missing_file(tmp_path):
    store_path = tmp_path / "db"
    store_path.mkdir()
    with pytest.raises(chat_mod.ChatError):
        chat_mod._load_records(store_path)


def test_load_records_invalid_json(tmp_path):
    store_path = tmp_path / "db"
    store_path.mkdir()
    (store_path / "records.json").write_text("not json", encoding="utf-8")
    with pytest.raises(chat_mod.ChatError):
        chat_mod._load_records(store_path)


def test_load_records_unexpected_structure(tmp_path):
    store_path = tmp_path / "db"
    store_path.mkdir()
    (store_path / "records.json").write_text("{}", encoding="utf-8")
    with pytest.raises(chat_mod.ChatError):
        chat_mod._load_records(store_path)


def test_load_records_normalises_metadata(tmp_path):
    store_path = tmp_path / "db"
    store_path.mkdir()
    payload = [
        {"text": "one", "metadata": "nope", "embedding": [1, 2]},
        {"text": "two", "metadata": {}, "embedding": None},
    ]
    (store_path / "records.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    records = chat_mod._load_records(store_path)
    assert len(records) == 2
    assert records[0].metadata == {}
    assert records[1].embedding is None


def test_cosine_similarity_handles_zero_vector():
    assert chat_mod._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def _build_runtime_for_validation(cfg, repo, tmp_path):
    session_store = session_mod.SessionStore(tmp_path / "sessions")
    embedder = StubEmbeddingClient()
    chat_client = StubChatClient()
    return chat_mod.ChatRuntime(
        config=cfg,
        repository=repo,
        session_store=session_store,
        embedder=embedder,
        chat_client=chat_client,
    )


def test_prepare_session_provider_mismatch(runtime_env, tmp_path):
    cfg, _, _, _ = runtime_env
    manifests = {"bad": make_manifest(provider="cohere")}

    class Repo:
        def load_manifest(self, name):  # noqa: D401, ANN001
            return manifests[name]

        def store_path(self, name):  # noqa: D401, ANN001
            path = tmp_path / name
            path.mkdir(parents=True, exist_ok=True)
            return path

    runtime = _build_runtime_for_validation(cfg, Repo(), tmp_path)
    with pytest.raises(chat_mod.ChatError):
        runtime.prepare_session(resume_id=None, vector_dbs=("bad",))


def test_prepare_session_validates_secondary_manifests(runtime_env, tmp_path):
    cfg, _, _, _ = runtime_env
    manifests = {
        "alpha": make_manifest(),
        "beta": make_manifest(provider="cohere"),
    }

    class Repo:
        def load_manifest(self, name):  # noqa: D401, ANN001
            return manifests[name]

        def store_path(self, name):  # noqa: D401, ANN001
            path = tmp_path / name
            path.mkdir(parents=True, exist_ok=True)
            return path

    runtime = _build_runtime_for_validation(cfg, Repo(), tmp_path)
    with pytest.raises(chat_mod.ChatError) as excinfo:
        runtime.prepare_session(resume_id=None, vector_dbs=("alpha", "beta"))
    assert "different embedding providers" in str(excinfo.value)

    manifests["beta"] = make_manifest(provider="openai", model="different")
    with pytest.raises(chat_mod.ChatError) as excinfo:
        runtime.prepare_session(resume_id=None, vector_dbs=("alpha", "beta"))
    assert "different embedding models" in str(excinfo.value)


def test_prepare_session_validates_dimensions(runtime_env, tmp_path):
    cfg, _, _, _ = runtime_env
    manifests = {
        "alpha": make_manifest(dimension=2),
        "beta": make_manifest(dimension=3),
    }

    class Repo:
        def load_manifest(self, name):  # noqa: D401, ANN001
            return manifests[name]

        def store_path(self, name):  # noqa: D401, ANN001
            path = tmp_path / name
            path.mkdir(parents=True, exist_ok=True)
            return path

    runtime = _build_runtime_for_validation(cfg, Repo(), tmp_path)
    with pytest.raises(chat_mod.ChatError) as excinfo:
        runtime.prepare_session(resume_id=None, vector_dbs=("alpha", "beta"))
    assert "mismatched dimensions" in str(excinfo.value)


def test_chat_runtime_rejects_blank_prompt(runtime_env):
    _, runtime, _, _ = runtime_env
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))
    with pytest.raises(chat_mod.ChatError):
        runtime.ask(session, "   ")


def test_interactive_loop_handles_eof(runtime_env):
    _, runtime, _, _ = runtime_env
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))

    class EOFConsole:
        def print(self, message):  # noqa: D401, ANN001
            self.last = message

        def input(self, prompt):  # noqa: D401, ANN001
            raise EOFError

    console = EOFConsole()
    runtime.interactive_loop(session, console=console)
    assert "Exiting" in console.last


def test_interactive_loop_skips_blank_input(runtime_env):
    _, runtime, _, _ = runtime_env
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))

    class Console:
        def __init__(self):
            self.calls = []
            self._prompts = iter(["   ", ":quit"])

        def print(self, message):  # noqa: D401, ANN001
            self.calls.append(message)

        def input(self, prompt):  # noqa: D401, ANN001
            self.calls.append(prompt)
            return next(self._prompts)

    console = Console()
    runtime.interactive_loop(session, console=console)
    assert any("Goodbye" in str(item) for item in console.calls)


def test_interactive_loop_reports_chat_error(runtime_env, monkeypatch):
    _, runtime, _, _ = runtime_env
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))

    def fail(*args, **kwargs):  # noqa: D401, ANN001
        raise chat_mod.ChatError("nope")

    monkeypatch.setattr(runtime, "ask", fail)

    class Console:
        def __init__(self):
            self._prompts = iter(["fail", ":q"])
            self.messages = []

        def print(self, message):  # noqa: D401, ANN001
            self.messages.append(message)

        def input(self, prompt):  # noqa: D401, ANN001
            self.messages.append(prompt)
            return next(self._prompts)

    console = Console()
    runtime.interactive_loop(session, console=console)
    assert any("Error" in str(item) for item in console.messages)


def test_render_answer_renders_contexts(runtime_env):
    _, runtime, _, _ = runtime_env
    session = runtime.prepare_session(resume_id=None, vector_dbs=("notes",))
    answer = chat_mod.ChatAnswer(
        session_id=session.session_id,
        prompt="q",
        response="res",
        contexts=[
            chat_mod.RetrievalResult(
                db_name="db",
                text="context",
                score=0.5,
                metadata={"source_path": "doc", "chunk_index": 1},
            )
        ],
    )

    class Console:
        def __init__(self):
            self.messages = []

        def print(self, message):  # noqa: D401, ANN001
            self.messages.append(message)

    console = Console()
    runtime._render_answer(console, answer)
    titles = [str(getattr(item, "title", "")) for item in console.messages]
    assert "Context 1 (db)" in titles


def test_ensure_vector_dbs_detects_provider_mismatch(runtime_env, tmp_path):
    cfg, _, _, _ = runtime_env
    manifests = {"alpha": make_manifest(provider="openai")}

    class Repo:
        def load_manifest(self, name):  # noqa: D401, ANN001
            return manifests[name]

        def store_path(self, name):  # noqa: D401, ANN001
            return tmp_path / name

    runtime = _build_runtime_for_validation(cfg, Repo(), tmp_path)
    session = session_mod.ChatSession(
        session_id="sess",
        directory=tmp_path,
        created_at="now",
        updated_at="now",
        vector_dbs=("alpha",),
        embedding_provider="cohere",
        embedding_model="model",
        embedding_dimension=2,
        chat_model="gpt",
        messages=[],
    )
    with pytest.raises(chat_mod.ChatError):
        runtime._ensure_vector_dbs(session)


def test_ensure_vector_dbs_detects_model_and_dimension_mismatch(
    runtime_env, tmp_path
):
    cfg, _, _, _ = runtime_env
    manifests = {
        "alpha": make_manifest(provider="openai", model="model", dimension=3)
    }

    class Repo:
        def load_manifest(self, name):  # noqa: D401, ANN001
            return manifests[name]

        def store_path(self, name):  # noqa: D401, ANN001
            return tmp_path / name

    runtime = _build_runtime_for_validation(cfg, Repo(), tmp_path)
    session = session_mod.ChatSession(
        session_id="sess",
        directory=tmp_path,
        created_at="now",
        updated_at="now",
        vector_dbs=("alpha",),
        embedding_provider="openai",
        embedding_model="other",
        embedding_dimension=2,
        chat_model="gpt",
        messages=[],
    )
    with pytest.raises(chat_mod.ChatError) as excinfo:
        runtime._ensure_vector_dbs(session)
    assert "embedding model" in str(excinfo.value)

    session.embedding_model = "model"
    with pytest.raises(chat_mod.ChatError) as excinfo:
        runtime._ensure_vector_dbs(session)
    assert "dimension" in str(excinfo.value)
