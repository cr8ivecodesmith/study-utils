"""UAT script for testing the AI client against an OpenAI-compatible server.

Usage::

    uv run python -m uat.test_client                  # uses default config path
    uv run python -m uat.test_client --config foo.toml
    API_BASE=http://10.0.0.1:8080 uv run python -m uat.test_client
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:
    import tomli as tomllib  # type: ignore


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and merge TOML config with environment overrides."""
    cfg = _load_toml(config_path)
    if "api_base" in cfg:
        cfg["api_base"] = cfg["api_base"]
    return cfg


def _resolve_audio_path(cfg: dict[str, Any], config_dir: Path) -> Path:
    audio_file = cfg.get("audio", {}).get("file", "samples/kokoro_out_0.wav")
    p = Path(audio_file)
    if not p.is_absolute():
        p = config_dir / p
    if not p.exists():
        print(f"  WARNING: audio file not found: {p}")
    return p


def _print_header(title: str, width: int = 50) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _print_result(label: str, value: Any) -> None:
    print(f"  {label}: {value}")


def test_chat(client: Any, cfg: dict[str, Any]) -> bool:
    """Test chat completions endpoint."""
    chat_cfg = cfg.get("chat", {})
    model = chat_cfg.get("model", "default")
    prompt = chat_cfg.get("prompt", "Hello, who are you?")
    max_tokens = chat_cfg.get("max_tokens", 50)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        _print_header("[1/4] Chat Completions")
        _print_result("Model", response.model)
        _print_result("Response", choice.message.content)
        tokens = response.usage.total_tokens if response.usage else "N/A"
        _print_result("Tokens used", tokens)
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def test_embeddings(client: Any, cfg: dict[str, Any]) -> bool:
    """Test embeddings endpoint."""
    emb_cfg = cfg.get("embeddings", {})
    model = emb_cfg.get("model", "default")
    text = emb_cfg.get("input", "Test sentence for embeddings.")

    try:
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        data = response.data[0]
        vec = data.embedding
        _print_header("[2/4] Embeddings")
        _print_result("Model", response.model)
        _print_result("Dimensions", len(vec))
        preview = ", ".join(f"{v:.4f}" for v in vec[:6])
        _print_result("Vector preview", f"[{preview}, ...]")
        return True
    except Exception as exc:
        import traceback
        print(f"  FAILED: {exc}")
        traceback.print_exc()
        return False


def test_rerank(client: Any, cfg: dict[str, Any]) -> bool:
    """Test rerank endpoint."""
    rr_cfg = cfg.get("rerank", {})
    model = rr_cfg.get("model", "default")
    query = rr_cfg.get("query", "AI chatbots")
    docs = rr_cfg.get(
        "docs",
        [
            "Chatbots are AI.",
            "Robots are mechanical.",
            "LLMs power modern chatbots.",
        ],
    )

    try:
        body = {
            "model": model,
            "query": query,
            "documents": docs,
        }
        resp = client.post("/rerank", body=body, cast_to=object)
        data = resp
        results = data.get("results", data if isinstance(data, list) else [])
        _print_header("[3/4] Rerank")
        _print_result("Query", query)
        if isinstance(results, list) and len(results) > 0:
            if "index" in results[0]:
                for item in results:
                    idx = item["index"]
                    text = docs[idx] if idx < len(docs) else str(idx)
                    score = item.get(
                        "relevance_score", item.get("score", "N/A")
                    )
                    _print_result(
                        f"Rank {idx + 1}", f"{text} (score: {score})"
                    )
            else:
                for i, item in enumerate(results):
                    text = item.get(
                        "text", item.get("document", str(item))
                    )
                    score = item.get(
                        "relevance_score", item.get("score", "N/A")
                    )
                    _print_result(f"Rank {i + 1}", f"{text} (score: {score})")
        else:
            for i, doc in enumerate(docs):
                _print_result(f"Rank {i + 1}", doc)
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def test_audio(client: Any, cfg: dict[str, Any], config_dir: Path) -> bool:
    """Test audio transcription endpoint."""
    audio_cfg = cfg.get("audio", {})
    audio_file = _resolve_audio_path(cfg, config_dir)
    language = audio_cfg.get("language", "auto")
    model = audio_cfg.get("model", "whisper")

    try:
        with open(audio_file, "rb") as fh:
            response = client.audio.transcriptions.create(
                model=model,
                file=fh,
                language=language,
            )
        _print_header("[4/4] Audio Transcription")
        _print_result("File", str(audio_file.name))
        _print_result("Language", language)
        _print_result("Text", response.text)
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="UAT script for AI client")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config file (default: uat/local_client_config.toml)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
    else:
        config_path = script_dir / "local_client_config.toml"

    cfg = _load_config(config_path)
    api_base = cfg.get("api_base", "http://192.168.8.169:10000")
    key_source = cfg.get("key_source", "local")
    local = key_source == "local"

    print(f"\n{'=' * 50}")
    print(f"  AI Client UAT — {api_base}")
    print(f"  Key source: {key_source} (local={local})")
    print(f"  Config: {config_path}")
    print(f"{'=' * 50}")

    from study_utils.core.ai import load_client

    client = load_client(local=local, api_base=api_base)
    client_type = f"{type(client).__module__}.{type(client).__name__}"
    _print_result("Client", client_type)

    passed = 0
    total = 4

    try:
        if test_chat(client, cfg):
            passed += 1
    except Exception as exc:
        print(f"  FAILED (unexpected): {exc}")

    try:
        if test_embeddings(client, cfg):
            passed += 1
    except Exception as exc:
        print(f"  FAILED (unexpected): {exc}")

    try:
        if test_rerank(client, cfg):
            passed += 1
    except Exception as exc:
        print(f"  FAILED (unexpected): {exc}")

    try:
        if test_audio(client, cfg, config_dir=config_path.parent):
            passed += 1
    except Exception as exc:
        print(f"  FAILED (unexpected): {exc}")

    print(f"\n{'=' * 50}")
    status = f"{passed}/{total} tests passed"
    if passed == total:
        print("  All tests passed ✓")
    else:
        print(f"  {status} — see failures above")
    print(f"{'=' * 50}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
