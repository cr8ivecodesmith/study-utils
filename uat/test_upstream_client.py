"""UAT script for testing the llama-swap upstream client.

Tests the /upstream/{model_id}/{endpoint} route against a llama-swap instance.

Usage::

    uv run python -m uat.test_upstream_client  # uses default config
    uv run python -m uat.test_upstream_client --config foo.toml
    API_BASE=http://10.0.0.1:8080 uv run \
        python -m uat.test_upstream_client
"""

from __future__ import annotations

import argparse
import base64
import json
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


def _resolve_sample_path(
    cfg: dict[str, Any], config_dir: Path, key: str
) -> Path:
    sample_file = cfg.get(key, {}).get("file", "samples/doc-page-1.png")
    p = Path(sample_file)
    if not p.is_absolute():
        p = config_dir / p
    if not p.exists():
        print(f"  WARNING: sample file not found: {p}")
    return p


def _print_header(title: str, width: int = 50) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _print_result(label: str, value: Any) -> None:
    print(f"  {label}: {value}")


def _parse_snippet_config(
    cfg: dict[str, Any], test_name: str,
) -> dict[str, Any]:
    """Extract snippet settings from [snippet] section and per-test override."""
    snip = cfg.get("snippet", {})
    max_chars = int(snip.get("max_chars", 400))
    json_indent = int(snip.get("json_indent", 2))
    test_cfg = cfg.get(test_name, {})

    if test_name == "kokoro":
        return {
            "show_preview": bool(test_cfg.get("show_preview", True)),
            "max_len": int(
                test_cfg.get(
                    "base64_preview_len", snip.get(
                        "max_chars", 300,
                    ),
                ),
            ),
            "is_binary": True,
            "json_indent": json_indent,
        }
    else:
        if tc_extra := test_cfg.get("snippet_max_chars"):
            max_chars = int(tc_extra)
        return {
            "show_preview": bool(test_cfg.get("show_preview", True)),
            "max_chars": max_chars,
            "is_binary": False,
            "json_indent": json_indent,
        }


def _print_body_snippet(resp: Any) -> None:
    """Print a text/JSON body snippet truncated at the end.

    Tries ``resp.json()`` first; if it parses, pretty-prints with indentation.
    Otherwise falls back to ``resp.text`` raw display.
    Truncated content gets ``\u2026`` appended (truncation at the end).
    """
    snip_cfg = getattr(resp, "_snippet_config", {})
    max_chars = snip_cfg.get("max_chars", 400)
    indent = int(snip_cfg.get("json_indent", 2))

    try:
        parsed = resp.json()
        body_str = json.dumps(parsed, indent=indent, ensure_ascii=False)
    except (ValueError, AttributeError):
        body_str = resp.text if hasattr(resp, "text") else resp.content.decode(
            errors="replace",
        )

    if len(body_str) > max_chars:
        body_str = body_str[:max_chars] + "\u2026"

    # Wrap lines at ~80 chars for readability
    lines: list[str] = []
    for raw_line in body_str.splitlines(True):
        line = raw_line.rstrip("\n").rstrip("\r")
        if len(line) > 78:
            while len(line) > 78:
                mid = line.rfind(" ", 0, 79)
                if mid == -1:
                    lines.append(line[:75] + "\u2026")
                    break
                lines.append(line[:mid])
                line = line[mid:].lstrip()
        else:
            lines.append(line)

    print(f"  Snippet ({len(resp.content):,} bytes, first {max_chars} chars):")
    for sl in lines:
        print(f"    {sl}")


def _print_base64_snippet(resp: Any) -> None:
    """Print a base64-encoded binary snippet (audio / image)."""
    snip_cfg = getattr(resp, "_snippet_config", {})
    max_len = int(snip_cfg.get("max_len", 300))
    total_bytes = len(resp.content)

    b64_str = base64.b64encode(resp.content).decode("ascii")
    preview = b64_str[:max_len]

    if len(b64_str) > max_len:
        preview += "\u2026"
        preview += f"(truncated, total={total_bytes:,} bytes)"
    else:
        preview += f" ({total_bytes:,} bytes)"

    ct = resp.headers.get("content-type", "").lower()
    if "audio" in ct or "wav" in ct:
        tag = "[audio]"
    elif any(p in ct for p in ("png", "jpeg", "gif", "webp")):
        tag = "[image]"
    else:
        tag = "[binary]"

    print(f"  Snippet (base64 preview ~{max_len} chars):")
    print(f"    {tag} {preview}")


def test_mineru(cfg: dict[str, Any], config_dir: Path) -> bool:
    """Test Mineru document parsing via /upstream/mineru25/parse."""
    mineru_cfg = cfg.get("mineru", {})
    model_id = mineru_cfg.get("model_id", "mineru25")
    endpoint = mineru_cfg.get("endpoint", "parse")
    api_base = cfg.get("api_base", "http://localhost:8080")

    sample_path = _resolve_sample_path(cfg, config_dir, "mineru")

    try:
        from study_utils.core.ai import load_llama_swap_upstream_client

        with sample_path.open("rb") as fh:
            file_payload = {
                "file": (sample_path.name, fh, "application/octet-stream"),
            }
            resp = load_llama_swap_upstream_client(
                model_id=model_id,
                endpoint=endpoint,
                api_base=api_base,
                method="POST",
                files=file_payload,
            )

        _print_header("[1/3] Mineru Parse")
        _print_result("Endpoint", f"/upstream/{model_id}/{endpoint}")
        _print_result("File", str(sample_path.name))
        _print_result("Status", resp.status_code)
        _print_result("Content-Type", resp.headers.get("content-type", "N/A"))
        if resp.status_code == 200:
            _print_result("Response size", f"{len(resp.content)} bytes")
            snippet_cfg = _parse_snippet_config(cfg, "mineru")
            if snippet_cfg.get("show_preview"):
                resp._snippet_config = snippet_cfg
                _print_body_snippet(resp)
            return True
        else:
            print(f"  FAILED: unexpected status {resp.status_code}")
            return False
    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_kokoro(cfg: dict[str, Any], config_dir: Path) -> bool:
    """Test Kokoro text-to-speech via /upstream/kokoro/speak."""
    kokoro_cfg = cfg.get("kokoro", {})
    model_id = kokoro_cfg.get("model_id", "kokoro")
    endpoint = kokoro_cfg.get("endpoint", "speak")
    api_base = cfg.get("api_base", "http://localhost:8080")

    text = kokoro_cfg.get("text", "Testing Kokoro on the auxiliary swap.")
    voice = kokoro_cfg.get("voice", "af_bella")
    output_file = kokoro_cfg.get("output", "samples/kokoro_upstream.wav")
    p = Path(output_file)
    if not p.is_absolute():
        p = config_dir / p

    try:
        from study_utils.core.ai import load_llama_swap_upstream_client

        json_body = {"text": text, "voice": voice}
        resp = load_llama_swap_upstream_client(
            model_id=model_id,
            endpoint=endpoint,
            api_base=api_base,
            method="POST",
            json=json_body,
        )

        _print_header("[2/3] Kokoro Speak")
        _print_result("Endpoint", f"/upstream/{model_id}/{endpoint}")
        _print_result("Text", text)
        _print_result("Voice", voice)
        _print_result("Status", resp.status_code)
        _print_result("Content-Type", resp.headers.get("content-type", "N/A"))

        if resp.status_code == 200:
            # Write audio output to file
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(resp.content)
            _print_result("Output", str(p))
            _print_result("Audio size", f"{len(resp.content)} bytes")
            snippet_cfg = _parse_snippet_config(cfg, "kokoro")
            if snippet_cfg.get("show_preview"):
                resp._snippet_config = snippet_cfg
                _print_base64_snippet(resp)
            return True
        else:
            print(f"  FAILED: unexpected status {resp.status_code}")
            return False
    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_pyannote(cfg: dict[str, Any], config_dir: Path) -> bool:
    """Test Pyannote diarization via /upstream/pyannote/diarize."""
    pyannote_cfg = cfg.get("pyannote", {})
    model_id = pyannote_cfg.get("model_id", "pyannote")
    endpoint = pyannote_cfg.get("endpoint", "diarize")
    api_base = cfg.get("api_base", "http://localhost:8080")

    sample_path = _resolve_sample_path(cfg, config_dir, "pyannote")

    try:
        from study_utils.core.ai import load_llama_swap_upstream_client

        with sample_path.open("rb") as fh:
            file_payload = {
                "file": (sample_path.name, fh, "application/octet-stream"),
            }
            resp = load_llama_swap_upstream_client(
                model_id=model_id,
                endpoint=endpoint,
                api_base=api_base,
                method="POST",
                files=file_payload,
            )

        _print_header("[3/3] Pyannote Diarize")
        _print_result("Endpoint", f"/upstream/{model_id}/{endpoint}")
        _print_result("File", str(sample_path.name))
        _print_result("Status", resp.status_code)
        _print_result("Content-Type", resp.headers.get("content-type", "N/A"))
        if resp.status_code == 200:
            _print_result("Response size", f"{len(resp.content)} bytes")
            snippet_cfg = _parse_snippet_config(cfg, "pyannote")
            if snippet_cfg.get("show_preview"):
                resp._snippet_config = snippet_cfg
                _print_body_snippet(resp)
            return True
        else:
            print(f"  FAILED: unexpected status {resp.status_code}")
            return False
    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UAT script for llama-swap upstream client"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to TOML config file "
            "(default: uat/upstream_client_config.toml)"
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
    else:
        config_path = script_dir / "upstream_client_config.toml"

    cfg = _load_config(config_path)
    api_base = cfg.get("api_base", "http://localhost:8080")
    key_source = cfg.get("key_source", "local")

    print(f"\n{'=' * 50}")
    print(f"  Llama-swap Upstream UAT — {api_base}")
    print(f"  Key source: {key_source}")
    print(f"  Config: {config_path}")
    print(f"{'=' * 50}")

    passed = 0
    total = 3

    try:
        if test_mineru(cfg, config_dir=config_path.parent):
            passed += 1
    except Exception as exc:
        print(f"  FAILED (unexpected): {exc}")

    try:
        if test_kokoro(cfg, config_dir=config_path.parent):
            passed += 1
    except Exception as exc:
        print(f"  FAILED (unexpected): {exc}")

    try:
        if test_pyannote(cfg, config_dir=config_path.parent):
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
