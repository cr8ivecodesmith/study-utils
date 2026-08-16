import argparse
import os
import random
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from ..core import load_client

from .utils import (
    _find_config,
    _load_toml,
    _get_quiz_section,
    _read_files,
    iter_quiz_files,
    write_jsonl,
    read_jsonl,
)

from .manager.quiz import generate_questions, extract_topics
from .session import run_quiz_session
from .config import (
    CONFIG_ENV,
    CONFIG_FILENAME,
    QuizzerConfigError,
    load_config,
    validate_config,
    write_template,
)


def _out_dir_for(name: str, cfg: Optional[dict]) -> Path:
    d = None
    if cfg:
        st = cfg.get("storage") or {}
        d = st.get("out_dir") if isinstance(st, dict) else None
    base = (
        Path(d.replace("<name>", name))
        if isinstance(d, str)
        else Path(".quizzer") / name
    )
    return base.resolve()


def _cmd_init(args: argparse.Namespace) -> int:
    name: str = args.name
    path = Path("quizzer.toml").resolve()
    if path.exists():
        print(f"quizzer.toml already exists at {path}")
        return 0
    sample = (
        "# Quizzer configuration\n"
        "# Define quizzes under [quiz.<name>] sections\n\n"
        f"[quiz.{name}]\n"
        "# One or more files or directories (Markdown)\n"
        'sources = ["./materials"]\n'
        'types = ["mcq"]\n'
        "per_topic = 3\n"
        "ensure_coverage = true\n\n"
        "[storage]\n"
        "# Artifacts directory; <name> will be replaced with quiz name\n"
        'out_dir = ".quizzer/<name>"\n\n'
        "[ai]\n"
        'model = "gpt-4o-mini"\n'
        "temperature = 0.2\n"
        "max_tokens = 600\n"
    )
    path.write_text(sample, encoding="utf-8")
    print(f"Created template {path}")
    return 0


def _cmd_topics_generate(args: argparse.Namespace) -> int:
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found. Run 'quizzer init <name>' first.")
        return 2
    cfg = _load_toml(cfg_path)
    try:
        section = _get_quiz_section(cfg, args.name)
    except KeyError as exc:
        print(f"Error: {exc}")
        return 2
    sources = section.get("sources") or []
    if not sources:
        print(f"Error: [quiz.{args.name}] must define 'sources' list")
        return 2
    input_paths = [Path(s).expanduser().resolve() for s in sources]
    files = iter_quiz_files(
        input_paths,
        extensions=args.extensions,
        level_limit=int(args.level_limit),
    )
    if not files:
        print("No matching Markdown files found.")
        return 1
    pairs = _read_files(files)
    topics = extract_topics(pairs, use_ai=bool(args.use_ai))
    out_dir = _out_dir_for(args.name, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    topics_path = out_dir / "topics.jsonl"
    write_jsonl(topics_path, topics)
    print(f"Wrote {len(topics)} topic(s) -> {topics_path}")
    return 0


def _cmd_topics_list(args: argparse.Namespace) -> int:
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found.")
        return 2
    cfg = _load_toml(cfg_path)
    out_dir = _out_dir_for(args.name, cfg)
    topics_path = out_dir / "topics.jsonl"
    if not topics_path.exists():
        print(
            f"No topics found at {topics_path}. Run 'quizzer topics generate "
            f"{args.name}'."
        )
        return 1
    topics = read_jsonl(topics_path)
    filt = (args.filter or "").lower().strip()
    shown = 0
    for t in topics:
        name = str(t.get("name", ""))
        if not filt or filt in name.lower():
            print(f"- {name}")
            shown += 1
    if shown == 0:
        print("No topics match filter.")
        return 1
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    path = Path(getattr(args, "path", None) or CONFIG_FILENAME).expanduser()
    force = getattr(args, "force", False)

    try:
        write_template(path, overwrite=force)
        print(f"Created config template at {path}")
        return 0
    except QuizzerConfigError as exc:
        print(f"Error: {exc}")
        return 1


def _cmd_config_validate(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", None)
    try:
        target = Path(config_path) if config_path else None
        resolved = validate_config(path=target)
        print(f"Config validated: {resolved}")
        return 0
    except QuizzerConfigError as exc:
        print(f"Validation failed: {exc}")
        return 1


def _cmd_config_path(args: argparse.Namespace) -> int:
    from .config import _resolve_config_path

    config_path = getattr(args, "config", None)
    resolved = _resolve_config_path(
        config_path=Path(config_path) if config_path else None,
        env={CONFIG_ENV: os.environ.get(CONFIG_ENV, "")},
    )
    if resolved:
        print(resolved.resolve())
        return 0
    else:
        print("No config file found.")
        return 1


def _cmd_not_implemented(label: str) -> int:
    print(f"{label} is not implemented yet. Stay tuned.")
    return 2


def _cmd_questions_generate(args: argparse.Namespace) -> int:
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found. Run 'quizzer init <name>' first.")
        return 2
    cfg = _load_toml(cfg_path)
    try:
        section = _get_quiz_section(cfg, args.name)
    except KeyError as exc:
        print(f"Error: {exc}")
        return 2
    out_dir = _out_dir_for(args.name, cfg)
    topics_path = out_dir / "topics.jsonl"
    if not topics_path.exists():
        print(
            f"No topics found at {topics_path}. Run 'quizzer topics generate "
            f"{args.name}'."
        )
        return 1
    topics = read_jsonl(topics_path)
    per_topic = (
        int(args.per_topic)
        if args.per_topic is not None
        else int(section.get("per_topic", 3))
    )
    ensure_coverage = (
        bool(args.ensure_coverage)
        if hasattr(args, "ensure_coverage")
        else bool(section.get("ensure_coverage", True))
    )
    client = None
    if load_client is not None:
        try:
            cfg = load_config()
            client = load_client(
                local=cfg.ai.use_local,
                api_base=cfg.ai.api_base,
            )
        except Exception:
            client = None
    questions = generate_questions(
        topics,
        per_topic=per_topic,
        client=client,
        ensure_coverage=ensure_coverage,
    )
    if not questions:
        print("No questions generated.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / "questions.jsonl"
    write_jsonl(q_path, questions)
    print(f"Wrote {len(questions)} question(s) -> {q_path}")
    return 0


def _cmd_questions_list(args: argparse.Namespace) -> int:
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found.")
        return 2
    cfg = _load_toml(cfg_path)
    out_dir = _out_dir_for(args.name, cfg)
    q_path = out_dir / "questions.jsonl"
    if not q_path.exists():
        print(
            f"No questions found at {q_path}. Run 'quizzer questions generate "
            f"{args.name}'."
        )
        return 1
    questions = read_jsonl(q_path)
    topics_filter = set(args.topics or [])
    shown = 0
    for q in questions:
        t = str(q.get("topic_id", ""))
        if topics_filter and t not in topics_filter:
            continue
        stem = str(q.get("stem", ""))
        print(f"[{t}] {stem[:100]}")
        shown += 1
    if shown == 0:
        print("No questions to show with given filters.")
        return 1
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    """Start a Rich-powered quiz session in the current terminal."""
    cfg_path = _find_config(getattr(args, "config", None))
    if not cfg_path:
        print("Error: quizzer.toml not found.")
        return 2
    cfg = _load_toml(cfg_path)
    out_dir = _out_dir_for(args.name, cfg)
    q_path = out_dir / "questions.jsonl"
    if not q_path.exists():
        print(
            f"No questions found at {q_path}. Run 'quizzer questions generate "
            f"{args.name}'."
        )
        return 1
    questions = read_jsonl(q_path)
    if not questions:
        print("Question bank is empty.")
        return 1
    # Shuffle if requested
    if getattr(args, "shuffle", False):
        rnd = random.Random()
        rnd.shuffle(questions)
    # Apply limit
    n = int(getattr(args, "num", 0) or 0)
    if n > 0:
        questions = questions[:n]

    mix_mode = getattr(args, "mix", "")
    if mix_mode and mix_mode != "balanced":
        print("Warning: --mix is not implemented yet; ignoring selection.")

    if getattr(args, "resume", False):
        print(
            "Warning: --resume is not implemented yet; "
            "starting a fresh session."
        )

    console = Console()
    run_quiz_session(
        questions,
        console=console,
        input_provider=input,
        show_explanations=bool(getattr(args, "explain", True)),
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quizzer",
        description="Interactive quiz tool for Markdown sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp_init = sub.add_parser(
        "init", help="Create quizzer.toml template for a quiz"
    )
    sp_init.add_argument("name")

    sp_topics = sub.add_parser("topics", help="Topic-related commands")
    topics_sub = sp_topics.add_subparsers(dest="action", required=True)
    sp_t_gen = topics_sub.add_parser(
        "generate", help="Extract and persist topics"
    )
    sp_t_gen.add_argument("name")
    sp_t_gen.add_argument("--limit", type=int)
    sp_t_gen.add_argument(
        "--extensions",
        nargs="+",
        default=["md", "markdown"],
        help="File extensions to include for discovery",
    )
    sp_t_gen.add_argument(
        "--level-limit",
        type=int,
        default=0,
        help="Directory depth limit (0 = no limit)",
    )
    sp_t_gen.add_argument(
        "--use-ai",
        action="store_true",
        help="Use AI to assist topic extraction",
    )
    sp_t_list = topics_sub.add_parser("list", help="List topics")
    sp_t_list.add_argument("name")
    sp_t_list.add_argument("--filter")

    sp_q = sub.add_parser("questions", help="Question-related commands")
    q_sub = sp_q.add_subparsers(dest="action", required=True)
    sp_q_gen = q_sub.add_parser("generate", help="Generate questions")
    sp_q_gen.add_argument("name")
    sp_q_gen.add_argument("--per-topic", type=int)
    sp_q_gen.add_argument(
        "--ensure-coverage", dest="ensure_coverage", action="store_true"
    )
    sp_q_gen.add_argument(
        "--no-ensure-coverage", dest="ensure_coverage", action="store_false"
    )
    sp_q_gen.set_defaults(ensure_coverage=True)
    sp_q_list = q_sub.add_parser("list", help="List questions")
    sp_q_list.add_argument("name")
    sp_q_list.add_argument("--topics", nargs="*")

    sp_start = sub.add_parser("start", help="Start a quiz session")
    sp_start.add_argument("name")
    sp_start.add_argument("--num", type=int, default=10)
    sp_start.add_argument(
        "--mix", choices=["balanced", "random", "weakness"], default="balanced"
    )
    sp_start.add_argument("--resume", action="store_true")
    sp_start.add_argument("--shuffle", action="store_true")
    sp_start.add_argument("--explain", dest="explain", action="store_true")
    sp_start.add_argument("--no-explain", dest="explain", action="store_false")
    sp_start.set_defaults(explain=True)

    sp_rev = sub.add_parser(
        "review", help="Re-quiz wrong or weak topics from a session"
    )
    sp_rev.add_argument("name")
    sp_rev.add_argument("--session")
    sp_rep = sub.add_parser("report", help="Show session summary")
    sp_rep.add_argument("name")
    sp_rep.add_argument("--session")

    # Config subcommand group
    sp_config = sub.add_parser("config", help="Config management commands")
    config_sub = sp_config.add_subparsers(dest="action", required=True)

    sp_c_init = config_sub.add_parser(
        "init", help="Create a quizzer.toml template"
    )
    sp_c_init.add_argument(
        "--path",
        default=None,
        help="Output path (default: workspace/config/quizzer.toml)",
    )
    sp_c_init.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing file",
    )

    sp_c_validate = config_sub.add_parser(
        "validate", help="Validate current config file"
    )
    sp_c_validate.add_argument(
        "--config",
        default=None,
        help="Explicit path to validate",
    )

    sp_c_path = config_sub.add_parser(
        "path", help="Print resolved config file path"
    )
    sp_c_path.add_argument(
        "--config",
        default=None,
        help="Explicit path to resolve",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate command handler."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    code = _dispatch_command(args, parser)
    raise SystemExit(code)


def _dispatch_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Route parsed args to the correct command handler."""

    if args.command == "config":
        return _handle_config(args)

    if args.command == "init":
        return _cmd_init(args)

    is_topics = args.command == "topics" and args.action in ("generate", "list")
    if is_topics:
        handler = (
            _cmd_topics_generate
            if args.action == "generate"
            else _cmd_topics_list
        )
        return handler(args)

    questions_cmd = args.command == "questions" and args.action in (
        "generate",
        "list",
    )
    if questions_cmd:
        handler = (
            _cmd_questions_generate
            if args.action == "generate"
            else _cmd_questions_list
        )
        return handler(args)

    if args.command == "start":
        return _cmd_start(args)
    if args.command == "review":
        return _cmd_not_implemented("review")
    if args.command == "report":
        return _cmd_not_implemented("report")

    parser.print_help()
    return 2


def _handle_config(args: argparse.Namespace) -> int:
    """Handle config subcommands (init, validate, path)."""
    action = args.action or "init"

    handlers = {
        "init": _cmd_config_init,
        "validate": _cmd_config_validate,
        "path": _cmd_config_path,
    }

    handler = handlers.get(action)
    if handler is None:
        # Print config subparser help for unknown actions
        print(f"Unknown config action: {action}")
        return 2
    return handler(args)
