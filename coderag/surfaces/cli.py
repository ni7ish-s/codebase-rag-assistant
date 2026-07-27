"""The ``coderag`` command — index, search, watch, serve, ui, status.

Every subcommand is a thin adapter over :class:`coderag.api.CodeRAG`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

from coderag import __version__
from coderag.api import CodeRAG
from coderag.config import Config


def _build_config(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if getattr(args, "watched_dir", None):
        overrides["watched_dir"] = Path(args.watched_dir).expanduser()
    if getattr(args, "store_dir", None):
        overrides["store_dir"] = Path(args.store_dir).expanduser()
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "use_gitignore", None) is not None:
        overrides["use_gitignore"] = args.use_gitignore
    return Config.from_env(**overrides)


# --- commands ---


def cmd_index(args: argparse.Namespace) -> int:
    cr = CodeRAG(_build_config(args))
    if not args.quiet:
        # The provider/model loads on first index access; on a fresh install that is a
        # one-off model download. Say so on stderr, so the wait isn't a mystery.
        print(
            f"Preparing to index {cr.config.watched_dir} "
            "(first run downloads the embedding model)…",
            file=sys.stderr,
            flush=True,
        )
    started = time.monotonic()
    stats = cr.indexer.index(
        Path(args.path).expanduser() if args.path else None,
        full=args.full,
        progress=not args.quiet,
    )
    print(
        f"Indexed {stats.files_indexed} file(s), skipped {stats.files_skipped}, "
        f"removed {stats.files_removed} in {time.monotonic() - started:.1f}s. "
        f"Total: {stats.total_files} files / {stats.total_chunks} chunks."
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cr = CodeRAG(_build_config(args))
    hits = cr.search(args.query, top_k=args.k)
    if args.json:
        print(json.dumps([h.as_dict() for h in hits], indent=2))
        return 0 if hits else 1
    if not hits:
        print("No results. Has the codebase been indexed? Try: coderag index")
        return 1
    for i, h in enumerate(hits, 1):
        label = f" ({h.symbol})" if h.symbol else ""
        snippet = textwrap.shorten(
            h.text.replace("\n", " "), width=160, placeholder=" …"
        )
        print(f"{i}. {h.location}{label}  [{h.kind}, sim={h.similarity:.2f}]")
        print(f"   {snippet}")
    if args.answer:
        _print_answer(cr, args.query, args.k)
    return 0


def _print_answer(cr: CodeRAG, query: str, k: int) -> None:
    from coderag.llm import stream_answer

    print("\n--- Answer ---")
    try:
        for token in stream_answer(cr, query, k):
            sys.stdout.write(token)
            sys.stdout.flush()
        print()
    except RuntimeError as exc:
        print(f"(LLM answer unavailable: {exc})")


def cmd_status(args: argparse.Namespace) -> int:
    cr = CodeRAG(_build_config(args))
    print(json.dumps(cr.status(), indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from coderag import eval as ev

    cfg = _build_config(args)

    # `coderag eval --list-models` — show recommended local embedding models.
    if args.list_models:
        from coderag.embeddings.models import format_models

        print(format_models())
        return 0

    # `coderag eval build` — mine a dataset from the repo's git history.
    if args.build:
        from coderag.chunking.languages import extensions_for

        cases = ev.build_from_git(
            cfg.watched_dir,
            max_cases=args.max_cases,
            extensions=extensions_for(cfg.languages),
            symbols=args.level == "symbol",
        )
        out = args.dataset or "coderag-eval.jsonl"
        ev.save_dataset(cases, out)
        print(f"Wrote {len(cases)} eval case(s) to {out}")
        return 0 if cases else 1

    if not args.dataset:
        print("Provide --dataset PATH (or --build to mine one from git history).")
        return 1
    cases = ev.load_dataset(args.dataset)
    if not cases:
        print(f"No eval cases in {args.dataset}.")
        return 1

    ks = tuple(int(k) for k in args.ks.split(","))
    # --rerank / --adaptive / --graph force the optional stages on for this run.
    if args.rerank:
        cfg = cfg.with_overrides(rerank=True)
    if args.adaptive:
        cfg = cfg.with_overrides(adaptive_fusion=True)
    if args.graph:
        cfg = cfg.with_overrides(graph_expansion=True)
    cr = CodeRAG(cfg)
    cr.index()  # ensure the index is built / up to date before scoring

    if args.compare:
        reranker = None
        if args.rerank:
            from coderag.retrieval.rerank import get_reranker

            reranker = get_reranker(cfg)
        results = ev.compare_modes(
            cr,
            cases,
            ks=ks,
            level=args.level,
            reranker=reranker,
            adaptive=args.adaptive,
            graph=args.graph,
        )
    else:
        parts = ["hybrid"]
        if args.adaptive:
            parts = ["adaptive"]
        if args.graph:
            parts.append("graph")
        if args.rerank:
            parts.append("rerank")
        results = [
            ev.evaluate(
                cr.search, cases, label="+".join(parts), ks=ks, level=args.level
            )
        ]

    if args.json:
        print(json.dumps([r.as_dict() for r in results], indent=2))
    else:
        from coderag.eval.harness import format_table

        print(f"Eval: {len(cases)} case(s), level={args.level}\n")
        print(format_table(results))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    from coderag.watch import watch

    cr = CodeRAG(_build_config(args))
    print(f"Indexing {cr.config.watched_dir} before watching...")
    cr.indexer.index(progress=not args.quiet)
    watch(cr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from coderag.surfaces.http_api import run_server
    except ImportError:
        print(
            "The HTTP server needs extra deps. Install with: pip install 'coderag[server]'"
        )
        return 1
    cr = CodeRAG(_build_config(args))
    run_server(cr, host=args.host, port=args.port)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from coderag.surfaces.mcp_server import run_mcp
    except ImportError:
        print(
            "The MCP server needs extra deps. Install with: pip install 'coderag[mcp]'"
        )
        return 1
    cfg = _build_config(args)
    if args.all_text:
        cfg = cfg.with_overrides(index_all_text=True)
    cr = CodeRAG(cfg)
    run_mcp(
        cr,
        transport=args.transport,
        auto_index=not args.no_index,
        watch=not args.no_watch,
    )
    return 0


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower().startswith("y")


_NEXT_STEPS = {
    "claude": "Restart Claude Code (or run `claude mcp list`) to load coderag.",
    "hermes": "Restart Hermes (or run `hermes mcp list`) to load coderag.",
    "codex": "Restart Codex to load the coderag MCP server.",
}


def cmd_install(args: argparse.Namespace) -> int:
    """Register CodeRAG's MCP server in an AI agent (Claude Code, Hermes, Codex)."""
    from coderag import install as inst

    default_watched = (
        Path(args.watched_dir).expanduser()
        if args.watched_dir
        else inst.default_workspace()
    )
    explicit_watched = Path(args.watched_dir).expanduser() if args.watched_dir else None
    interactive = sys.stdin.isatty()

    # Bare `coderag install` on a terminal → the friendly wizard; otherwise the stable
    # auto-detect default (and never prompt when there is no TTY, e.g. in CI).
    use_wizard = args.wizard or (
        args.target is None and not args.yes and not args.print and interactive
    )
    if use_wizard:
        if not interactive:
            print("The wizard needs an interactive terminal. Pass a target or --yes.")
            return 1
        plans = inst.run_wizard(inst.detect_targets(), default_watched)
    else:
        targets = [args.target] if args.target else inst.detect_targets()
        if not targets:
            print(
                "No supported agents detected. Pass a target (claude|hermes|codex) "
                "or run `coderag install --wizard`."
            )
            return 1
        plans = [
            inst.Plan(target=t, watched_dir=explicit_watched, scope=args.scope)
            for t in targets
        ]

    # Always preview (dry-run) first.
    previews = [
        (
            plan,
            inst.install(
                plan.target,
                watched_dir=plan.watched_dir,
                scope=plan.scope,
                tools=plan.tools,
                dry_run=True,
            ),
        )
        for plan in plans
    ]
    print()
    for _plan, r in previews:
        print(f"== {r.target}  ({r.path})  [{r.action}]")
        if r.detail:
            print(textwrap.indent(r.detail.rstrip(), "   "))
        print()

    if args.print:
        return 0
    if not args.yes and interactive and not _confirm("Apply these changes?"):
        print("Aborted.")
        return 1

    final = [
        inst.install(
            plan.target,
            watched_dir=plan.watched_dir,
            scope=plan.scope,
            tools=plan.tools,
            dry_run=False,
        )
        for plan, _ in previews
    ]
    print()
    for r in final:
        line = f"  [{r.action}] {r.target}: {r.path}"
        if r.action in ("manual", "error") and r.detail:
            line += f"\n      {r.detail.splitlines()[0]}"
        print(line)
    steps = {_NEXT_STEPS[r.target] for r in final if r.target in _NEXT_STEPS}
    if steps:
        print("\nNext steps:")
        for s in sorted(steps):
            print(f"  - {s}")

    wd = next((p.watched_dir for p in plans if p.watched_dir is not None), Path.cwd())
    print("\nHow indexing works:")
    print(
        "  CodeRAG indexes your workspace the first time the agent starts its server. It\n"
        "  runs in the background, so search works right away and fills in as it goes —\n"
        "  seconds for a repo. Large trees (a whole home/system) are supported too; the\n"
        "  first pass just takes longer. It skips version-control, build, and dependency\n"
        "  directories automatically (see `coderag index --help` for throughput options)."
    )
    print("\nHandy commands:")
    print(f"  coderag status --watched-dir {wd}   # totals + where the index lives")
    print(
        f"  coderag index  --watched-dir {wd}   # build/refresh it now, with progress"
    )
    return 0 if all(r.action != "error" for r in final) else 1


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from coderag.surfaces.webui import run_ui
    except ImportError:
        print("The web UI needs extra deps. Install with: pip install 'coderag[ui]'")
        return 1
    cr = CodeRAG(_build_config(args))
    host = args.host or os.getenv("CODERAG_UI_HOST") or "127.0.0.1"
    port = args.port if args.port is not None else _env_port("CODERAG_UI_PORT", 8501)
    run_ui(cr, host=host, port=port)
    return 0


def _env_port(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw and raw.isdigit() else default


# --- parser ---


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--watched-dir", help="Codebase root to index/search.")
    p.add_argument(
        "--store-dir",
        help="Where the index/database live (default <watched-dir>/.coderag).",
    )
    p.add_argument(
        "--provider",
        help="Embedding provider: fastembed (local, default) | openai (OpenAI API or "
        "any OpenAI-compatible/local server via OPENAI_BASE_URL) | fake.",
    )
    p.add_argument("--model", help="Embedding model name.")
    p.add_argument(
        "--gitignore",
        dest="use_gitignore",
        action="store_true",
        default=None,
        help="Honor .gitignore files while indexing/searching (default).",
    )
    p.add_argument(
        "--no-gitignore",
        dest="use_gitignore",
        action="store_false",
        help="Do not honor .gitignore files.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coderag",
        description="Standalone, local-first semantic code-search engine.",
    )
    parser.add_argument("--version", action="version", version=f"coderag {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser(
        "index", help="Index (or incrementally update) a codebase."
    )
    p_index.add_argument(
        "path", nargs="?", help="Path to index (defaults to watched dir)."
    )
    p_index.add_argument("--full", action="store_true", help="Force a clean rebuild.")
    p_index.add_argument("--quiet", action="store_true", help="Hide the progress bar.")
    _add_common(p_index)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search the indexed codebase.")
    p_search.add_argument("query", help="What to search for.")
    p_search.add_argument(
        "-k", type=int, default=8, help="Number of results (default 8)."
    )
    p_search.add_argument("--json", action="store_true", help="Emit JSON.")
    p_search.add_argument(
        "--answer",
        action="store_true",
        help="Also stream a grounded LLM answer. Works with a local model (set "
        "OPENAI_BASE_URL to an Ollama/LM Studio/vLLM server), OpenAI, or Anthropic. "
        "See docs/configuration.md.",
    )
    _add_common(p_search)
    p_search.set_defaults(func=cmd_search)

    p_status = sub.add_parser("status", help="Show index statistics.")
    _add_common(p_status)
    p_status.set_defaults(func=cmd_status)

    p_eval = sub.add_parser(
        "eval",
        help="Measure retrieval quality against a dataset (recall@k, MRR, nDCG).",
    )
    p_eval.add_argument(
        "--dataset", help="JSONL dataset of query -> relevant files/symbols."
    )
    p_eval.add_argument(
        "--build",
        action="store_true",
        help="Mine a dataset from git history into --dataset (default coderag-eval.jsonl).",
    )
    p_eval.add_argument(
        "--max-cases",
        type=int,
        default=200,
        help="Cap cases when building (default 200).",
    )
    p_eval.add_argument(
        "--compare",
        action="store_true",
        help="Score dense-only vs BM25-only vs hybrid on one index.",
    )
    p_eval.add_argument(
        "--level",
        choices=("file", "symbol"),
        default="file",
        help="Localization granularity (default file).",
    )
    p_eval.add_argument(
        "--ks", default="1,5,10", help="Comma-separated cutoffs (default 1,5,10)."
    )
    p_eval.add_argument("--json", action="store_true", help="Emit JSON.")
    p_eval.add_argument("--quiet", action="store_true", help="Hide the progress bar.")
    p_eval.add_argument(
        "--rerank",
        action="store_true",
        help="Enable the local cross-encoder reranker (two-stage retrieve-then-rerank).",
    )
    p_eval.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable query-type-aware fusion weighting (dense-up for NL, BM25-up for code).",
    )
    p_eval.add_argument(
        "--graph",
        action="store_true",
        help="Enable structure-aware 1-hop symbol-graph neighbor expansion.",
    )
    p_eval.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended local embedding models for code search and exit.",
    )
    _add_common(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_watch = sub.add_parser(
        "watch", help="Index, then keep the index live on changes."
    )
    p_watch.add_argument("--quiet", action="store_true")
    _add_common(p_watch)
    p_watch.set_defaults(func=cmd_watch)

    p_serve = sub.add_parser("serve", help="Run the HTTP/REST API server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    _add_common(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_mcp = sub.add_parser(
        "mcp",
        help="Run the MCP server so AI agents (Claude Code/Codex/Cursor) can search "
        "this workspace instead of grepping.",
    )
    p_mcp.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport (default stdio — how editors/agents launch servers).",
    )
    p_mcp.add_argument(
        "--no-index",
        action="store_true",
        help="Don't index the workspace on startup; use the existing index as-is.",
    )
    p_mcp.add_argument(
        "--no-watch",
        action="store_true",
        help="Don't keep the index live with the filesystem watcher.",
    )
    p_mcp.add_argument(
        "--all-text",
        action="store_true",
        help="Index any text file, not just code (docs/notes/config) — for a plain "
        "file directory, not only a code repo.",
    )
    _add_common(p_mcp)
    p_mcp.set_defaults(func=cmd_mcp)

    p_install = sub.add_parser(
        "install",
        help="Register CodeRAG's MCP server in an AI agent (Claude Code, Hermes, Codex) "
        "— one command instead of hand-editing config.",
    )
    p_install.add_argument(
        "target",
        nargs="?",
        choices=("claude", "hermes", "codex"),
        help="Agent to install for. Omit to auto-detect (or launch the wizard on a TTY).",
    )
    p_install.add_argument(
        "--wizard", action="store_true", help="Interactive guided install."
    )
    p_install.add_argument(
        "--print",
        dest="print",
        action="store_true",
        help="Preview the config changes without writing anything (dry-run).",
    )
    p_install.add_argument(
        "--yes",
        action="store_true",
        help="Apply without the confirmation prompt (non-interactive).",
    )
    p_install.add_argument(
        "--scope",
        choices=("user", "project"),
        default="project",
        help="Config scope where applicable (default project).",
    )
    _add_common(p_install)
    p_install.set_defaults(func=cmd_install)

    p_ui = sub.add_parser("ui", help="Launch the built-in web UI.")
    p_ui.add_argument(
        "--host",
        default=None,
        help="Bind address (default 127.0.0.1 / CODERAG_UI_HOST).",
    )
    p_ui.add_argument(
        "--port", type=int, default=None, help="Port (default 8501 / CODERAG_UI_PORT)."
    )
    _add_common(p_ui)
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
