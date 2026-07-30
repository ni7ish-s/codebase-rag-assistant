"""Typed, injectable configuration for CodeRAG.

The whole app reads configuration from a single immutable :class:`Config` object that
is built once (usually via :meth:`Config.from_env`) and passed down explicitly. Nothing
deep in the call stack reaches for ``os.environ`` — that keeps the engine testable and
free of import-time side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# Languages that ship with symbol-aware chunking in v1.0. Anything not listed (or that
# fails to parse) falls back to the line-window chunker.
DEFAULT_LANGUAGES: Tuple[str, ...] = (
    "python",
    "javascript",
    "typescript",
    "tsx",
    "go",
    "rust",
    "java",
)

# Directories/globs never worth indexing. Note we deliberately do NOT ignore ``tests`` —
# people search their tests too. The dependency/cache entries matter most at home/system
# scale (e.g. indexing ``/home``), where they are the bulk of the file count; each
# ``<name>/*`` entry prunes that directory wholesale anywhere in the tree.
DEFAULT_IGNORE_GLOBS: Tuple[str, ...] = (
    # VCS
    ".git/*",
    ".hg/*",
    ".svn/*",
    # build / packaging output
    "build/*",
    "dist/*",
    "target/*",
    "*.egg-info/*",
    ".next/*",
    ".nuxt/*",
    # language / tool caches
    "__pycache__/*",
    ".mypy_cache/*",
    ".pytest_cache/*",
    ".ruff_cache/*",
    ".tox/*",
    ".ipynb_checkpoints/*",
    ".gradle/*",
    ".terraform/*",
    ".coderag/*",
    # virtualenvs / vendored dependencies
    ".venv/*",
    "venv/*",
    "env/*",
    "node_modules/*",
    "site-packages/*",
    "vendor/*",
    # user/home caches (dominant at /home scale)
    ".cache/*",
    ".local/*",
    ".npm/*",
    ".cargo/*",
    ".rustup/*",
    ".m2/*",
    ".nuget/*",
    # editor metadata
    ".idea/*",
    ".vscode/*",
)


def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val.strip() else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return Path(raw).expanduser()


def _env_path_opt(key: str) -> Path | None:
    """Like :func:`_env_path` but returns ``None`` when the var is unset/blank.

    Used for ``store_dir`` so an absent ``CODERAG_STORE_DIR`` leaves it unset and lets it be
    derived from ``watched_dir`` (see :meth:`Config.__post_init__`) rather than the cwd.
    """
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser()


def _env_tuple(key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """Parse a comma-separated env var into a tuple of trimmed, non-empty values."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Config:
    """Immutable configuration for an indexing/search session."""

    # --- Embedding provider ---
    provider: str = "sentence-transformers"  # "sentence-transformers" | "openai" | "fake"
    model: str = "BAAI/bge-small-en-v1.5"
    openai_model: str = "text-embedding-3-small"
    # Secret: kept out of repr() so it can't leak into logs/tracebacks (see M1).
    openai_api_key: str | None = field(default=None, repr=False)
    # Point the OpenAI-compatible client at a self-hosted / local server (Ollama, vLLM,
    # LM Studio, LocalAI, text-embeddings-inference, …). Used for both embeddings and the
    # ``openai`` answer backend. When set, an API key is optional.
    openai_base_url: str | None = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "coderag")

    # --- Locations ---
    watched_dir: Path = field(default_factory=Path.cwd)
    # Defaults to ``<watched_dir>/.coderag`` (resolved in ``__post_init__``), so the index
    # lives next to the code it indexes. It deliberately does NOT default to the *current
    # working directory*: a cwd-relative store silently pointed at a different (often empty)
    # ``.coderag`` whenever a command ran from a directory other than the one indexed — e.g.
    # ``coderag index --watched-dir /home/me`` would write to ``/home/me/.coderag`` only if
    # you happened to be standing in ``/home/me``, and ``coderag status`` from elsewhere then
    # found nothing. Set it explicitly with ``--store-dir`` / ``CODERAG_STORE_DIR`` to override.
    store_dir: Path = None  # type: ignore[assignment]  # filled in by __post_init__

    # --- What to index ---
    languages: Tuple[str, ...] = DEFAULT_LANGUAGES
    ignore_globs: Tuple[str, ...] = DEFAULT_IGNORE_GLOBS
    # Honor .gitignore files while walking (in addition to ignore_globs), so a repo's own
    # build/output exclusions are respected. On by default; disable with
    # CODERAG_GITIGNORE=0 or `--no-gitignore`.
    use_gitignore: bool = True
    # Index any UTF-8-decodable file as plain text, even with an unknown/absent extension
    # (Dockerfile, Makefile, LICENSE, .log, ...). Off by default so code repos aren't
    # polluted; turn on (CODERAG_INDEX_ALL_TEXT / `coderag mcp --all-text`) to make
    # CodeRAG a general document/file-directory search engine. Binary files are still
    # skipped (NUL-byte sniff in the indexer).
    index_all_text: bool = False
    max_file_bytes: int = 1_000_000  # skip files larger than this
    max_chunk_lines: int = 200  # split oversized symbols into windows above this
    window_lines: int = 60  # fallback line-window size
    window_overlap: int = 10

    # --- Retrieval ---
    top_k: int = 8
    fetch_k: int = 50  # candidates pulled from each retriever before fusion
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    # --- Adaptive fusion weighting (query-type-aware) ---
    # Off by default. When on, fusion weights tilt by query type: dense up for
    # natural-language queries, BM25 up for exact-identifier/code queries (a fixed 1:1 is a
    # compromise — see docs/eval.md). These pairs override dense_weight/lexical_weight only
    # when adaptive_fusion is enabled.
    adaptive_fusion: bool = False
    # NL queries: lean dense (weak BM25 otherwise drags a strong dense ranking down).
    nl_dense_weight: float = 1.0
    nl_lexical_weight: float = 0.4
    # Identifier/code queries: stay balanced. Up-weighting BM25 here *hurt* on this repo
    # (short, common identifiers are lexically ambiguous, and the embedder already matches
    # them well) — so the default is neutral, and BM25-leaning is left configurable for
    # larger repos where exact-string recall matters more. See docs/eval.md.
    code_dense_weight: float = 1.0
    code_lexical_weight: float = 1.0

    # --- Structure-aware 1-hop neighbor expansion (optional, opt-in) ---
    # Off by default. When on, the top ``graph_seeds`` fused hits are expanded with their
    # 1-hop call-graph neighbors — the definitions of what each seed *calls* (its callees),
    # resolved from the seed's text via the store's symbol index (no schema change). The
    # neighbors are fused back in at ``graph_weight``, down-weighted so they add recall
    # without overpowering the direct signal. A 3-repo symbol-level eval (flask/requests/
    # click) showed a small, consistent lift at low weight; the opposite edge (a symbol's
    # callers) *hurt*. See docs/eval.md and docs/research/code-retrieval-strategy.md (§4).
    graph_expansion: bool = False
    graph_seeds: int = 5  # how many top fused hits to expand from
    graph_neighbors: int = 5  # max new callee definitions pulled per seed
    # RRF weight of the neighbor list relative to dense/lexical. 0.15 was a strict Pareto
    # improvement across the eval repos; higher trades rank precision for marginal recall.
    graph_weight: float = 0.15

    # --- Reranking (optional two-stage retrieve-then-rerank) ---
    # Off by default so the zero-config engine stays tiny/fast. When on, the top
    # ``rerank_candidates`` fused hits are re-scored by a local cross-encoder and reordered.
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # sentence-transformers cross-encoder
    rerank_candidates: int = 50  # fused hits to rerank before trimming to top_k

    # --- Indexing throughput ---
    embed_batch_size: int = 64
    index_workers: int = 4
   # Embedding device for the local (sentence-transformers) provider. "auto" uses a CUDA GPU when
    # onnxruntime exposes one (10-50x faster indexing) and falls back to CPU otherwise;
    # "cpu"/"cuda" force it. A missing/broken GPU always degrades to CPU rather than failing.
    embed_device: str = "auto"  # "auto" | "cpu" | "cuda"
    embed_threads: int = 0  # ONNX CPU threads (0 => library default)

    # --- Optional LLM answer surface ---
    # Which backend turns retrieved chunks into a grounded answer.
    llm_provider: str = "openai"  # "openai" | "anthropic"
    # OpenAI (or self-hosted OpenAI-compatible) chat model used for answers.
    chat_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = field(default=None, repr=False)  # secret: no repr
    anthropic_model: str = "claude-opus-4-8"
    answer_max_tokens: int = 1024

    # --- HTTP API server (optional [server] surface) ---
    # When set, the HTTP API requires this key on every request (Authorization:
    # ``Bearer <key>`` or the ``X-API-Key`` header). Unset => no auth, which is only
    # safe for trusted, loopback-only local use. ALWAYS set this (via CODERAG_API_KEY)
    # when binding to a non-loopback host, e.g. in the container/Helm deployment.
    api_key: str | None = field(default=None, repr=False)  # secret: no repr
    # Explicit CORS allowlist for the HTTP API. Empty => no cross-origin browser
    # access. Never use "*": the API can read indexed source and file contents, so a
    # wildcard lets any website the user visits exfiltrate them via the browser.
    cors_origins: Tuple[str, ...] = ()

    # --- MCP server surface (optional [mcp] surface) ---
    # `coderag mcp` runs a persistent, warm process so the embedding model loads once and
    # every query is fast — the win over an agent's cold, repeated grep/read loop. By
    # default it indexes the watched dir on startup (in the background, so the server is
    # responsive immediately) and keeps it live via the filesystem watcher, so an agent
    # gets fresh results with zero manual steps.
    mcp_auto_index: bool = True
    mcp_watch: bool = True
    # Lines of a chunk returned in a search_code snippet by default (the agent can request
    # the full text, or fetch a precise range via get_file) — keeps responses token-cheap.
    mcp_snippet_lines: int = 12

    # --- Demo mode (public, untrusted UI) ---
    # When on, the Streamlit UI shows a notice, hides the Reindex button, and limits
    # LLM answers per browser session. The per-session limit is soft (session-state
    # based); pair it with an Ollama concurrency cap (OLLAMA_NUM_PARALLEL) for a hard
    # GPU backstop. Also lower CODERAG_ANSWER_MAX_TOKENS to keep each answer cheap.
    demo_mode: bool = False
    demo_max_answers: int = 5  # LLM answers allowed per browser session
    demo_cooldown_seconds: int = 20  # minimum seconds between answers in a session

    def __post_init__(self) -> None:
        # Derive the store location from the watched dir when it wasn't set explicitly, so the
        # index is found regardless of the current working directory. Runs on every
        # construction — including the ``replace()`` inside ``with_overrides`` — so overriding
        # ``watched_dir`` without a ``store_dir`` re-derives the store from the new watched dir.
        if self.store_dir is None:
            # Frozen dataclass: assign through object.__setattr__.
            object.__setattr__(self, "store_dir", self.watched_dir / ".coderag")

    def with_overrides(self, **kwargs: object) -> "Config":
        """Return a copy with the given fields replaced (config stays immutable).

        Overriding ``watched_dir`` alone also moves the store: ``replace`` re-runs
        ``__post_init__``, but only when ``store_dir`` is left unset here. To keep an
        auto-derived store from following a new watched dir, pass ``store_dir`` explicitly.
        """
        # If watched_dir is being moved but store_dir isn't given *and* the current store_dir
        # was auto-derived from the old watched_dir, clear it so __post_init__ re-derives it
        # against the new watched_dir.
        if (
            "watched_dir" in kwargs
            and "store_dir" not in kwargs
            and self.store_dir == self.watched_dir / ".coderag"
        ):
            kwargs = {**kwargs, "store_dir": None}
        return replace(self, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        """Build a Config from environment / .env, applying explicit overrides last."""
        load_dotenv()
        base = cls(
            provider=_env_str("CODERAG_PROVIDER", cls.provider),
            model=_env_str("CODERAG_MODEL", cls.model),
            openai_model=_env_str("CODERAG_OPENAI_MODEL", cls.openai_model),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            cache_dir=_env_path(
                "CODERAG_CACHE_DIR", Path.home() / ".cache" / "coderag"
            ),
            watched_dir=_env_path("CODERAG_WATCHED_DIR", Path.cwd()),
            # None => derived from watched_dir in __post_init__ (never the cwd).
            store_dir=_env_path_opt("CODERAG_STORE_DIR"),  # type: ignore[arg-type]
            top_k=_env_int("CODERAG_TOP_K", cls.top_k),
            fetch_k=_env_int("CODERAG_FETCH_K", cls.fetch_k),
            rrf_k=_env_int("CODERAG_RRF_K", cls.rrf_k),
            dense_weight=_env_float("CODERAG_DENSE_WEIGHT", cls.dense_weight),
            lexical_weight=_env_float("CODERAG_LEXICAL_WEIGHT", cls.lexical_weight),
            adaptive_fusion=_env_bool("CODERAG_ADAPTIVE_FUSION", cls.adaptive_fusion),
            nl_dense_weight=_env_float("CODERAG_NL_DENSE_WEIGHT", cls.nl_dense_weight),
            nl_lexical_weight=_env_float(
                "CODERAG_NL_LEXICAL_WEIGHT", cls.nl_lexical_weight
            ),
            code_dense_weight=_env_float(
                "CODERAG_CODE_DENSE_WEIGHT", cls.code_dense_weight
            ),
            code_lexical_weight=_env_float(
                "CODERAG_CODE_LEXICAL_WEIGHT", cls.code_lexical_weight
            ),
            graph_expansion=_env_bool("CODERAG_GRAPH_EXPANSION", cls.graph_expansion),
            graph_seeds=_env_int("CODERAG_GRAPH_SEEDS", cls.graph_seeds),
            graph_neighbors=_env_int("CODERAG_GRAPH_NEIGHBORS", cls.graph_neighbors),
            graph_weight=_env_float("CODERAG_GRAPH_WEIGHT", cls.graph_weight),
            rerank=_env_bool("CODERAG_RERANK", cls.rerank),
            rerank_model=_env_str("CODERAG_RERANK_MODEL", cls.rerank_model),
            rerank_candidates=_env_int(
                "CODERAG_RERANK_CANDIDATES", cls.rerank_candidates
            ),
            embed_batch_size=_env_int("CODERAG_EMBED_BATCH", cls.embed_batch_size),
            index_workers=_env_int("CODERAG_WORKERS", cls.index_workers),
            embed_device=_env_str("CODERAG_EMBED_DEVICE", cls.embed_device),
            embed_threads=_env_int("CODERAG_EMBED_THREADS", cls.embed_threads),
            llm_provider=_env_str("CODERAG_LLM_PROVIDER", cls.llm_provider),
            chat_model=_env_str("CODERAG_CHAT_MODEL", cls.chat_model),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=_env_str("CODERAG_ANTHROPIC_MODEL", cls.anthropic_model),
            answer_max_tokens=_env_int(
                "CODERAG_ANSWER_MAX_TOKENS", cls.answer_max_tokens
            ),
            api_key=os.getenv("CODERAG_API_KEY"),
            cors_origins=_env_tuple("CODERAG_CORS_ORIGINS", cls.cors_origins),
            index_all_text=_env_bool("CODERAG_INDEX_ALL_TEXT", cls.index_all_text),
            # CODERAG_IGNORE_GLOBS *appends* extra excludes to the built-in defaults.
            ignore_globs=DEFAULT_IGNORE_GLOBS + _env_tuple("CODERAG_IGNORE_GLOBS", ()),
            use_gitignore=_env_bool("CODERAG_GITIGNORE", cls.use_gitignore),
            mcp_auto_index=_env_bool("CODERAG_MCP_AUTO_INDEX", cls.mcp_auto_index),
            mcp_watch=_env_bool("CODERAG_MCP_WATCH", cls.mcp_watch),
            mcp_snippet_lines=_env_int(
                "CODERAG_MCP_SNIPPET_LINES", cls.mcp_snippet_lines
            ),
            demo_mode=_env_bool("CODERAG_DEMO_MODE", cls.demo_mode),
            demo_max_answers=_env_int("CODERAG_DEMO_MAX_ANSWERS", cls.demo_max_answers),
            demo_cooldown_seconds=_env_int(
                "CODERAG_DEMO_COOLDOWN_SECONDS", cls.demo_cooldown_seconds
            ),
        )
        if overrides:
            base = base.with_overrides(**overrides)
        return base
