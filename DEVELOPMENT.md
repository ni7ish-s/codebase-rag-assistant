# 🛠️ Development Guide

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev,server,openai]"             # editable install + all extras
pre-commit install                                # optional: run quality checks on commit
```

No configuration is required to run — the default `fastembed` provider downloads a small
local model on first use. Copy `example.env` to `.env` only if you want to set defaults.

## Architecture

CodeRAG is one engine (`coderag.api.CodeRAG`) behind several surfaces. The facade wires
together four swappable pieces and constructs them lazily.

```
coderag/
├── api.py              # CodeRAG facade — the public entry point every surface uses
├── config.py           # Immutable Config dataclass (Config.from_env)
├── types.py            # Chunk, SearchHit, IndexStats
├── indexer.py          # Incremental indexing: hash-diff, delete-before-add, prune
├── watch.py            # Debounced filesystem watcher -> indexer
├── llm.py              # Optional streamed LLM answer over retrieved chunks
├── embeddings/         # EmbeddingProvider protocol + fastembed / openai / fake
├── chunking/           # Symbol-aware chunking: python_ast, treesitter, line-window base
├── store/              # Single embedded LanceDB store
│   └── lance_store.py  #   files/chunks + BM25 (FTS) + vectors (ANN) in one place
├── retrieval/          # Hybrid search: dense + BM25, fused with RRF
└── surfaces/           # cli.py · http_api.py (FastAPI) · webui.py · mcp_server.py (MCP)
```

### Design invariants (don't break these)

- **One LanceDB store holds everything** (chunk metadata, text/BM25, and vectors/ANN). It is
  rebuildable from source: re-indexing recreates it, and a `--full` pass clears and rebuilds.
- **`chunks.id` is a store-managed integer id** used as the fusion/hydrate key; ids are not
  reused within a run.
- **Delete-before-add.** A changed file's old rows are removed before new ones are added
  (`Indexer._write` → `LanceStore.write_file(replace=True)`), so editing never accumulates
  stale or duplicate rows.
- **The embedding dimension comes from the provider**, never a hard-coded constant. A model
  change is detected via the store's `meta.json` and clears the store for a clean re-index.
- **Writes serialize; reads don't block.** All indexing/deletion goes through one lock on the
  `CodeRAG` facade (`_index_lock`); the store buffers writes on the writer and reads query
  committed data — so the MCP server's background index and live watcher run safely alongside
  concurrent agent searches. Indexing may parallelize chunk+embed across `index_workers`
  threads, but the store writes stay single-writer (`Indexer._write`).

## Quality gate

The same commands CI runs:

```bash
pytest -m "not integration"   # fast & offline — uses the deterministic fake embedder
pytest -m integration         # exercises the real fastembed model (downloads once)
ruff check .                  # lint + import sort (replaces flake8 + isort)
ruff format --check .         # formatting (replaces black)
mypy coderag
```

CI additionally installs with [uv](https://github.com/astral-sh/uv), reports coverage
(`pytest --cov=coderag`) in the run summary, builds the package (`uv build` + `twine
check`), and runs the `integration` suite on a daily schedule.

Tests never hit the network or download a model unless marked `integration`. Use the
`config`/`repo`/`write` fixtures in `tests/conftest.py` (they default to the `fake` provider).

## Adding things

- **A new embedding backend:** implement the `EmbeddingProvider` protocol
  (`coderag/embeddings/__init__.py`) and wire it into `get_provider()`.
- **A new language:** add the extension in `chunking/languages.py`; for symbol-aware
  chunking, add a grammar + node types in `chunking/treesitter.py` (or rely on the
  line-window fallback).
- **A new surface:** keep it a thin adapter over `coderag.api.CodeRAG` — no engine logic in
  surfaces.

## Conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- ruff for format + lint + import sort (88-col code; line length is formatter-owned).
- Typed signatures and concise docstrings on public functions.
