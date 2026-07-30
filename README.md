# Codebase RAG Assistant

A local-first RAG (Retrieval-Augmented Generation) system for asking natural-language questions about any codebase -- "Where is auth handled?", "How does X function work?" -- and getting accurate, cited answers grounded in the actual source.

## Features

- **AST-aware chunking** (tree-sitter) -- chunks code by function/class boundaries instead of naive line-splitting, preserving semantic context.
- **Hybrid retrieval** -- combines dense vector search with BM25 keyword search, fused via reciprocal rank fusion.
- **Cross-encoder reranking** -- a second-stage reranker reorders top candidates for higher precision.
- **Local LLM inference** -- answers generated via Ollama, no API key or cloud dependency required.
- **File + line-level citations** -- every answer points back to the exact file and line range it was grounded in.
- **CLI + FastAPI backend** -- usable as a command-line tool or served over HTTP.

## Tech Stack

Python, tree-sitter, Sentence-Transformers, ChromaDB, BM25 (rank_bm25), FastAPI, Ollama

## Architecture

Codebase is chunked (AST-aware via tree-sitter), embedded (sentence-transformers), and stored (ChromaDB).

A query runs through hybrid search (vector + BM25), gets reranked by a cross-encoder, and the top results are passed to a local LLM (Ollama) to generate a cited answer.

## Setup

```bash
conda create -n reporag python=3.11 -y
conda activate reporag
pip install -r requirements.txt

cp example.env .env
# edit .env: set REPORAG_WATCHED_DIR to the repo you want to index
ollama pull qwen2.5-coder:7b
```

## Usage

```bash
# Index a codebase
python -m reporag.surfaces.cli index

# Search
python -m reporag.surfaces.cli search "how does chunking work"

# Search + generate a cited answer
python -m reporag.surfaces.cli search "how does chunking work" --answer

# Check index status
python -m reporag.surfaces.cli status

# Run as an HTTP API
python -m reporag.surfaces.cli serve
```

## Screenshots

**Index status**

(screenshot here)

**Search with generated answer**

(screenshot here)

**FastAPI Swagger UI**

(screenshot here)

## Credits

Built on top of CodeRAG (https://github.com/Neverdecel/CodeRAG), used as a base RAG template and substantially modified.

## License

Apache 2.0 -- see LICENSE.