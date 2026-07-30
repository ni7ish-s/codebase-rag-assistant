"""Curated registry of local (fastembed/ONNX) embedding models for code search.

These are the no-API-key models worth considering for CodeRAG, with short notes on the
accuracy/size trade-off. All are loadable via ``--model <name>`` (provider ``fastembed``).
The numbers in the notes are external benchmark figures (see docs/research/) — run
``reporag eval`` to measure them on *your* codebase.

Code-specific models (trained on code) generally beat general-purpose text embedders on
code retrieval, at the cost of a larger download.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelInfo:
    name: str  # fastembed model id (pass to --model)
    dim: int
    size_gb: float
    code_specific: bool
    note: str


# Ordered best-first for code search among models fastembed can load locally. fastembed
# does not (yet) ship CodeRankEmbed/CodeSage; those need a custom ONNX export — tracked as
# a follow-up. jina-embeddings-v2-base-code is the strongest code-specific option available
# out of the box.
RECOMMENDED: Tuple[ModelInfo, ...] = (
    ModelInfo(
        "jinaai/jina-embeddings-v2-base-code",
        768,
        0.64,
        True,
        "Code-specific, 8192-ctx, Apache-2.0. Best out-of-the-box local code retriever.",
    ),
    ModelInfo(
        "BAAI/bge-base-en-v1.5",
        768,
        0.21,
        False,
        "General text. Stronger than bge-small; modest code retrieval.",
    ),
    ModelInfo(
        "snowflake/snowflake-arctic-embed-m-long",
        768,
        0.54,
        False,
        "General, long-context (base model behind CodeRankEmbed).",
    ),
    ModelInfo(
        "nomic-ai/nomic-embed-text-v1.5",
        768,
        0.52,
        False,
        "General, long-context, Matryoshka dims.",
    ),
    ModelInfo(
        "BAAI/bge-small-en-v1.5",
        384,
        0.067,
        False,
        "Current default. Smallest/fastest; weakest on code (~45.8 CoIR).",
    ),
)


def format_models() -> str:
    """Human-readable table of recommended models for the CLI."""
    rows = [("model", "dim", "size", "code?", "note")]
    rows += [
        (
            m.name,
            str(m.dim),
            f"{m.size_gb:g}GB",
            "yes" if m.code_specific else "no",
            m.note,
        )
        for m in RECOMMENDED
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = []
    for i, r in enumerate(rows):
        head = "  ".join(r[j].ljust(widths[j]) for j in range(4))
        lines.append(f"{head}  {r[4]}")
        if i == 0:
            lines.append("  ".join("-" * w for w in widths) + "  " + "-" * len(r[4]))
    lines.append("")
    lines.append("Rerankers (set REPORAG_RERANK=1, REPORAG_RERANK_MODEL=<name>):")
    rwidth = max(len(rr.name) for rr in RECOMMENDED_RERANKERS)
    for rr in RECOMMENDED_RERANKERS:
        lines.append(f"  {rr.name.ljust(rwidth)}  {f'{rr.size_gb:g}GB':>8}  {rr.note}")
    return "\n".join(lines)


@dataclass(frozen=True)
class RerankerInfo:
    name: str  # fastembed TextCrossEncoder model id (pass via REPORAG_RERANK_MODEL)
    size_gb: float
    note: str


# Local cross-encoder rerankers loadable via fastembed's TextCrossEncoder. The MiniLM
# pair is web-trained (small/fast); bge/jina are larger and worth testing for code.
RECOMMENDED_RERANKERS: Tuple[RerankerInfo, ...] = (
    RerankerInfo(
        "Xenova/ms-marco-MiniLM-L-12-v2",
        0.12,
        "Default. Tiny/fast (~30ms CPU); web-trained, not code-specific.",
    ),
    RerankerInfo(
        "Xenova/ms-marco-MiniLM-L-6-v2",
        0.08,
        "Smallest/fastest MiniLM; slightly weaker than L-12.",
    ),
    RerankerInfo(
        "BAAI/bge-reranker-base",
        1.04,
        "Larger, stronger general reranker; multilingual incl. code-ish text.",
    ),
    RerankerInfo(
        "jinaai/jina-reranker-v2-base-multilingual",
        1.11,
        "Strong multilingual reranker with code in its training mix.",
    ),
)
