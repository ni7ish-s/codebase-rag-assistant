"""Tests for two-stage retrieve-then-rerank (offline via a fake reranker).

These never load the real cross-encoder; they verify the searcher's two-stage wiring:
deeper candidate pool, re-scoring, reordering, and trimming to top_k.
"""

from __future__ import annotations

from typing import List, Sequence

from reporag.api import CodeRAG
from reporag.config import Config
from reporag.eval import EvalCase, compare_modes
from reporag.retrieval.rerank import get_reranker
from reporag.retrieval.search import HybridSearcher
from tests.conftest import write


class KeywordReranker:
    """Deterministic fake reranker: score = count of query words present in the doc."""

    model_id = "fake-reranker"

    def rerank(self, query: str, documents: Sequence[str]) -> List[float]:
        terms = query.lower().split()
        return [float(sum(t in doc.lower() for t in terms)) for doc in documents]


def _indexed(config: Config) -> CodeRAG:
    config.watched_dir.mkdir(parents=True, exist_ok=True)
    write(
        config.watched_dir / "auth.py",
        "def authenticate_user(token):\n"
        "    '''Validate a session token and return the user.'''\n"
        "    return verify(token)\n",
    )
    write(
        config.watched_dir / "math_utils.py",
        "def add_numbers(a, b):\n    return a + b\n",
    )
    cr = CodeRAG(config)
    cr.index()
    return cr


def test_get_reranker_off_by_default(config):
    assert get_reranker(config) is None


def test_get_reranker_built_when_enabled(config):
    r = get_reranker(config.with_overrides(rerank=True))
    assert r is not None
    assert r.model_id  # default model id present


def test_reranker_reorders_and_sets_score(config):
    cr = _indexed(config)
    searcher = HybridSearcher(
        cr.config, cr.provider, cr.store, reranker=KeywordReranker()
    )
    hits = searcher.search("validate session token", top_k=2)
    assert hits
    # The auth chunk contains all three query words -> must rank first after rerank.
    assert hits[0].path == "auth.py"
    # Score is replaced by the cross-encoder score (here, the keyword overlap count).
    assert hits[0].score >= hits[-1].score


def test_rerank_trims_to_top_k(config):
    cr = _indexed(config)
    searcher = HybridSearcher(
        cr.config, cr.provider, cr.store, reranker=KeywordReranker()
    )
    assert len(searcher.search("token", top_k=1)) == 1


def test_reranker_empty_query(config):
    cr = _indexed(config)
    searcher = HybridSearcher(
        cr.config, cr.provider, cr.store, reranker=KeywordReranker()
    )
    assert searcher.search("   ", top_k=3) == []


def test_compare_modes_adds_rerank_row(config):
    cr = _indexed(config)
    cases = [EvalCase("validate session token", ["auth.py"])]
    results = compare_modes(cr, cases, ks=(1, 3), reranker=KeywordReranker())
    labels = [r.label for r in results]
    assert labels == ["dense", "bm25", "hybrid", "hybrid+rerank"]
    rerank_res = results[-1]
    assert rerank_res.hit[1] == 1.0  # keyword reranker nails the auth file at rank 1


def test_status_reports_rerank(config):
    cr = _indexed(config.with_overrides(rerank=True))
    status = cr.status()
    assert status["rerank"] is True
    assert status["rerank_model"]
