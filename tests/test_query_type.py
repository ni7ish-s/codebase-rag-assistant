"""Tests for query-type detection and adaptive fusion weighting."""

from __future__ import annotations

from coderag.api import CodeRAG
from coderag.retrieval.query_type import (
    fusion_weights,
    looks_like_identifier,
    references_identifier,
)
from tests.conftest import write


def test_identifier_queries_detected():
    assert references_identifier("fts_search")
    assert references_identifier("reciprocal_rank_fusion")
    assert references_identifier("HybridSearcher.search")
    assert references_identifier("getUserToken")  # camelCase
    assert references_identifier("authenticate(token)")  # call paren


def test_identifier_embedded_in_prose_detected():
    # The case external validation exposed: a prose query that names a symbol.
    assert references_identifier("Fix tuple order in AliasGenerator.generate_aliases()")
    assert references_identifier("why does build_context drop the last chunk")
    assert references_identifier("update the `validate_token` helper")


def test_natural_language_queries_detected():
    assert not references_identifier("where is retry backoff handled")
    assert not references_identifier("how does indexing work")
    assert not references_identifier("the auth flow")
    assert not references_identifier("user authentication flow")
    assert not references_identifier("")


def test_prose_abbreviations_and_versions_are_not_identifiers():
    # Common false positives that must NOT trip the dotted-path rule.
    assert not references_identifier("fix the bug e.g. on startup")
    assert not references_identifier("bump to version 3.11 support")


def test_looks_like_identifier_alias():
    assert looks_like_identifier is references_identifier


def test_detection_is_linear_no_redos():
    # A long adversarial run must be handled in linear time (the query is API-reachable).
    # Quadratic backtracking on this input would take minutes; linear is milliseconds.
    big = "a" * 200_000 + " "
    assert references_identifier(big) is False
    assert references_identifier(big + "needs_a_match") is True


def test_fusion_weights_static_when_adaptive_off(config):
    cfg = config.with_overrides(dense_weight=1.0, lexical_weight=1.0)
    assert fusion_weights("anything at all here", cfg) == (1.0, 1.0)


def test_fusion_weights_tilt_by_query_type(config):
    cfg = config.with_overrides(
        adaptive_fusion=True,
        nl_dense_weight=1.0,
        nl_lexical_weight=0.4,
        code_dense_weight=0.4,
        code_lexical_weight=1.0,
    )
    # Natural language -> dense up.
    assert fusion_weights("where is the token validated", cfg) == (1.0, 0.4)
    # Identifier -> BM25 up.
    assert fusion_weights("validate_token", cfg) == (0.4, 1.0)


def test_default_weights_lean_dense_for_nl_neutral_for_code(config):
    # Validated defaults: NL leans dense; identifiers stay neutral (BM25-up hurt on-repo).
    cfg = config.with_overrides(adaptive_fusion=True)
    assert fusion_weights("how is the index rebuilt", cfg) == (1.0, 0.4)
    assert fusion_weights("rebuild_from_store", cfg) == (1.0, 1.0)


def test_adaptive_search_runs_end_to_end(config):
    repo = config.watched_dir
    repo.mkdir(parents=True, exist_ok=True)
    write(repo / "auth.py", "def validate_token(token):\n    return token\n")
    write(repo / "math_utils.py", "def add_numbers(a, b):\n    return a + b\n")
    cr = CodeRAG(config.with_overrides(adaptive_fusion=True))
    cr.index()
    # Identifier query still retrieves its exact symbol with adaptive weighting on.
    hits = cr.search("validate_token", top_k=3)
    assert any(h.symbol == "validate_token" for h in hits)
