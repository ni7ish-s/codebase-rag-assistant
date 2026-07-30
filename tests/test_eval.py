"""Tests for the code-retrieval eval harness (metrics, dataset, scoring).

All offline/deterministic via the `fake` provider fixture.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from reporag.api import CodeRAG
from reporag.eval import (
    EvalCase,
    build_from_git,
    compare_modes,
    evaluate,
    load_dataset,
    save_dataset,
)
from reporag.eval.harness import (
    EvalResult,
    aggregate_by_mode,
    best_label,
    format_table,
    mean_results,
)
from reporag.eval.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from tests.conftest import write

# --- metrics ---


def test_recall_at_k_counts_fraction_found():
    ranked = ["a.py", "b.py", "c.py"]
    assert recall_at_k(ranked, {"b.py", "c.py"}, 3) == 1.0
    assert recall_at_k(ranked, {"b.py", "c.py"}, 2) == 0.5
    assert recall_at_k(ranked, {"z.py"}, 3) == 0.0


def test_hit_at_k_is_binary():
    ranked = ["a.py", "b.py"]
    assert hit_at_k(ranked, {"b.py"}, 2) == 1.0
    assert hit_at_k(ranked, {"b.py"}, 1) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_rewards_higher_ranks():
    high = ndcg_at_k(["rel", "x", "y"], {"rel"}, 3)
    low = ndcg_at_k(["x", "y", "rel"], {"rel"}, 3)
    assert high == 1.0  # single relevant at rank 1 is perfect
    assert 0.0 < low < high


def test_metrics_dedupe_ranked_ids():
    # Duplicate file paths (multiple chunks per file) must not consume top-k slots:
    # deduped to ["a.py", "b.py"], so both relevant files land within k=2.
    ranked = ["a.py", "a.py", "b.py"]
    assert recall_at_k(ranked, {"a.py", "b.py"}, 2) == 1.0
    # Without dedup the second "a.py" would have pushed "b.py" out of the top 2.
    assert recall_at_k(ranked, {"a.py", "b.py"}, 1) == 0.5


def test_metrics_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), 1) == 0.0
    assert ndcg_at_k(["a"], set(), 1) == 0.0


# --- dataset ---


def test_dataset_roundtrip(tmp_path: Path):
    cases = [
        EvalCase(
            "find auth", ["auth.py"], ["authenticate_user"], id="c1", source="git"
        ),
        EvalCase("find math", ["math_utils.py"]),
    ]
    path = tmp_path / "ds.jsonl"
    save_dataset(cases, path)
    loaded = load_dataset(path)
    assert [c.query for c in loaded] == ["find auth", "find math"]
    assert loaded[0].relevant_symbols == ["authenticate_user"]
    assert loaded[1].relevant_symbols == []


def test_load_dataset_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        '{"query": "q", "relevant_files": ["a.py"]}\n\n   \n', encoding="utf-8"
    )
    assert len(load_dataset(path)) == 1


# --- harness: end-to-end scoring against a real (fake-embedded) index ---


def _indexed(config) -> CodeRAG:
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


def test_evaluate_perfect_retrieval_scores_one(config):
    cr = _indexed(config)
    cases = [EvalCase("add_numbers", ["math_utils.py"])]
    res = evaluate(cr.search, cases, ks=(1, 3))
    assert res.n == 1
    assert res.recall[1] == 1.0
    assert res.mrr == 1.0
    assert res.ndcg[1] == 1.0


def test_evaluate_skips_cases_without_ground_truth_at_level(config):
    cr = _indexed(config)
    # File-only ground truth -> nothing to score at the symbol level.
    cases = [EvalCase("add_numbers", ["math_utils.py"])]
    res = evaluate(cr.search, cases, ks=(1,), level="symbol")
    assert res.n == 0


def test_evaluate_symbol_level(config):
    cr = _indexed(config)
    cases = [EvalCase("authenticate_user", ["auth.py"], ["authenticate_user"])]
    res = evaluate(cr.search, cases, ks=(1, 3), level="symbol")
    assert res.n == 1
    assert res.hit[3] == 1.0


def test_compare_modes_returns_three_labels(config):
    cr = _indexed(config)
    cases = [
        EvalCase("add_numbers", ["math_utils.py"]),
        EvalCase("authenticate session token", ["auth.py"]),
    ]
    results = compare_modes(cr, cases, ks=(1, 3))
    assert [r.label for r in results] == ["dense", "bm25", "hybrid"]
    assert all(r.n == 2 for r in results)


def test_compare_modes_appends_adaptive_and_graph_rows(config):
    cr = _indexed(config)
    cases = [EvalCase("add_numbers", ["math_utils.py"])]
    results = compare_modes(cr, cases, ks=(1, 3), adaptive=True, graph=True)
    labels = [r.label for r in results]
    assert labels == ["dense", "bm25", "hybrid", "adaptive", "hybrid+graph"]


def test_bm25_recalls_exact_identifier(config):
    # Lexical retrieval should find an exact identifier even when dense recall is weak.
    cr = _indexed(config)
    cases = [EvalCase("add_numbers", ["math_utils.py"])]
    results = compare_modes(cr, cases, ks=(1, 3))
    bm25 = next(r for r in results if r.label == "bm25")
    assert bm25.hit[3] == 1.0


def test_format_table_and_best_label(config):
    cr = _indexed(config)
    cases = [EvalCase("add_numbers", ["math_utils.py"])]
    results = compare_modes(cr, cases, ks=(1, 3))
    table = format_table(results)
    assert "mode" in table and "MRR" in table and "hybrid" in table
    assert best_label(results, metric="ndcg", k=3) in {"dense", "bm25", "hybrid"}


# --- git dataset miner ---


def test_build_from_git_mines_changed_files(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "Tester")
    git("config", "commit.gpgsign", "false")
    write(repo / "auth.py", "def authenticate_user(token):\n    return token\n")
    git("add", "-A")
    git("commit", "-q", "-m", "add user authentication helper")

    cases = build_from_git(repo, max_cases=10)
    assert len(cases) == 1
    assert cases[0].query == "add user authentication helper"
    assert cases[0].relevant_files == ["auth.py"]
    assert cases[0].source == "git"


def _mk(label: str, mrr: float, n: int = 10) -> EvalResult:
    ks = (1, 5)
    return EvalResult(
        label=label,
        level="symbol",
        n=n,
        ks=ks,
        recall={k: mrr for k in ks},
        hit={k: mrr for k in ks},
        ndcg={k: mrr for k in ks},
        mrr=mrr,
    )


def test_mean_results_macro_averages():
    out = mean_results(
        [_mk("hybrid", 0.6, n=5), _mk("hybrid", 0.4, n=50)], label="mean"
    )
    assert out.label == "mean"
    assert out.mrr == 0.5  # equal weight per repo, not per case
    assert out.n == 55  # total cases carried through
    assert out.recall[1] == 0.5


def test_aggregate_by_mode_groups_across_repos():
    per_repo = {
        "repoA": [_mk("hybrid", 0.6), _mk("adaptive", 0.7)],
        "repoB": [_mk("hybrid", 0.4), _mk("adaptive", 0.5)],
    }
    agg = aggregate_by_mode(per_repo)
    by_label = {r.label: r.mrr for r in agg}
    assert by_label == {"mean:hybrid": 0.5, "mean:adaptive": 0.6}
    # First-seen mode order preserved.
    assert [r.label for r in agg] == ["mean:hybrid", "mean:adaptive"]


def test_mean_results_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        mean_results([])


def test_extensions_for_uses_canonical_map():
    from reporag.chunking.languages import extensions_for

    exts = extensions_for(("python", "go"))
    assert ".py" in exts and ".go" in exts
    assert ".rs" not in exts  # rust not requested
    # Unknown language names contribute nothing rather than raising.
    assert extensions_for(("nonsense",)) == []


def test_build_from_git_skips_merges_and_short_subjects(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "Tester")
    git("config", "commit.gpgsign", "false")
    write(repo / "a.py", "x = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "wip")  # too short -> filtered out

    assert build_from_git(repo, max_cases=10, min_query_len=12) == []


def test_build_from_git_extracts_changed_symbols(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "Tester")
    git("config", "commit.gpgsign", "false")
    write(
        repo / "m.py",
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n",
    )
    git("add", "-A")
    git("commit", "-q", "-m", "initial two functions")
    # Change only beta's body -> only beta should be reported as changed.
    write(
        repo / "m.py",
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 22\n",
    )
    git("add", "-A")
    git("commit", "-q", "-m", "tweak beta return value")

    cases = build_from_git(repo, max_cases=10, symbols=True, min_query_len=5)
    latest = next(c for c in cases if c.id and c.query.startswith("tweak beta"))
    assert latest.relevant_files == ["m.py"]
    assert latest.relevant_symbols == ["beta"]  # alpha untouched


def test_build_from_git_symbols_off_by_default(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "Tester")
    git("config", "commit.gpgsign", "false")
    write(repo / "m.py", "def alpha():\n    return 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "add alpha function")

    cases = build_from_git(repo, max_cases=10)  # symbols=False
    assert cases[0].relevant_symbols == []
