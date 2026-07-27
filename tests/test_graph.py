"""Structure-aware 1-hop callee expansion (retrieval.graph + searcher wiring)."""

from __future__ import annotations

from coderag.api import CodeRAG
from coderag.config import Config
from coderag.retrieval.graph import called_names, neighbor_ids
from tests.conftest import write

# A callee definition and a seed function that *calls* it. The query words appear only in
# the seed's docstring, so the callee is unreachable lexically — it can enter the results
# only through the call-graph edge (seed -> compute_tax definition).
_CALLEE = "def compute_tax(amount):\n    return amount * 0.2\n"
_SEED = (
    "def run_billing(order):\n"
    '    """Process the quarterly billing statement run."""\n'
    "    return compute_tax(order.total)\n"
)
_QUERY = "quarterly billing statement run"


def _indexed(config: Config, *, distractors: int = 13) -> CodeRAG:
    config.watched_dir.mkdir(parents=True, exist_ok=True)
    write(config.watched_dir / "core.py", _CALLEE)
    write(config.watched_dir / "handlers.py", _SEED)
    filler = "".join(
        f"def helper_{n}(x):\n    return x + {n}\n\n" for n in range(distractors)
    )
    write(config.watched_dir / "filler.py", filler)
    cr = CodeRAG(config)
    cr.index()
    return cr


def _chunk_id(cr: CodeRAG, path: str, query: str = "compute_tax run_billing") -> int:
    for h in cr.search(query, top_k=20):
        if h.path == path:
            return h.chunk_id
    raise AssertionError(f"no chunk indexed for {path}")


def test_called_names_extracts_call_targets():
    names = called_names("return compute_tax(order.total) + helper(1) + compute_tax(2)")
    assert names == ["compute_tax", "helper"]  # source order, deduped
    assert called_names("") == []
    assert called_names("a(1)") == []  # too short (<3 chars)


def test_symbol_index_maps_bare_names(config):
    cr = _indexed(config)
    idx = cr.store.symbol_index()
    callee = _chunk_id(cr, "core.py")
    assert idx.get("compute_tax") == [callee]
    assert "run_billing" in idx
    # A second call with no write in between is served from cache (same object).
    assert cr.store.symbol_index() is idx


def test_symbol_index_invalidates_on_write(config):
    cr = _indexed(config)
    assert "compute_tax" in cr.store.symbol_index()
    # A new definition must show up after (re)indexing — the cache keys on a write gen.
    write(config.watched_dir / "extra.py", "def brand_new_symbol():\n    return 1\n")
    cr.index()
    assert "brand_new_symbol" in cr.store.symbol_index()


def test_neighbor_ids_resolves_callees(config):
    cr = _indexed(config)
    seed = _chunk_id(cr, "handlers.py")
    callee = _chunk_id(cr, "core.py")
    seed_text = cr.store.hydrate([seed])[seed]["text"]
    neighbors = neighbor_ids(
        cr.store, [seed], {seed: seed_text}, per_seed=5, max_seeds=5
    )
    assert callee in neighbors  # the definition of what the seed calls
    assert seed not in neighbors


def test_neighbor_ids_empty_inputs(config):
    cr = _indexed(config)
    assert neighbor_ids(cr.store, [], {}, per_seed=5, max_seeds=5) == []
    assert neighbor_ids(cr.store, [1], {1: ""}, per_seed=0, max_seeds=5) == []


def test_neighbor_ids_empty_symbol_index(config):
    # No indexed chunks -> no symbol index -> nothing to resolve callees against.
    config.watched_dir.mkdir(parents=True, exist_ok=True)
    cr = CodeRAG(config)
    cr.index()
    assert cr.store.symbol_index() == {}
    assert neighbor_ids(cr.store, [1], {1: "foo()"}, per_seed=5, max_seeds=5) == []


def test_neighbor_ids_caps_per_seed(config):
    # A seed that calls two defined callees, capped at one per seed.
    config.watched_dir.mkdir(parents=True, exist_ok=True)
    write(config.watched_dir / "a.py", "def alpha():\n    return 1\n")
    write(config.watched_dir / "b.py", "def beta():\n    return 2\n")
    write(config.watched_dir / "g.py", "def gamma():\n    return alpha() + beta()\n")
    cr = CodeRAG(config)
    cr.index()
    seed = _chunk_id(cr, "g.py", query="gamma alpha beta")
    seed_text = cr.store.hydrate([seed])[seed]["text"]
    one = neighbor_ids(cr.store, [seed], {seed: seed_text}, per_seed=1, max_seeds=5)
    assert len(one) == 1
    both = neighbor_ids(cr.store, [seed], {seed: seed_text}, per_seed=5, max_seeds=5)
    assert len(both) == 2


def test_graph_expansion_on_empty_index_returns_nothing(config):
    config.watched_dir.mkdir(parents=True, exist_ok=True)
    cr = CodeRAG(config.with_overrides(graph_expansion=True))
    cr.index()
    assert cr.search("anything", top_k=5) == []


def test_graph_expansion_surfaces_callee(config):
    """A query that only matches the caller still pulls in its callee's definition."""
    cr = _indexed(
        config.with_overrides(graph_expansion=True, graph_weight=8.0, top_k=5)
    )
    paths = [h.path for h in cr.search(_QUERY, top_k=5)]
    assert "handlers.py" in paths  # the seed (caller), matched by the query
    assert "core.py" in paths  # the callee definition, reached only via the graph edge


def test_graph_expansion_off_by_default(config):
    assert Config().graph_expansion is False
    cr = _indexed(config.with_overrides(top_k=5))
    assert "core.py" not in [h.path for h in cr.search(_QUERY, top_k=5)]


def test_graph_expansion_keeps_primary_hit(config):
    cr = _indexed(config.with_overrides(graph_expansion=True))
    hits = cr.search("run_billing", top_k=5)
    assert hits and hits[0].path == "handlers.py"
