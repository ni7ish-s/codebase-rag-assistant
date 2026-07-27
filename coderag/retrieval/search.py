"""Hybrid searcher: dense + lexical retrieval fused with RRF, hydrated from the store."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from coderag.config import Config
from coderag.embeddings import EmbeddingProvider
from coderag.retrieval.fusion import reciprocal_rank_fusion
from coderag.retrieval.graph import neighbor_ids
from coderag.retrieval.query_type import fusion_weights
from coderag.types import SearchHit

if TYPE_CHECKING:
    from coderag.retrieval.rerank import Reranker
    from coderag.store.lance_store import LanceStore

logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(
        self,
        config: Config,
        provider: EmbeddingProvider,
        store: "LanceStore",
        reranker: Optional["Reranker"] = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.store = store
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int,
        *,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[SearchHit]:
        """Hybrid search. Pass a ``timings`` dict to receive a per-phase latency
        breakdown (``embed_ms``/``dense_ms``/``lexical_ms``/``hydrate_ms``/``rerank_ms``)
        in milliseconds — used by the demo UI to show where retrieval time actually goes.
        """
        if not query or not query.strip():
            return []

        # When reranking, pull a deeper candidate pool to rerank, then trim to top_k.
        pool = top_k
        if self.reranker is not None:
            pool = max(self.config.rerank_candidates, top_k)
        fetch_k = max(self.config.fetch_k, pool)

        # Dense retrieval (vector ANN over the store). The query embedding is the model
        # inference; on a busy/throttled host it can dwarf the store ops, so time it apart.
        t0 = time.perf_counter()
        qvec = self.provider.embed_query(query)
        if timings is not None:
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        dense = self.store.vector_search(qvec, fetch_k)
        if timings is not None:
            timings["dense_ms"] = (time.perf_counter() - t0) * 1000.0
        similarity: Dict[int, float] = {
            cid: float(max(0.0, min(1.0, s))) for cid, s in dense
        }
        dense_ranked = [cid for cid, _ in dense]

        # Lexical retrieval (BM25 over the store).
        t0 = time.perf_counter()
        lexical_ranked = [cid for cid, _ in self.store.lexical_search(query, fetch_k)]
        if timings is not None:
            timings["lexical_ms"] = (time.perf_counter() - t0) * 1000.0

        # Fuse, then trim to the candidate pool (top_k, or deeper when reranking).
        # Weights may adapt to the query type (dense-up for NL, BM25-up for identifiers).
        dense_w, lexical_w = fusion_weights(query, self.config)
        ranked_lists: List[List[int]] = [dense_ranked, lexical_ranked]
        weights: List[float] = [dense_w, lexical_w]

        # Optionally enrich the pool with 1-hop call-graph neighbors of the top hits
        # (definitions of what a seed calls — its callees). Fused in as a third,
        # down-weighted list so it adds recall without overpowering the direct signal.
        if self.config.graph_expansion:
            neighbors = self._graph_neighbors(
                dense_ranked, lexical_ranked, dense_w, lexical_w
            )
            if neighbors:
                ranked_lists.append(neighbors)
                weights.append(self.config.graph_weight)

        fused = reciprocal_rank_fusion(
            ranked_lists,
            k=self.config.rrf_k,
            weights=weights,
        )[:pool]
        if not fused:
            return []

        ids = [cid for cid, _ in fused]
        t0 = time.perf_counter()
        rows = self.store.hydrate(ids)
        if timings is not None:
            timings["hydrate_ms"] = (time.perf_counter() - t0) * 1000.0

        hits: List[SearchHit] = []
        for cid, score in fused:
            row = rows.get(cid)
            if row is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    path=row["path"],
                    symbol=row["symbol"] or None,
                    kind=row["kind"],
                    language=row["language"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    text=row["text"],
                    score=float(score),
                    similarity=similarity.get(cid, 0.0),
                )
            )

        if self.reranker is not None:
            t0 = time.perf_counter()
            hits = self._rerank(query, hits)
            if timings is not None:
                timings["rerank_ms"] = (time.perf_counter() - t0) * 1000.0
        return hits[:top_k]

    def _graph_neighbors(
        self,
        dense_ranked: List[int],
        lexical_ranked: List[int],
        dense_w: float,
        lexical_w: float,
    ) -> List[int]:
        """1-hop call-graph neighbors of the current top hits, ranked best-first.

        Fuses dense+lexical once to pick stable seeds, then resolves the callees named in
        each seed's text to their definitions (see ``retrieval.graph``).
        """
        seeds = reciprocal_rank_fusion(
            [dense_ranked, lexical_ranked],
            k=self.config.rrf_k,
            weights=[dense_w, lexical_w],
        )
        seed_ids = [cid for cid, _ in seeds[: self.config.graph_seeds]]
        if not seed_ids:
            return []
        rows = self.store.hydrate(seed_ids)
        seed_texts = {
            cid: (rows[cid]["text"] if cid in rows else "") for cid in seed_ids
        }
        return neighbor_ids(
            self.store,
            seed_ids,
            seed_texts,
            per_seed=self.config.graph_neighbors,
            max_seeds=self.config.graph_seeds,
        )

    def _rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """Re-score candidates jointly with the query and sort by the new score.

        The cross-encoder score replaces ``score`` (the relative ranking signal) so order
        and score agree; ``similarity`` keeps the dense cosine for display.
        """
        if not hits or self.reranker is None:
            return hits
        scores = self.reranker.rerank(query, [h.text for h in hits])
        for hit, s in zip(hits, scores, strict=False):
            hit.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
