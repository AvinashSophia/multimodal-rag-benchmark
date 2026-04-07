"""Hybrid text retriever combining Elasticsearch BM25 and Qdrant dense retrieval.

Runs both retrievers independently, then fuses results using Reciprocal Rank
Fusion (RRF). RRF is score-scale agnostic — it only uses rank positions, so
BM25 raw scores and cosine similarities combine correctly without normalization.

Config keys (under retrieval.text):
    top_k:                       Final number of results to return
    hybrid_elastic_qdrant.rrf_k: RRF constant k (default: 60)
    hybrid_elastic_qdrant.bm25_top_k:  Candidate pool from BM25 before fusion (default: 20)
    hybrid_elastic_qdrant.dense_top_k: Candidate pool from dense before fusion (default: 20)

    All Elasticsearch and Qdrant sub-config keys are read from their respective
    sections (retrieval.text.elasticsearch and retrieval.text.qdrant) exactly as
    BM25ElasticRetriever and DenseQdrantRetriever expect them.
"""

from typing import Dict, List, Any, Optional

from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.retrieval.bm25_elastic import BM25ElasticRetriever
from pipeline.retrieval.dense_qdrant import DenseQdrantRetriever
from pipeline.utils import RetrievalResult


@register_retriever("hybrid_elastic_qdrant")
class HybridElasticQdrantRetriever(BaseRetriever):
    """Hybrid text retriever: Elasticsearch BM25 + Qdrant dense, fused via RRF.

    Both backends are constructed from the same config dict, so all existing
    Elasticsearch and Qdrant config keys continue to work unchanged.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hybrid_cfg = config["retrieval"]["text"].get("hybrid_elastic_qdrant", {})
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)
        self.bm25_top_k = hybrid_cfg.get("bm25_top_k", 20)
        self.dense_top_k = hybrid_cfg.get("dense_top_k", 20)

        self.bm25 = BM25ElasticRetriever(config)
        self.dense = DenseQdrantRetriever(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_score(rank: int, k: int) -> float:
        """Reciprocal Rank Fusion score for a document at the given rank (1-based)."""
        return 1.0 / (k + rank)

    def _fuse(
        self,
        bm25_result: RetrievalResult,
        dense_result: RetrievalResult,
        top_k: int,
    ) -> RetrievalResult:
        """Fuse two RetrievalResults using RRF and return the top-k merged result."""
        rrf_scores: Dict[str, float] = {}

        # Accumulate RRF scores from BM25 ranked list
        for rank, doc_id in enumerate(bm25_result.text_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._rrf_score(rank, self.rrf_k)

        # Accumulate RRF scores from dense ranked list
        for rank, doc_id in enumerate(dense_result.text_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._rrf_score(rank, self.rrf_k)

        # Build a lookup: doc_id → text chunk (BM25 result takes precedence for text)
        text_lookup: Dict[str, str] = {}
        for doc_id, text in zip(dense_result.text_ids, dense_result.text_chunks):
            text_lookup[doc_id] = text
        for doc_id, text in zip(bm25_result.text_ids, bm25_result.text_chunks):
            text_lookup[doc_id] = text  # BM25 stored text wins on collision

        # Sort by fused RRF score descending, take top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused_ids = [doc_id for doc_id, _ in ranked]
        fused_scores = [score for _, score in ranked]
        fused_texts = [text_lookup.get(doc_id, "") for doc_id in fused_ids]

        return RetrievalResult(
            text_chunks=fused_texts,
            text_scores=fused_scores,
            text_ids=fused_ids,
            images=[],
            image_scores=[],
            image_ids=[],
            metadata={
                "method": "hybrid_elastic_qdrant",
                "rrf_k": self.rrf_k,
                "bm25_candidates": len(bm25_result.text_ids),
                "dense_candidates": len(dense_result.text_ids),
            },
        )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[str], corpus_ids: Optional[List[str]] = None) -> None:
        """Index corpus into both Elasticsearch and Qdrant. Each backend skips if already indexed."""
        self.bm25.index(corpus, corpus_ids)
        self.dense.index(corpus, corpus_ids)

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve from both backends and fuse results via RRF. query_image is ignored."""
        bm25_result = self.bm25.retrieve(query, top_k=self.bm25_top_k)
        dense_result = self.dense.retrieve(query, top_k=self.dense_top_k)
        return self._fuse(bm25_result, dense_result, top_k=top_k)
