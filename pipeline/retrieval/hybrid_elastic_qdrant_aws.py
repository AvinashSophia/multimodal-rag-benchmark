"""Hybrid text retriever: EKS Elasticsearch BM25 + EKS Qdrant dense, fused via RRF.

AWS variant of hybrid_elastic_qdrant.py — composes BM25ElasticAWSRetriever and
DenseQdrantAWSRetriever instead of their local counterparts. RRF fusion logic
is identical to the local variant.

EKS services:
    elasticsearch  →  localhost:9200  (BM25)
    bge-embedding  →  localhost:8112  (dense encoding via TEI)
    qdrant         →  localhost:6333  (dense vector storage)
"""

from typing import Dict, List, Any, Optional

from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.retrieval.bm25_elastic_aws import BM25ElasticAWSRetriever
from pipeline.retrieval.dense_qdrant_aws import DenseQdrantAWSRetriever
from pipeline.utils import RetrievalResult


@register_retriever("hybrid_elastic_qdrant_aws")
class HybridElasticQdrantAWSRetriever(BaseRetriever):
    """Hybrid text retriever: EKS Elasticsearch BM25 + EKS Qdrant dense, fused via RRF.

    Composes BM25ElasticAWSRetriever and DenseQdrantAWSRetriever — both backed by
    EKS services. RRF fusion, candidate pools, and all config keys are identical
    to HybridElasticQdrantRetriever. Only the sub-retriever config sections change
    (elasticsearch_aws, dense_qdrant_aws).

    Config keys (under retrieval.text.hybrid_elastic_qdrant_aws):
        rrf_k:       RRF constant k           (default: 60)
        bm25_top_k:  BM25 candidate pool size (default: 20)
        dense_top_k: Dense candidate pool size(default: 20)

    All elasticsearch_aws and dense_qdrant_aws sub-config keys are read from their
    respective sections exactly as BM25ElasticAWSRetriever and DenseQdrantAWSRetriever expect.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hybrid_cfg = config["retrieval"]["text"].get("hybrid_elastic_qdrant_aws", {})
        self.rrf_k       = hybrid_cfg.get("rrf_k", 60)
        self.bm25_top_k  = hybrid_cfg.get("bm25_top_k", 20)
        self.dense_top_k = hybrid_cfg.get("dense_top_k", 20)

        self.bm25  = BM25ElasticAWSRetriever(config)
        self.dense = DenseQdrantAWSRetriever(config)

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

        for rank, doc_id in enumerate(bm25_result.text_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._rrf_score(rank, self.rrf_k)

        for rank, doc_id in enumerate(dense_result.text_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._rrf_score(rank, self.rrf_k)

        text_lookup: Dict[str, str] = {}
        for doc_id, text in zip(dense_result.text_ids, dense_result.text_chunks):
            text_lookup[doc_id] = text
        for doc_id, text in zip(bm25_result.text_ids, bm25_result.text_chunks):
            text_lookup[doc_id] = text  # BM25 stored text wins on collision

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused_ids    = [doc_id for doc_id, _ in ranked]
        fused_scores = [score  for _, score  in ranked]
        fused_texts  = [text_lookup.get(doc_id, "") for doc_id in fused_ids]

        return RetrievalResult(
            text_chunks=fused_texts,
            text_scores=fused_scores,
            text_ids=fused_ids,
            images=[],
            image_scores=[],
            image_ids=[],
            metadata={
                "method": "hybrid_elastic_qdrant_aws",
                "rrf_k": self.rrf_k,
                "bm25_candidates": len(bm25_result.text_ids),
                "dense_candidates": len(dense_result.text_ids),
            },
        )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[str], corpus_ids: Optional[List[str]] = None) -> None:
        """Index corpus into EKS Elasticsearch and EKS Qdrant. Each backend skips if already indexed."""
        self.bm25.index(corpus, corpus_ids)
        self.dense.index(corpus, corpus_ids)

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve from both EKS backends in parallel and fuse via RRF. query_image is ignored."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future  = executor.submit(self.bm25.retrieve,  query, self.bm25_top_k)
            dense_future = executor.submit(self.dense.retrieve, query, self.dense_top_k)
            bm25_result  = bm25_future.result()
            dense_result = dense_future.result()
        return self._fuse(bm25_result, dense_result, top_k=top_k)

    def is_indexed(self) -> bool:
        """Return True if both BM25 and Dense sub-retrievers are indexed."""
        return self.bm25.is_indexed() and self.dense.is_indexed()

    def mark_not_applicable(self) -> None:
        """Mark both sub-retrievers as not applicable for the current dataset."""
        self.bm25.mark_not_applicable()
        self.dense.mark_not_applicable()

    def storage_info(self) -> Dict[str, Any]:
        return {"bm25": self.bm25.storage_info(), "dense": self.dense.storage_info()}
