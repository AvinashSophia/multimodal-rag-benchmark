"""BM25 sparse text retrieval module."""

from typing import Dict, List, Any, Optional
from rank_bm25 import BM25Okapi
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("bm25")
class BM25Retriever(BaseRetriever):
    """BM25 keyword-based text retrieval (baseline).

    Uses term frequency and inverse document frequency for ranking.
    Fast, no GPU needed, good baseline for comparison.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bm25 = None
        self.corpus: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def index(self, corpus: List[str], corpus_ids: Optional[List[str]] = None) -> None:
        """Build BM25 index from text corpus."""
        self.corpus = corpus
        self.corpus_ids = corpus_ids or [f"chunk_{i}" for i in range(len(corpus))]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k text chunks using BM25 scoring. query_image is ignored."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return RetrievalResult(
            text_chunks=[self.corpus[i] for i in top_indices],
            text_scores=[float(scores[i]) for i in top_indices],
            text_ids=[self.corpus_ids[i] for i in top_indices],
            images=[],
            image_scores=[],
            image_ids=[],
            metadata={"method": "bm25", "query": query},
        )
