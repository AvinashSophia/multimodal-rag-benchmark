"""Dense embedding-based text retrieval module."""

import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("dense")
class DenseRetriever(BaseRetriever):
    """Dense semantic text retrieval using sentence embeddings.

    Uses a pre-trained embedding model (e.g., bge-large) to encode
    both documents and queries into dense vectors, then retrieves
    by cosine similarity.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = config["retrieval"]["text"].get("model_name", "BAAI/bge-large-en-v1.5")
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None

    def index(self, corpus: List[str]) -> None:
        """Encode and index the text corpus."""
        self.corpus = corpus
        self.embeddings = self.model.encode(
            corpus, show_progress_bar=True, normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k text chunks by semantic similarity."""
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call index() first.")

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        return RetrievalResult(
            text_chunks=[self.corpus[i] for i in top_indices],
            text_scores=[float(scores[i]) for i in top_indices],
            images=[],
            image_scores=[],
            metadata={"method": "dense", "model": self.model.get_config_dict().get("model_name_or_path", ""), "query": query},
        )
