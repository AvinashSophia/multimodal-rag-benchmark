"""Production Dense retriever backed by Qdrant vector database."""

import uuid
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, ScoredPoint
)
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("dense_qdrant")
class DenseQdrantRetriever(BaseRetriever):
    """Production-grade dense text retriever backed by Qdrant.

    Encodes text chunks with a SentenceTransformer model and stores
    vectors in a persistent Qdrant collection. On subsequent runs with
    the same corpus, encoding is skipped entirely — vectors are loaded
    directly from Qdrant.

    Text content and IDs are stored as Qdrant payload so the retriever
    is fully self-contained: no in-memory corpus needed after indexing.

    Config keys (under retrieval.text):
        model_name:           SentenceTransformer model (default: BAAI/bge-large-en-v1.5)
        top_k:                Number of results to retrieve
        qdrant.path:          Local path for Qdrant storage (default: pipeline/outputs/qdrant_store)
        qdrant.collection:    Collection name (default: dense_text)
    """

    BATCH_SIZE = 64  # Encode and upsert in batches to avoid OOM

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        text_cfg = config["retrieval"]["text"]
        qdrant_cfg = text_cfg.get("qdrant", {})

        self.model_name = text_cfg.get("model_name", "BAAI/bge-large-en-v1.5")
        dataset_name = config.get("dataset", {}).get("name", "default")
        self.collection = f"{qdrant_cfg.get('collection', 'dense_text')}_{dataset_name}"
        qdrant_path = qdrant_cfg.get("path", "pipeline/outputs/qdrant_store")

        self.encoder = SentenceTransformer(self.model_name)
        self.vector_size: int = self.encoder.get_sentence_embedding_dimension() or 768
        self.qdrant = QdrantClient(path=qdrant_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collection_exists(self, expected_count: int) -> bool:
        """Return True if collection exists with the expected number of vectors."""
        try:
            info = self.qdrant.get_collection(self.collection)
            stored_count = info.points_count or 0
            stored_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            return stored_count == expected_count and stored_dim == self.vector_size
        except Exception:
            return False

    def _recreate_collection(self) -> None:
        """Drop (if exists) and recreate the Qdrant collection."""
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection in existing:
            self.qdrant.delete_collection(self.collection)
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[str], corpus_ids: Optional[List[str]] = None) -> None:
        """Encode corpus and store in Qdrant. Skips encoding if already indexed."""
        ids = corpus_ids or [f"chunk_{i}" for i in range(len(corpus))]

        if self._collection_exists(len(corpus)):
            print(f"  [DenseQdrant] Collection '{self.collection}' already indexed "
                  f"({len(corpus)} chunks). Skipping encoding.")
            return

        print(f"  [DenseQdrant] Encoding {len(corpus)} chunks with '{self.model_name}'...")
        self._recreate_collection()

        # Encode and upsert in batches
        for start in range(0, len(corpus), self.BATCH_SIZE):
            batch_texts = corpus[start: start + self.BATCH_SIZE]
            batch_ids   = ids[start: start + self.BATCH_SIZE]

            embeddings = self.encoder.encode(
                batch_texts, normalize_embeddings=True, show_progress_bar=False
            )

            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=embedding.tolist(),
                    payload={"text": text, "id": chunk_id},
                )
                for chunk_id, text, embedding in zip(batch_ids, batch_texts, embeddings)
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points)

        print(f"  [DenseQdrant] Indexed {len(corpus)} chunks into '{self.collection}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k chunks by semantic similarity from Qdrant. query_image is ignored."""
        query_vector = self.encoder.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        hits: List[ScoredPoint] = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        ).points

        return RetrievalResult(
            text_chunks=[hit.payload["text"] for hit in hits],       # type: ignore[index]
            text_scores=[hit.score for hit in hits],
            text_ids=[hit.payload["id"] for hit in hits],             # type: ignore[index]
            images=[],
            image_scores=[],
            image_ids=[],
            metadata={"method": "dense_qdrant", "model": self.model_name, "query": query},
        )
