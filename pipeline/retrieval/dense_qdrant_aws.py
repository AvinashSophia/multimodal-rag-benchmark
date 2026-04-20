"""Dense text retriever backed by EKS-hosted Qdrant + BGE embedding service.

AWS variant of dense_qdrant.py — replaces local SentenceTransformer with
HTTP calls to the EKS BGE-large TEI service. Qdrant storage and skip-if-indexed
logic are identical to the local variant.

EKS services:
    bge-embedding  →  localhost:8112  (HuggingFace TEI, BAAI/bge-large-en-v1.5)
    qdrant         →  localhost:6333  (shared with all Qdrant-backed retrievers)
"""

import uuid
import requests  # type: ignore[import-untyped]
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, ScoredPoint,
    Filter, FieldCondition, MatchValue,
)
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("dense_qdrant_aws")
class DenseQdrantAWSRetriever(BaseRetriever):
    """Dense text retriever backed by EKS Qdrant + BGE TEI embedding service.

    Encodes text chunks by calling the EKS BGE-large TEI service (POST /embed)
    and stores the resulting vectors in the EKS Qdrant instance. No model weights
    are loaded locally — all encoding happens on the GPU node running the service.

    Skip-if-indexed, collection naming, and storage_info() are identical to
    DenseQdrantRetriever — only encoding is delegated to the remote service.

    Config keys (under retrieval.text.dense_qdrant_aws):
        base_url:     TEI service base URL  (default: http://localhost:8112)
        vector_size:  Embedding dimension   (default: 1024 for BGE-large-en-v1.5)

    Config keys (under retrieval.text.qdrant):
        collection:   Collection name prefix (default: dense_text)

    Config keys (under retrieval.qdrant):
        url:          Qdrant URL (default: http://localhost:6333)
    """

    BATCH_SIZE = 32  # Texts per Qdrant upsert batch (embedding is done one-at-a-time)
    _METADATA_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "__dense_qdrant_aws_index_metadata__"))

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        text_cfg = config["retrieval"]["text"]
        aws_cfg = text_cfg.get("dense_qdrant_aws", {})
        qdrant_cfg = text_cfg.get("qdrant", {})

        self.base_url = aws_cfg.get("base_url", "http://localhost:8112").rstrip("/")
        self.vector_size: int = aws_cfg.get("vector_size", 1024)  # BGE-large-en-v1.5

        dataset_name = config.get("dataset", {}).get("name", "default")
        self.collection = f"{qdrant_cfg.get('collection', 'dense_text')}_{dataset_name}"
        qdrant_url = config["retrieval"].get("qdrant", {}).get("url", "http://localhost:6333")

        self.qdrant = QdrantClient(url=qdrant_url)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call BGE TEI /embed one text at a time and return a list of float vectors.

        TEI API (HuggingFace Text Embeddings Inference):
            POST /embed
            Body: {"inputs": "text"}  → [0.12, -0.45, ...]  (single vector)
        """
        vectors = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/embed",
                json={"inputs": text},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            # Service returns either a flat vector [f, f, ...] or [[f, f, ...]]
            if isinstance(result[0], list):
                vectors.append(result[0])
            else:
                vectors.append(result)
        return vectors

    def is_indexed(self) -> bool:
        """Return True if the Qdrant collection has a valid metadata sentinel.

        Returns True when:
          - sentinel exists and corpus_count > 0  (normal full index), OR
          - sentinel exists and not_applicable=True (dataset has no text corpus)
        """
        try:
            result = self.qdrant.retrieve(
                collection_name=self.collection,
                ids=[self._METADATA_ID],
                with_payload=True,
            )
            if result and result[0].payload:
                if result[0].payload.get("not_applicable"):
                    return True
                return (result[0].payload.get("corpus_count") or 0) > 0
            return False
        except Exception:
            return False

    def mark_not_applicable(self) -> None:
        """Mark this retriever as not applicable for the current dataset."""
        self._recreate_collection()
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._METADATA_ID,
                vector=[0.0] * self.vector_size,
                payload={"type": "metadata", "corpus_count": 0, "not_applicable": True},
            )],
        )
        print(f"  [DenseQdrantAWS] Marked '{self.collection}' as not applicable for this dataset.")

    def _collection_exists(self, expected_count: int) -> bool:
        """Return True if collection sentinel exists with the expected corpus_count."""
        try:
            result = self.qdrant.retrieve(
                collection_name=self.collection,
                ids=[self._METADATA_ID],
                with_payload=True,
            )
            if result and result[0].payload:
                return result[0].payload.get("corpus_count") == expected_count
            return False
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
        """Encode corpus via BGE TEI service and store in EKS Qdrant. Skips if already indexed."""
        ids = corpus_ids or [f"chunk_{i}" for i in range(len(corpus))]

        if self._collection_exists(len(corpus)):
            print(f"  [DenseQdrantAWS] Collection '{self.collection}' already indexed "
                  f"({len(corpus)} chunks). Skipping encoding.")
            return

        print(f"  [DenseQdrantAWS] Encoding {len(corpus)} chunks via BGE TEI ({self.base_url})...")
        self._recreate_collection()

        for start in range(0, len(corpus), self.BATCH_SIZE):
            batch_texts = corpus[start: start + self.BATCH_SIZE]
            batch_ids   = ids[start: start + self.BATCH_SIZE]

            embeddings = self._embed(batch_texts)

            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=embedding,
                    payload={"text": text, "id": chunk_id},
                )
                for chunk_id, text, embedding in zip(batch_ids, batch_texts, embeddings)
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points)

        # Write metadata sentinel — marks a successfully completed full-corpus index.
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._METADATA_ID,
                vector=[0.0] * self.vector_size,
                payload={"type": "metadata", "corpus_count": len(corpus)},
            )],
        )

        print(f"  [DenseQdrantAWS] Indexed {len(corpus)} chunks into '{self.collection}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k chunks by semantic similarity from EKS Qdrant. query_image is ignored."""
        query_vector = self._embed([query])[0]

        exclude_metadata = Filter(
            must_not=[FieldCondition(key="type", match=MatchValue(value="metadata"))]
        )

        hits: List[ScoredPoint] = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=exclude_metadata,
        ).points

        return RetrievalResult(
            text_chunks=[hit.payload["text"] for hit in hits],       # type: ignore[index]
            text_scores=[hit.score for hit in hits],
            text_ids=[hit.payload["id"] for hit in hits],             # type: ignore[index]
            images=[],
            image_scores=[],
            image_ids=[],
            metadata={"method": "dense_qdrant_aws", "base_url": self.base_url, "query": query},
        )

    def storage_info(self) -> Dict[str, Any]:
        try:
            info = self.qdrant.get_collection(self.collection)
            vectors = info.points_count or 0
            dim = self.vector_size
            return {
                "type": "qdrant",
                "collection": self.collection,
                "vectors": vectors,
                "dimension": dim,
                "estimated_mb": round((vectors * dim * 4) / (1024 * 1024), 2),
            }
        except Exception:
            return {"type": "qdrant", "collection": self.collection}
