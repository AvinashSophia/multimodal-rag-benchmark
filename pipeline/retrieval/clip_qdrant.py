
"""Production CLIP image retriever backed by Qdrant vector database."""

import uuid
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, ScoredPoint
)
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("clip_qdrant")
class CLIPQdrantRetriever(BaseRetriever):
    """Production-grade CLIP image retriever backed by Qdrant.

    Encodes images with CLIP and stores vectors in a persistent Qdrant
    collection. On subsequent runs with the same image corpus, encoding
    is skipped entirely — vectors are loaded directly from Qdrant.

    Image IDs are stored as Qdrant payload. PIL images are kept in a
    memory dict (keyed by image_id) since binary blobs don't belong in
    a vector DB — in production these would be fetched from S3/blob storage.

    Config keys (under retrieval.image):
        model_name:           CLIP model name (default: ViT-B-32)
        top_k:                Number of results to retrieve
        qdrant.path:          Local path for Qdrant storage (default: pipeline/outputs/qdrant_store)
        qdrant.collection:    Collection name (default: clip_images)
    """

    CLIP_VECTOR_SIZE = 512   # ViT-B-32 output dimension
    BATCH_SIZE = 32          # Encode and upsert in batches

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import open_clip

        image_cfg = config["retrieval"]["image"]
        qdrant_cfg = image_cfg.get("qdrant", {})
        self.model_name = image_cfg.get("model_name", "ViT-B-32")
        self.fusion_alpha: float = image_cfg.get("fusion_alpha", 0.5)
        dataset_name = config.get("dataset", {}).get("name", "default")
        self.collection = f"{qdrant_cfg.get('collection', 'clip_images')}_{dataset_name}"
        qdrant_url = config["retrieval"].get("qdrant", {}).get("url", "http://localhost:6333")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        # Detect actual vector size from model
        self.vector_size: int = self.model.visual.output_dim

        self.qdrant = QdrantClient(url=qdrant_url)

        # In-memory image store: {image_id: PIL.Image}
        # Images are kept in memory since they can't be stored in Qdrant efficiently.
        # In production, replace with S3 fetch by image_id.
        self._image_store: Dict[str, Any] = {}

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

    def _encode_image(self, img: Any) -> np.ndarray:
        """Encode a single PIL image to a normalised CLIP vector."""
        import torch
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        with torch.no_grad():
            tensor = self.preprocess(img).unsqueeze(0)
            embedding = self.model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[Any], corpus_ids: Optional[List[str]] = None) -> None:
        """Encode images and store vectors in Qdrant. Skips encoding if already indexed."""
        ids = corpus_ids or [f"image_{i}" for i in range(len(corpus))]

        # Always populate in-memory image store (needed for retrieve())
        self._image_store = {img_id: img for img_id, img in zip(ids, corpus)}

        if self._collection_exists(len(corpus)):
            print(f"  [CLIPQdrant] Collection '{self.collection}' already indexed "
                  f"({len(corpus)} images). Skipping encoding.")
            return

        print(f"  [CLIPQdrant] Encoding {len(corpus)} images with CLIP '{self.model_name}'...")
        self._recreate_collection()

        # Encode and upsert in batches
        for start in range(0, len(corpus), self.BATCH_SIZE):
            batch_images = corpus[start: start + self.BATCH_SIZE]
            batch_ids    = ids[start: start + self.BATCH_SIZE]

            embeddings = np.array([self._encode_image(img) for img in batch_images])

            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, img_id)),
                    vector=embedding.tolist(),
                    payload={"image_id": img_id},
                )
                for img_id, embedding in zip(batch_ids, embeddings)
            ]
            self.qdrant.upsert(collection_name=self.collection, points=points)

        print(f"  [CLIPQdrant] Indexed {len(corpus)} images into '{self.collection}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k images using CLIP with late fusion.

        - query only:       text→image (text encoder)
        - query_image only: image→image (visual encoder)
        - both:             query both paths with top_k*2 candidates each,
                            merge by image_id, fuse scores, return top-k.
        """
        import torch

        has_query = bool(query and query.strip())
        has_image = query_image is not None

        # Fetch more candidates per path so merged pool covers top-k after fusion
        candidate_limit = top_k * 2

        text_scores: Dict[str, float] = {}
        image_scores: Dict[str, float] = {}

        if has_query:
            with torch.no_grad():
                text_tokens = self.tokenizer([query])
                text_emb = self.model.encode_text(text_tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                text_vector = text_emb.cpu().numpy().flatten().tolist()
            text_hits: List[ScoredPoint] = self.qdrant.query_points(
                collection_name=self.collection,
                query=text_vector,
                limit=candidate_limit,
            ).points
            text_scores = {
                hit.payload["image_id"]: hit.score
                for hit in text_hits
                if hit.payload is not None
            }

        if has_image:
            image_hits: List[ScoredPoint] = self.qdrant.query_points(
                collection_name=self.collection,
                query=self._encode_image(query_image).tolist(),
                limit=candidate_limit,
            ).points
            image_scores = {
                hit.payload["image_id"]: hit.score
                for hit in image_hits
                if hit.payload is not None
            }

        # Merge candidate sets and fuse scores (zero-impute missing side)
        all_ids = set(text_scores) | set(image_scores)
        if has_query and has_image:
            fused = {
                img_id: self.fusion_alpha * text_scores.get(img_id, 0.0)
                        + (1 - self.fusion_alpha) * image_scores.get(img_id, 0.0)
                for img_id in all_ids
            }
            mode = "fusion"
        elif has_query:
            fused = text_scores
            mode = "text→image"
        else:
            fused = image_scores
            mode = "image→image"

        sorted_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:top_k]
        results = [
            (img_id, self._image_store[img_id], fused[img_id])
            for img_id in sorted_ids
            if img_id in self._image_store
        ]

        return RetrievalResult(
            text_chunks=[],
            text_scores=[],
            text_ids=[],
            images=[r[1] for r in results],
            image_scores=[r[2] for r in results],
            image_ids=[r[0] for r in results],
            metadata={"method": "clip_qdrant", "model": self.model_name, "mode": mode, "fusion_alpha": self.fusion_alpha, "query": query, "image_query": has_image},
        )
