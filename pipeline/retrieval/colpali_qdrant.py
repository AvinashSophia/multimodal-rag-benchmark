"""ColPali image retriever backed by Qdrant vector database.

Stores patch-level embeddings in Qdrant (one Qdrant point per image patch).
At query time, uses approximate MaxSim scoring across query token vectors to rank pages.
"""

import uuid
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("colpali_qdrant")
class ColPaliQdrantRetriever(BaseRetriever):
    """ColPali image retriever backed by Qdrant.

    Encodes document page images with ColPali (PaliGemma-based) and stores
    patch-level embeddings in Qdrant. At query time, approximate MaxSim scoring
    across all query token vectors ranks pages by relevance.

    Unlike CLIP (single vector per image), ColPali uses late interaction:
    each page produces N patch vectors; scoring sums the max similarity per query token.
    This makes it far superior for engineering drawings, schematics, and text-heavy docs.

    query_image is intentionally ignored — ColPali's text encoder is sufficient for
    document retrieval and has no image query encoder.

    Config keys (under retrieval.image):
        colpali.model_name:   ColPali checkpoint (default: vidore/colpali-v1.2)
        qdrant.path:          Local path for Qdrant storage
        qdrant.collection:    Collection name prefix (default: colpali_images)
    """

    COLPALI_DIM = 128     # ColPali projection dimension (vidore/colpali-v1.2)
    BATCH_SIZE = 4        # Images per encoding batch (ColPali is memory-heavy)
    _METADATA_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "__colpali_index_metadata__"))

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        image_cfg = config["retrieval"]["image"]
        colpali_cfg = image_cfg.get("colpali", {})
        qdrant_cfg = image_cfg.get("qdrant", {})

        self.model_name = colpali_cfg.get("model_name", "vidore/colpali-v1.2")
        dataset_name = config.get("dataset", {}).get("name", "default")
        self.collection = f"{colpali_cfg.get('collection', 'colpali_images')}_{dataset_name}"
        qdrant_url = config["retrieval"].get("qdrant", {}).get("url", "http://localhost:6333")

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if self.device == "cuda":
            # CUDA: bfloat16 + device_map handles meta tensor loading correctly
            self.model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
        else:
            # CPU/MPS: use device_map="cpu" so accelerate handles meta→CPU allocation.
            # Stays on CPU for inference — reliable for all Mac configurations.
            self.device = "cpu"
            self.model = ColPali.from_pretrained(
                self.model_name,
                device_map="cpu",
            ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

        self.qdrant = QdrantClient(url=qdrant_url)

        # In-memory image store: {page_id: PIL.Image} — for returning result images.
        # In production, replace with blob storage fetch by page_id.
        self._image_store: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collection_exists(self, expected_page_count: int) -> bool:
        """Return True if collection was indexed with the same number of pages."""
        try:
            result = self.qdrant.retrieve(
                collection_name=self.collection,
                ids=[self._METADATA_ID],
                with_payload=True,
            )
            if result and result[0].payload:
                return result[0].payload.get("page_count") == expected_page_count
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
            vectors_config=VectorParams(size=self.COLPALI_DIM, distance=Distance.COSINE),
        )

    def _encode_images_batch(self, images: List[Any]) -> List[np.ndarray]:
        """Encode a batch of PIL images → list of (n_patches, dim) float32 arrays."""
        import torch
        pil_images = [
            Image.open(img).convert("RGB") if isinstance(img, str)
            else img.convert("RGB")
            for img in images
        ]
        batch = self.processor.process_images(pil_images).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)  # (batch, n_patches, dim)
        return [emb.float().cpu().numpy() for emb in embeddings]

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a text query → (n_query_tokens, dim) float32 array."""
        import torch
        batch = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            embeddings = self.model(**batch)  # (1, n_tokens, dim)
        return embeddings[0].float().cpu().numpy()

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def index(self, corpus: List[Any], corpus_ids: Optional[List[str]] = None) -> None:
        """Encode page images and store patch embeddings in Qdrant. Skips if already indexed."""
        ids = corpus_ids or [f"page_{i}" for i in range(len(corpus))]

        # Always populate in-memory image store (needed for retrieve())
        self._image_store = {page_id: img for page_id, img in zip(ids, corpus)}

        if self._collection_exists(len(ids)):
            print(f"  [ColPaliQdrant] Collection '{self.collection}' already indexed "
                  f"({len(ids)} pages). Skipping encoding.")
            return

        print(f"  [ColPaliQdrant] Encoding {len(ids)} pages with ColPali '{self.model_name}'...")
        self._recreate_collection()

        total_patches = 0
        for start in range(0, len(corpus), self.BATCH_SIZE):
            batch_images = corpus[start: start + self.BATCH_SIZE]
            batch_ids = ids[start: start + self.BATCH_SIZE]

            patch_arrays = self._encode_images_batch(batch_images)

            points = []
            for page_id, patches in zip(batch_ids, patch_arrays):
                for patch_idx, patch_vec in enumerate(patches):
                    points.append(PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{page_id}_patch_{patch_idx}")),
                        vector=patch_vec.tolist(),
                        payload={"page_id": page_id, "patch_idx": patch_idx},
                    ))

            self.qdrant.upsert(collection_name=self.collection, points=points)
            batch_patches = sum(len(p) for p in patch_arrays)
            total_patches += batch_patches
            print(f"    Pages {start + 1}–{min(start + self.BATCH_SIZE, len(corpus))}: "
                  f"{batch_patches} patches upserted.")

        # Metadata sentinel so skip-if-indexed works on subsequent runs
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._METADATA_ID,
                vector=[0.0] * self.COLPALI_DIM,
                payload={"type": "metadata", "page_count": len(ids)},
            )],
        )

        print(f"  [ColPaliQdrant] Indexed {len(ids)} pages "
              f"({total_patches} patches) into '{self.collection}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k pages using approximate ColPali MaxSim.

        For each query token vector, searches Qdrant for nearest patches.
        Aggregates per page: score(q, d) = Σ_i max_j(q_i · d_j)  (MaxSim).

        query_image is ignored — ColPali uses text-only query encoding.
        """
        query_embeddings = self._encode_query(query)  # (n_query_tokens, dim)

        # Larger candidate pool → more accurate MaxSim approximation
        candidate_limit = max(top_k * 20, 100)

        exclude_metadata = Filter(
            must_not=[FieldCondition(key="type", match=MatchValue(value="metadata"))]
        )

        page_scores: Dict[str, float] = defaultdict(float)

        for q_vec in query_embeddings:
            hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=q_vec.tolist(),
                limit=candidate_limit,
                with_payload=True,
                query_filter=exclude_metadata,
            ).points

            # Keep max score per page for this query token
            page_max: Dict[str, float] = {}
            for hit in hits:
                if hit.payload and "page_id" in hit.payload:
                    page_id = hit.payload["page_id"]
                    if page_id not in page_max or hit.score > page_max[page_id]:
                        page_max[page_id] = hit.score

            for page_id, score in page_max.items():
                page_scores[page_id] += score

        # Normalize by number of query tokens → score in [0, 1]
        n_query_tokens = len(query_embeddings)
        if n_query_tokens > 0:
            for page_id in page_scores:
                page_scores[page_id] /= n_query_tokens

        sorted_pages = sorted(page_scores, key=lambda x: page_scores[x], reverse=True)[:top_k]

        results = [
            (page_id, self._image_store.get(page_id), page_scores[page_id])
            for page_id in sorted_pages
            if page_id in self._image_store
        ]

        return RetrievalResult(
            text_chunks=[],
            text_scores=[],
            text_ids=[],
            images=[r[1] for r in results],
            image_scores=[r[2] for r in results],
            image_ids=[r[0] for r in results],
            metadata={
                "method": "colpali_qdrant",
                "model": self.model_name,
                "n_query_tokens": len(query_embeddings),
                "query": query,
            },
        )
