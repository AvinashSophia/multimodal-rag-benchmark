"""ColQwen2 image retriever backed by EKS Qdrant + ColQwen2 inference service.

AWS variant of colqwen2_qdrant.py — replaces local ColQwen2 model loading with
HTTP calls to the EKS ColQwen2 inference service. Qdrant patch storage, MaxSim
scoring, skip-if-indexed, and image store are identical to the local variant.
No model weights are loaded locally.

EKS services:
    colqwen2  →  localhost:8111  (colpali-engine server, vidore/colqwen2-v1.0)
    qdrant    →  localhost:6333  (shared with all Qdrant-backed retrievers)
"""

import uuid
import base64
import numpy as np
import requests  # type: ignore[import-untyped]
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Any, Optional
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("colqwen2_qdrant_aws")
class ColQwen2QdrantAWSRetriever(BaseRetriever):
    """ColQwen2 image retriever backed by EKS Qdrant + ColQwen2 inference service.

    Encodes document page images by calling the EKS ColQwen2 service (POST /embed/images)
    and stores patch-level embeddings in EKS Qdrant. At query time, calls POST /embed/text
    and performs approximate MaxSim scoring via Qdrant ANN — identical to the local variant.

    ColQwen2 uses Qwen2-VL as its vision backbone (vs PaliGemma for ColPali).
    Both share the same late-interaction / MaxSim scoring mechanism and 128-dim projection.
    No model weights are loaded locally.

    Config keys (under retrieval.image.colqwen2_aws):
        base_url:    ColQwen2 service URL       (default: http://localhost:8111)
        collection:  Collection name prefix     (default: colqwen2_images)

    Config keys (under retrieval.qdrant):
        url:         Qdrant URL                 (default: http://localhost:6333)

    ColQwen2 service API (colpali-engine server, same interface as ColPali):
        POST /embed/images  {"images": [<base64>]}  → {"embeddings": [[[...patch vecs...]]]}
        POST /embed/text    {"queries": ["text"]}   → {"embeddings": [[[...token vecs...]]]}
        GET  /health
    """

    COLQWEN2_DIM = 128
    BATCH_SIZE = 4
    _METADATA_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "__colqwen2_aws_index_metadata__"))

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        image_cfg = config["retrieval"]["image"]
        colqwen2_cfg = image_cfg.get("colqwen2_aws", {})

        self.base_url = colqwen2_cfg.get("base_url", "http://localhost:8111").rstrip("/")
        dataset_name = config.get("dataset", {}).get("name", "default")
        self.collection = f"{colqwen2_cfg.get('collection', 'colqwen2_images')}_{dataset_name}"
        qdrant_url = config["retrieval"].get("qdrant", {}).get("url", "http://localhost:6333")

        self.qdrant = QdrantClient(url=qdrant_url)

        # S3 client for on-demand image fetch during retrieve().
        # Images are NOT kept in RAM — fetched by page_id at query time.
        from pipeline.utils.s3 import S3Client
        self._s3 = S3Client(config)
        # Use s3_prefix (e.g. "altumint") not dataset name (e.g. "altumint_aws")
        # so image keys match what parse_documents_aws.py uploaded.
        self._s3_prefix = config.get("dataset", {}).get("s3_prefix", "altumint")
        self._indexed_ids: set = self._load_indexed_ids()  # populated from Qdrant

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """Return True if the Qdrant collection has a valid metadata sentinel.

        Returns True when:
          - sentinel exists and page_count > 0  (normal full index), OR
          - sentinel exists and not_applicable=True (dataset has no images)
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
                return (result[0].payload.get("page_count") or 0) > 0
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
                vector=[0.0] * self.COLQWEN2_DIM,
                payload={"type": "metadata", "page_count": 0, "not_applicable": True},
            )],
        )
        print(f"  [ColQwen2QdrantAWS] Marked '{self.collection}' as not applicable for this dataset.")

    def _load_indexed_ids(self) -> set:
        """Read page_ids from Qdrant collection payload (avoids loading corpus).

        Called once at init so retrieve() knows which pages are indexed without
        needing index() to be called first.
        """
        try:
            ids: set = set()
            offset = None
            while True:
                result, next_offset = self.qdrant.scroll(
                    collection_name=self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                )
                for point in result:
                    if point.payload and "page_id" in point.payload:
                        ids.add(point.payload["page_id"])
                if next_offset is None:
                    break
                offset = next_offset
            return ids
        except Exception:
            return set()

    def _pil_to_b64(self, image: Any) -> str:
        """Convert a PIL Image to a base64-encoded PNG string."""
        pil = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _encode_images_batch(self, images: List[Any]) -> List[np.ndarray]:
        """Encode a batch of PIL images via the ColQwen2 service.

        POST /embed/images
        Body:     {"images": [<base64_png>, ...]}
        Response: {"embeddings": [[[patch_vec, ...], ...], ...]}
                  One list of patch vectors per image.
        """
        b64_images = [self._pil_to_b64(img) for img in images]
        resp = requests.post(
            f"{self.base_url}/embed/images",
            json={"images": b64_images},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["embeddings"]
        return [np.array(patch_vecs, dtype=np.float32) for patch_vecs in raw]

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a text query via the ColQwen2 service.

        POST /embed/text
        Body:     {"queries": ["text"]}
        Response: {"embeddings": [[[token_vec, ...], ...]]}
                  One list of token vectors per query.
        """
        resp = requests.post(
            f"{self.base_url}/embed/text",
            json={"queries": [query]},
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["embeddings"]
        return np.array(raw[0], dtype=np.float32)

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
            vectors_config=VectorParams(size=self.COLQWEN2_DIM, distance=Distance.COSINE),
        )

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def _fetch_image(self, page_id: str) -> Any:
        """Fetch a page screenshot from S3 by page_id."""
        from io import BytesIO
        s3_key = self._s3.image_key(self._s3_prefix, f"figures/{page_id}_page.png")
        data = self._s3.download_bytes(s3_key)
        return Image.open(BytesIO(data)).convert("RGB")

    def index(self, corpus: List[Any], corpus_ids: Optional[List[str]] = None) -> None:
        """Encode page images via ColQwen2 service and store patch embeddings in EKS Qdrant."""
        ids = corpus_ids or [f"page_{i}" for i in range(len(corpus))]

        self._indexed_ids = set(ids)

        if self._collection_exists(len(ids)):
            print(f"  [ColQwen2QdrantAWS] Collection '{self.collection}' already indexed "
                  f"({len(ids)} pages). Skipping encoding.")
            return

        print(f"  [ColQwen2QdrantAWS] Encoding {len(ids)} pages via ColQwen2 service ({self.base_url})...")
        self._recreate_collection()

        total_patches = 0
        for start in range(0, len(corpus), self.BATCH_SIZE):
            batch_images = corpus[start: start + self.BATCH_SIZE]
            batch_ids    = ids[start: start + self.BATCH_SIZE]

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

        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._METADATA_ID,
                vector=[0.0] * self.COLQWEN2_DIM,
                payload={"type": "metadata", "page_count": len(ids)},
            )],
        )

        print(f"  [ColQwen2QdrantAWS] Indexed {len(ids)} pages "
              f"({total_patches} patches) into '{self.collection}'.")

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k pages using approximate ColQwen2 MaxSim via EKS Qdrant.

        Sends all query token vectors in a single batched Qdrant request instead of
        one HTTP call per token — reduces N round trips to 1.
        query_image is ignored — ColQwen2 uses text-only query encoding.
        """
        from concurrent.futures import ThreadPoolExecutor
        from qdrant_client.models import QueryRequest as QdrantQueryRequest

        query_embeddings = self._encode_query(query)  # (n_query_tokens, dim)
        n_query_tokens = len(query_embeddings)

        candidate_limit = max(top_k * 10, 50)

        exclude_metadata = Filter(
            must_not=[FieldCondition(key="type", match=MatchValue(value="metadata"))]
        )

        # Single batched Qdrant request — one HTTP call for all token vectors
        batch_requests = [
            QdrantQueryRequest(
                query=q_vec.tolist(),
                limit=candidate_limit,
                with_payload=True,
                filter=exclude_metadata,
            )
            for q_vec in query_embeddings
        ]
        batch_results = self.qdrant.query_batch_points(
            collection_name=self.collection,
            requests=batch_requests,
        )

        page_scores: Dict[str, float] = defaultdict(float)
        for result in batch_results:
            page_max: Dict[str, float] = {}
            for hit in result.points:
                if hit.payload and "page_id" in hit.payload:
                    pid = hit.payload["page_id"]
                    if pid not in page_max or hit.score > page_max[pid]:
                        page_max[pid] = hit.score
            for pid, score in page_max.items():
                page_scores[pid] += score

        if n_query_tokens > 0:
            for pid in page_scores:
                page_scores[pid] /= n_query_tokens

        sorted_pages = [
            pid for pid in sorted(page_scores, key=lambda x: page_scores[x], reverse=True)[:top_k]
            if pid in self._indexed_ids
        ]

        if not sorted_pages:
            return RetrievalResult(
                text_chunks=[], text_scores=[], text_ids=[],
                images=[], image_scores=[], image_ids=[],
                metadata={"method": "colqwen2_qdrant_aws", "n_query_tokens": n_query_tokens},
            )

        # Fetch images from S3 in parallel
        def _fetch(pid):
            try:
                return (pid, self._fetch_image(pid), page_scores[pid])
            except Exception as e:
                print(f"    WARNING: could not fetch image for {pid}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=min(len(sorted_pages), 5)) as executor:
            fetched = list(executor.map(_fetch, sorted_pages))

        results = [r for r in fetched if r is not None]

        return RetrievalResult(
            text_chunks=[],
            text_scores=[],
            text_ids=[],
            images=[r[1] for r in results],
            image_scores=[r[2] for r in results],
            image_ids=[r[0] for r in results],
            metadata={
                "method": "colqwen2_qdrant_aws",
                "base_url": self.base_url,
                "n_query_tokens": n_query_tokens,
                "query": query,
            },
        )

    def storage_info(self) -> Dict[str, Any]:
        try:
            info = self.qdrant.get_collection(self.collection)
            vectors = info.points_count or 0
            dim = self.COLQWEN2_DIM
            return {
                "type": "qdrant",
                "collection": self.collection,
                "vectors": vectors,
                "dimension": dim,
                "estimated_mb": round((vectors * dim * 4) / (1024 * 1024), 2),
            }
        except Exception:
            return {"type": "qdrant", "collection": self.collection}
