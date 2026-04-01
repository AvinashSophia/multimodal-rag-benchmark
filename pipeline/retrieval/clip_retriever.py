"""CLIP-based image retrieval module."""

import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
from pipeline.retrieval.base import BaseRetriever, register_retriever
from pipeline.utils import RetrievalResult


@register_retriever("clip")
class CLIPRetriever(BaseRetriever):
    """CLIP-based image retrieval.

    Uses OpenAI CLIP to encode both text queries and images into a
    shared embedding space, enabling cross-modal retrieval.
    Text query -> retrieve matching images.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import open_clip

        image_cfg = config["retrieval"]["image"]
        model_name = image_cfg.get("model_name", "ViT-B-32")
        self.fusion_alpha: float = image_cfg.get("fusion_alpha", 0.5)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.images: List[Any] = []
        self.image_embeddings: Optional[np.ndarray] = None

    def _encode_image(self, img: Any) -> np.ndarray:
        """Encode a single PIL image to a normalised CLIP vector."""
        import torch
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        with torch.no_grad():
            img_tensor = self.preprocess(img).unsqueeze(0)
            embedding = self.model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def index(self, corpus: List[Any], corpus_ids: Optional[List[str]] = None) -> None:
        """Encode and index images using CLIP vision encoder."""
        self.images = corpus
        self.image_ids: List[str] = corpus_ids if corpus_ids else [f"image_{i}" for i in range(len(corpus))]
        self.image_embeddings = np.array([self._encode_image(img) for img in corpus])

    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve top-k images using CLIP with late fusion.

        - query only:              text→image (text encoder)
        - query_image only:        image→image (visual encoder)
        - both:                    fuse both scores with fusion_alpha
        """
        import torch

        if self.image_embeddings is None:
            raise RuntimeError("Index not built. Call index() first.")

        has_query = bool(query and query.strip())
        has_image = query_image is not None

        n = len(self.images)
        text_scores = np.zeros(n)
        image_scores = np.zeros(n)

        if has_query:
            with torch.no_grad():
                text_tokens = self.tokenizer([query])
                text_emb = self.model.encode_text(text_tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_scores = np.dot(self.image_embeddings, text_emb.cpu().numpy().flatten())

        if has_image:
            image_scores = np.dot(self.image_embeddings, self._encode_image(query_image))

        if has_query and has_image:
            fused = self.fusion_alpha * text_scores + (1 - self.fusion_alpha) * image_scores
            mode = "fusion"
        elif has_query:
            fused = text_scores
            mode = "text→image"
        else:
            fused = image_scores
            mode = "image→image"

        top_indices = np.argsort(fused)[::-1][:top_k]

        return RetrievalResult(
            text_chunks=[],
            text_scores=[],
            text_ids=[],
            images=[self.images[i] for i in top_indices],
            image_scores=[float(fused[i]) for i in top_indices],
            image_ids=[self.image_ids[i] for i in top_indices],
            metadata={"method": "clip", "mode": mode, "fusion_alpha": self.fusion_alpha, "query": query, "image_query": has_image},
        )
