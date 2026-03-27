"""CLIP-based image retrieval module."""

import numpy as np
from typing import Dict, List, Any
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

        model_name = config["retrieval"]["image"].get("model_name", "ViT-B-32")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.images = []
        self.image_embeddings = None

    def index(self, images: List[Any]) -> None:
        """Encode and index images using CLIP vision encoder."""
        import torch

        self.images = images
        embeddings = []

        with torch.no_grad():
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, Image.Image):
                    img = img.convert("RGB")

                img_tensor = self.preprocess(img).unsqueeze(0)
                embedding = self.model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().numpy().flatten())

        self.image_embeddings = np.array(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k images matching the text query."""
        import torch

        if self.image_embeddings is None:
            raise RuntimeError("Index not built. Call index() first.")

        with torch.no_grad():
            text_tokens = self.tokenizer([query])
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy().flatten()

        # Cosine similarity
        scores = np.dot(self.image_embeddings, text_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return RetrievalResult(
            text_chunks=[],
            text_scores=[],
            text_ids=[],
            images=[self.images[i] for i in top_indices],
            image_scores=[float(scores[i]) for i in top_indices],
            image_ids=[f"image_{i}" for i in top_indices],
            metadata={"method": "clip", "query": query},
        )
