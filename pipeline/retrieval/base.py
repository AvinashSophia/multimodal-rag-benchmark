"""Base retriever class and retriever registry."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pipeline.utils import RetrievalResult

class BaseRetriever(ABC):
    """Abstract base class for all retrieval modules.

    Every retriever must implement index() and retrieve() methods.
    This ensures retrievers are independently swappable.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def index(self, corpus: List[Any], corpus_ids: Optional[List[str]] = None) -> None:
        """Build the retrieval index from a corpus.

        Args:
            corpus: List of text chunks or images to index.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve relevant content for a query.

        Args:
            query:       The search query (text).
            top_k:       Number of results to return.
            query_image: Optional PIL image to use as the query (image→image retrieval).
                         Text retrievers ignore this. CLIP retrievers use it when provided.

        Returns:
            RetrievalResult with text chunks and/or images.
        """
        raise NotImplementedError


class HybridRetriever:
    """Combines text and image retrievers into a single retrieval call.

    This is the main retriever used in the pipeline. It delegates to
    a text retriever and an image retriever, then merges results.
    """

    def __init__(self, text_retriever: BaseRetriever, image_retriever: Optional[BaseRetriever] = None):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever

    def index(self, text_corpus: List[str], corpus_ids: List[str], images: Optional[List[Any]] = None, image_ids: Optional[List[str]] = None) -> None:
        """Index both text and images."""
        self._has_text = bool(text_corpus)
        if text_corpus:
            self.text_retriever.index(text_corpus, corpus_ids)
        if images and self.image_retriever:
            self.image_retriever.index(images, image_ids)

    def retrieve(self, query: str, text_top_k: int = 5, image_top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve from both text and image indexes.

        Args:
            query:       Text query used for text retrieval and text→image CLIP.
            text_top_k:  Number of text results.
            image_top_k: Number of image results.
            query_image: Optional PIL image for image→image CLIP retrieval.
                         When provided, the image retriever uses the visual encoder
                         instead of the text encoder.
        """
        text_result = self.text_retriever.retrieve(query, top_k=text_top_k) if getattr(self, "_has_text", False) else RetrievalResult()

        image_result = RetrievalResult()
        if self.image_retriever:
            image_result = self.image_retriever.retrieve(query, top_k=image_top_k, query_image=query_image)

        return RetrievalResult(
            text_chunks=text_result.text_chunks,
            text_scores=text_result.text_scores,
            text_ids=text_result.text_ids,
            images=image_result.images,
            image_scores=image_result.image_scores,
            image_ids=image_result.image_ids,
            metadata={
                "text_retriever": type(self.text_retriever).__name__,
                "image_retriever": type(self.image_retriever).__name__ if self.image_retriever else None,
            },
        )


# Registry for retrievers
RETRIEVER_REGISTRY: Dict[str, type] = {}


def register_retriever(name: str):
    """Decorator to register a retriever."""
    def decorator(cls):
        RETRIEVER_REGISTRY[name] = cls
        return cls
    return decorator


def get_retriever(config: Dict[str, Any], retriever_type: str = "text") -> BaseRetriever:
    """Factory function to get the right retriever from config."""
    if retriever_type == "text":
        name = config["retrieval"]["text"]["method"]
    elif retriever_type == "image":
        name = config["retrieval"]["image"]["method"]
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    if name not in RETRIEVER_REGISTRY:
        raise ValueError(
            f"Retriever '{name}' not found. Available: {list(RETRIEVER_REGISTRY.keys())}"
        )
    return RETRIEVER_REGISTRY[name](config)
