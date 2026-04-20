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

    def is_indexed(self) -> bool:
        """Return True if this retriever has a valid persistent index.

        Override in Qdrant/Elasticsearch-backed retrievers to check their
        persistent store. Default returns False (e.g. in-memory BM25).
        """
        return False

    def mark_not_applicable(self) -> None:
        """Mark this retriever as not applicable for the current dataset.

        Called by HybridRetriever.index() when this retriever is skipped because
        the dataset has no content for this modality (e.g. hotpotqa has no images,
        docvqa/gqa have no text corpus). Writing a not_applicable sentinel ensures
        is_indexed() returns True on subsequent runs so the full corpus is not
        re-loaded from S3 unnecessarily.

        Override in persistent-store retrievers. Default is a no-op.
        """
        pass

    def storage_info(self) -> Dict[str, Any]:
        """Return storage statistics for this retriever's index.

        Override in subclasses to provide accurate stats.
        Default returns an empty dict (e.g. for retrievers not yet instrumented).
        """
        return {}


class HybridRetriever:
    """Combines text and image retrievers into a single retrieval call.

    This is the main retriever used in the pipeline. It delegates to
    a text retriever and an image retriever, then merges results.
    """

    def __init__(self, text_retriever: BaseRetriever, image_retriever: Optional[BaseRetriever] = None):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        # Default True — assumes index already exists (built by batch job).
        # index() overrides this based on actual corpus content.
        self._has_text = True

    def index(self, text_corpus: List[str], corpus_ids: List[str], images: Optional[List[Any]] = None, image_ids: Optional[List[str]] = None) -> None:
        """Index both text and images."""
        self._has_text = bool(text_corpus)
        if text_corpus:
            self.text_retriever.index(text_corpus, corpus_ids)
        else:
            # Dataset has no text corpus (e.g. docvqa, gqa) — mark as not applicable
            # so is_indexed() returns True on subsequent runs without re-loading corpus.
            self.text_retriever.mark_not_applicable()

        if images and self.image_retriever:
            self.image_retriever.index(images, image_ids)
        elif self.image_retriever:
            # Dataset has no images (e.g. hotpotqa) — mark as not applicable.
            self.image_retriever.mark_not_applicable()

    def retrieve(self, query: str, text_top_k: int = 5, image_top_k: int = 5, query_image: Optional[Any] = None) -> RetrievalResult:
        """Retrieve from both text and image indexes in parallel.

        After retrieval, any page ID that appears in both text and image results
        is treated as cross-modal confirmed and boosted to the front of each list.
        The text chunk and image for that page are both kept — they are complementary,
        not redundant.

        Args:
            query:       Text query used for text retrieval and text→image CLIP.
            text_top_k:  Number of text results.
            image_top_k: Number of image results.
            query_image: Optional PIL image for image→image CLIP retrieval.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        text_result = RetrievalResult()
        image_result = RetrievalResult()

        def _text_retrieve():
            if getattr(self, "_has_text", False):
                return self.text_retriever.retrieve(query, top_k=text_top_k)
            return RetrievalResult()

        def _image_retrieve():
            if self.image_retriever:
                return self.image_retriever.retrieve(query, top_k=image_top_k, query_image=query_image)
            return RetrievalResult()

        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(_text_retrieve)
            image_future = executor.submit(_image_retrieve)
            text_result = text_future.result()
            image_result = image_future.result()

        text_result, image_result = self._boost_overlap(text_result, image_result)

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
                "cross_modal_overlap": len(set(text_result.text_ids) & set(image_result.image_ids)),
            },
        )

    @staticmethod
    def _boost_overlap(text_result: RetrievalResult, image_result: RetrievalResult):
        """Boost cross-modal confirmed pages to the front of each result list.

        A page ID appearing in both text_ids and image_ids has been confirmed
        relevant by two independent modalities. Move it to the top of each list
        so the model sees it first (primacy bias), preserving relative order
        within overlapping and non-overlapping groups.
        """
        overlap = set(text_result.text_ids) & set(image_result.image_ids)
        if not overlap:
            return text_result, image_result

        print(f"  [HybridRetriever] Cross-modal overlap: {sorted(overlap)}")

        def reorder(ids, chunks_or_images, scores):
            order = sorted(
                range(len(ids)),
                key=lambda i: (0 if ids[i] in overlap else 1, i),
            )
            return (
                [ids[i] for i in order],
                [chunks_or_images[i] for i in order],
                [scores[i] for i in order],
            )

        new_text_ids, new_text_chunks, new_text_scores = reorder(
            text_result.text_ids, text_result.text_chunks, text_result.text_scores
        )
        new_image_ids, new_images, new_image_scores = reorder(
            image_result.image_ids, image_result.images, image_result.image_scores
        )

        return (
            RetrievalResult(
                text_chunks=new_text_chunks,
                text_scores=new_text_scores,
                text_ids=new_text_ids,
                images=text_result.images,
                image_scores=text_result.image_scores,
                image_ids=text_result.image_ids,
                metadata=text_result.metadata,
            ),
            RetrievalResult(
                text_chunks=image_result.text_chunks,
                text_scores=image_result.text_scores,
                text_ids=image_result.text_ids,
                images=new_images,
                image_scores=new_image_scores,
                image_ids=new_image_ids,
                metadata=image_result.metadata,
            ),
        )


    def is_indexed(self) -> bool:
        """Return True if both text and image retrievers have valid persistent indices."""
        text_ok = self.text_retriever.is_indexed() if self.text_retriever else True
        image_ok = self.image_retriever.is_indexed() if self.image_retriever else True
        return text_ok and image_ok

    def storage_info(self) -> Dict[str, Any]:
        """Return combined storage info from text and image retrievers."""
        info: Dict[str, Any] = {}
        if self.text_retriever:
            info["text"] = self.text_retriever.storage_info()
        if self.image_retriever:
            info["image"] = self.image_retriever.storage_info()
        return info


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
