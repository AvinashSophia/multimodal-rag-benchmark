"""PipelineService — wraps the RAG pipeline for single-query API use.

Initializes once on startup (corpus indexing, model loading) and serves
multiple queries without re-indexing. Zero modifications to existing pipeline
components — purely a service wrapper.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from pipeline.utils import load_config
from pipeline.datasets import get_dataset
from pipeline.retrieval import get_retriever, HybridRetriever
from pipeline.models import get_model
from pipeline.evaluation import Evaluator
from pipeline.api.schemas import (
    QueryRequest, QueryResponse, RetrievedTextChunk, RetrievedImage,
)


_AVAILABLE_DATASETS = ["altumint", "docvqa", "hotpotqa", "gqa"]
_AVAILABLE_TEXT_METHODS = ["bm25", "dense", "dense_qdrant", "bm25_elastic", "hybrid_elastic_qdrant"]
_AVAILABLE_IMAGE_METHODS = ["clip", "clip_qdrant", "colpali_qdrant", "colqwen2_qdrant"]
_AVAILABLE_MODELS = ["gpt", "gemini", "gemini_vertex", "qwen_vl", "qwen_vl_aws"]


class PipelineService:
    """Single-query RAG pipeline service.

    Call initialize() once on startup. Then call query() for each request.
    The corpus index is built during initialize() and reused for all queries
    (skip-if-indexed applies — Qdrant collections are not rebuilt if they exist).

    Supports per-query overrides for model, text_method, and image_method.
    When an override differs from the active config, the affected component is
    swapped without reloading the full corpus.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = load_config(config_path)
        self._retriever: Optional[HybridRetriever] = None
        self._model: Any = None
        self._evaluator: Optional[Evaluator] = None
        self._initialized: bool = False
        self._text_top_k: int = self.config["retrieval"]["text"].get("top_k", 5)
        self._image_top_k: int = self.config["retrieval"]["image"].get("top_k", 5)
        # Cached corpus for re-indexing when retriever changes
        self._corpus: list = []
        self._corpus_ids: list = []
        self._images: list = []
        self._image_ids: list = []
        # Active component names (may differ from config defaults after overrides)
        self._active_text_method: str = self.config["retrieval"]["text"]["method"]
        self._active_image_method: str = self.config["retrieval"]["image"]["method"]
        self._active_model: str = self.config["model"]["name"]
        self._active_dataset: str = self.config["dataset"]["name"]

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build retrieval index and load model. Called once on app startup."""
        print("[PipelineService] Initializing...")

        # Load dataset — corpus is built from parsed JSONs (Altumint)
        dataset = get_dataset(self.config)
        dataset.load()

        self._images, self._image_ids = dataset.get_images()
        self._corpus, self._corpus_ids = dataset.get_corpus()

        # Initialize retrievers
        text_retriever = get_retriever(self.config, "text")
        image_retriever = get_retriever(self.config, "image") if self._images else None
        self._retriever = HybridRetriever(text_retriever, image_retriever)

        # Index corpus — skip-if-indexed for Qdrant collections
        self._retriever.index(
            self._corpus, self._corpus_ids,
            self._images if self._images else None,
            self._image_ids if self._image_ids else None,
        )
        print(f"  Indexed {len(self._corpus)} text chunks, {len(self._images)} images")

        # Load model and evaluator
        self._model = get_model(self.config)
        self._evaluator = Evaluator(self.config)

        self._active_text_method = self.config["retrieval"]["text"]["method"]
        self._active_image_method = self.config["retrieval"]["image"]["method"]
        self._active_model = self.config["model"]["name"]

        self._initialized = True
        print("[PipelineService] Ready.")

    def _swap_text_retriever(self, method: str) -> None:
        """Swap text retriever in-place; re-index using cached corpus."""
        print(f"[PipelineService] Swapping text retriever: {self._active_text_method} → {method}")
        cfg = dict(self.config)
        cfg["retrieval"] = {**self.config["retrieval"]}
        cfg["retrieval"]["text"] = {**self.config["retrieval"]["text"], "method": method}
        text_retriever = get_retriever(cfg, "text")
        image_retriever = (
            self._retriever.image_retriever if self._retriever else None
        )
        self._retriever = HybridRetriever(text_retriever, image_retriever)
        self._retriever.index(
            self._corpus, self._corpus_ids,
            self._images if self._images else None,
            self._image_ids if self._image_ids else None,
        )
        self._active_text_method = method

    def _swap_image_retriever(self, method: str) -> None:
        """Swap image retriever in-place; re-index using cached images."""
        print(f"[PipelineService] Swapping image retriever: {self._active_image_method} → {method}")
        cfg = dict(self.config)
        cfg["retrieval"] = {**self.config["retrieval"]}
        cfg["retrieval"]["image"] = {**self.config["retrieval"]["image"], "method": method}
        image_retriever = get_retriever(cfg, "image") if self._images else None
        text_retriever = self._retriever.text_retriever if self._retriever else None
        if text_retriever is None:
            raise RuntimeError("Cannot swap image retriever: no active text retriever")
        self._retriever = HybridRetriever(text_retriever, image_retriever)
        self._retriever.index(
            self._corpus, self._corpus_ids,
            self._images if self._images else None,
            self._image_ids if self._image_ids else None,
        )
        self._active_image_method = method

    def _swap_dataset(self, dataset_name: str) -> None:
        """Switch to a different dataset.

        Re-instantiates retrievers with the new dataset config so their
        collection/index names reflect the new dataset. The skip-if-indexed
        check inside each retriever's index() will then skip encoding if
        the collection already exists in Qdrant/Elastic.
        """
        print(f"[PipelineService] Swapping dataset: {self._active_dataset} → {dataset_name}")
        cfg = dict(self.config)
        cfg["dataset"] = {**self.config["dataset"], "name": dataset_name}

        # Load new dataset corpus
        dataset = get_dataset(cfg)
        dataset.load()
        self._images, self._image_ids = dataset.get_images()
        self._corpus, self._corpus_ids = dataset.get_corpus()

        # Re-instantiate retrievers with new dataset config so collection names
        # resolve to e.g. dense_text_hotpotqa instead of dense_text_altumint.
        text_retriever = get_retriever(cfg, "text")
        image_retriever = get_retriever(cfg, "image") if self._images else None
        self._retriever = HybridRetriever(text_retriever, image_retriever)

        # index() is a no-op for Qdrant/Elastic if the collection already exists
        self._retriever.index(
            self._corpus, self._corpus_ids,
            self._images if self._images else None,
            self._image_ids if self._image_ids else None,
        )
        self._active_dataset = dataset_name
        self.config = cfg

    def _swap_model(self, model_name: str) -> None:
        """Swap the generative model."""
        print(f"[PipelineService] Swapping model: {self._active_model} → {model_name}")
        cfg = dict(self.config)
        cfg["model"] = {**self.config["model"], "name": model_name}
        self._model = get_model(cfg)
        self._active_model = model_name

    def _ensure_config(self, request: "QueryRequest") -> None:
        """Apply per-request config overrides if they differ from active config."""
        if request.dataset and request.dataset != self._active_dataset:
            self._swap_dataset(request.dataset)
        if request.text_method and request.text_method != self._active_text_method:
            self._swap_text_retriever(request.text_method)
        if request.image_method and request.image_method != self._active_image_method:
            self._swap_image_retriever(request.image_method)
        if request.model and request.model != self._active_model:
            self._swap_model(request.model)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, request: "QueryRequest") -> QueryResponse:
        """Process a single query through the full RAG pipeline.

        retrieve → generate → (evaluate if ground_truth provided)
        Applies per-request config overrides before retrieving.
        """
        if not self._initialized or self._retriever is None or self._evaluator is None:
            raise RuntimeError("PipelineService not initialized. Call initialize() first.")

        self._ensure_config(request)

        start = time.monotonic()

        # Load query image if provided (visual query)
        query_image = None
        if request.query_image_path:
            query_image = Image.open(request.query_image_path).convert("RGB")

        # Retrieve
        t0 = time.monotonic()
        retrieved = self._retriever.retrieve(
            request.query,
            text_top_k=self._text_top_k,
            image_top_k=self._image_top_k,
            query_image=query_image,
        )
        retrieval_ms = (time.monotonic() - t0) * 1000

        # Generate answer
        t0 = time.monotonic()
        model_result = self._model.run_model(
            question=request.query,
            text_context=retrieved.text_chunks,
            image_context=retrieved.images,
            text_ids=retrieved.text_ids,
            image_ids=retrieved.image_ids,
        )
        generation_ms = (time.monotonic() - t0) * 1000

        # Evaluate — only if ground truth is provided
        t0 = time.monotonic()
        metrics: Optional[Dict[str, float]] = None
        if request.ground_truth is not None:
            metrics = self._evaluator.evaluate_sample(
                prediction=model_result.answer,
                ground_truth=request.ground_truth,
                all_ground_truths=[request.ground_truth],
                retrieved_texts=retrieved.text_chunks,
                retrieved_text_ids=retrieved.text_ids or None,
                used_sources=model_result.sources,
            )
        evaluation_ms = (time.monotonic() - t0) * 1000

        latency_ms = (time.monotonic() - start) * 1000

        # Build response
        retrieved_text = [
            RetrievedTextChunk(
                page_id=pid,
                text=text,
                score=score,
            )
            for pid, text, score in zip(
                retrieved.text_ids or [],
                retrieved.text_chunks,
                retrieved.text_scores,
            )
        ]

        retrieved_images = [
            RetrievedImage(page_id=pid, score=score)
            for pid, score in zip(
                retrieved.image_ids or [],
                retrieved.image_scores,
            )
        ]

        return QueryResponse(
            query=request.query,
            answer=model_result.answer,
            sources=model_result.sources,
            retrieved_text=retrieved_text,
            retrieved_images=retrieved_images,
            metrics=metrics,
            latency_ms=round(latency_ms, 1),
            latency_breakdown={
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "evaluation_ms": round(evaluation_ms, 1),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def dataset_name(self) -> str:
        return self._active_dataset

    @property
    def text_retriever_name(self) -> str:
        return self._active_text_method

    @property
    def image_retriever_name(self) -> str:
        return self._active_image_method

    @property
    def image_retriever(self):
        """Return the active image retriever instance (or None)."""
        return self._retriever.image_retriever if self._retriever else None

    @property
    def model_name(self) -> str:
        return self._active_model

    def config_options(self):
        """Return available and active configuration options."""
        from pipeline.api.schemas import ConfigOptions
        return ConfigOptions(
            datasets=_AVAILABLE_DATASETS,
            text_methods=_AVAILABLE_TEXT_METHODS,
            image_methods=_AVAILABLE_IMAGE_METHODS,
            models=_AVAILABLE_MODELS,
            active_dataset=self.dataset_name,
            active_text_method=self._active_text_method,
            active_image_method=self._active_image_method,
            active_model=self._active_model,
        )
