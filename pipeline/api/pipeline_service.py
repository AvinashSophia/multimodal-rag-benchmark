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
from pipeline.retrieval import get_retriever, HybridRetriever
from pipeline.models import get_model
from pipeline.evaluation import Evaluator
from pipeline.api.schemas import (
    QueryRequest, QueryResponse, RetrievedTextChunk, RetrievedImage,
    QdrantCollectionStats, StorageOverview,
)


# Cost per 1M tokens in USD (input / output) — update when pricing changes
_TOKEN_COST_PER_1M: dict = {
    "gpt":           {"input": 2.50,  "output": 10.00},  # gpt-4o
    "gemini":        {"input": 0.10,  "output": 0.40},   # gemini-2.0-flash
    "gemini_vertex": {"input": 0.15,  "output": 0.60},   # gemini-2.5-flash
    "qwen_vl":       {"input": 0.00,  "output": 0.00},   # self-hosted
    "qwen_vl_aws":   {"input": 0.00,  "output": 0.00},   # self-hosted
}

_AVAILABLE_DATASETS = ["altumint_aws", "hotpotqa_aws", "docvqa_aws", "gqa_aws"]
_AVAILABLE_TEXT_METHODS = ["bm25_elastic_aws", "dense_qdrant_aws", "hybrid_elastic_qdrant_aws"]
_AVAILABLE_IMAGE_METHODS = ["colpali_qdrant_aws", "colqwen2_qdrant_aws"]
_AVAILABLE_MODELS = ["gpt", "gemini", "gemini_vertex", "qwen_vl_aws"]


class PipelineService:
    """Single-query RAG pipeline service.

    Call initialize() once on startup. Then call query() for each request.
    The corpus index is built during initialize() and reused for all queries
    (skip-if-indexed applies — Qdrant collections are not rebuilt if they exist).

    Supports per-query overrides for model, text_method, and image_method.
    When an override differs from the active config, the affected component is
    swapped without reloading the full corpus.

    Retriever instances are cached by (retriever_type, method, dataset) key so
    swapping back and forth between configurations never re-initializes a retriever
    that was already set up — no redundant Qdrant scrolls or ES reconnections.
    """

    def __init__(self, config_path: str = "configs/aws.yaml"):
        self.config = load_config(config_path)
        self._retriever: Optional[HybridRetriever] = None
        self._model: Any = None
        self._evaluator: Optional[Evaluator] = None
        self._initialized: bool = False
        self._text_top_k: int = self.config["retrieval"]["text"].get("top_k", 5)
        self._image_top_k: int = self.config["retrieval"]["image"].get("top_k", 5)
        # Active component names (may differ from config defaults after overrides)
        self._active_text_method: str = self.config["retrieval"]["text"]["method"]
        self._active_image_method: str = self.config["retrieval"]["image"]["method"]
        self._active_model: str = self.config["model"]["name"]
        self._active_dataset: str = self.config["dataset"]["name"]
        # Retriever instance cache — keyed by "{type}_{method}_{dataset}"
        # Instances are reused across swaps so _indexed_ids scroll and connection
        # setup only happen once per (type, method, dataset) combination.
        self._retriever_cache: Dict[str, Any] = {}
        # Model instance cache — keyed by model name
        # API client objects are lightweight but caching keeps behavior consistent.
        self._model_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Connect retrievers to existing indices and load model. Called once on app startup.

        No corpus build or indexing — the batch job (run_benchmark.py) is responsible
        for building the corpus and indexing into Qdrant/Elastic before the UI server starts.
        The UI server is a pure query interface: retrieve → generate → evaluate.
        """
        print("[PipelineService] Initializing...")

        # Connect retrievers to existing Qdrant/Elastic indices (batch job built these).
        # Seed the cache with startup instances so the first query never pays init cost.
        text_retriever = self._get_or_create_retriever(
            "text", self._active_text_method, self._active_dataset
        )
        image_retriever = self._get_or_create_retriever(
            "image", self._active_image_method, self._active_dataset
        )
        self._retriever = HybridRetriever(text_retriever, image_retriever)

        # Load model and evaluator — seed cache with startup model.
        self._model = self._get_or_create_model(self._active_model)
        self._evaluator = Evaluator(self.config)

        self._active_text_method = self.config["retrieval"]["text"]["method"]
        self._active_image_method = self.config["retrieval"]["image"]["method"]
        self._active_model = self.config["model"]["name"]

        # Warn if index has not been built yet — queries will return empty context.
        if not self._retriever.is_indexed():
            print("=" * 60)
            print("WARNING: No index found for the active dataset+retriever config.")
            print("Queries will run with zero retrieved context (hallucinated answers).")
            print("Run the batch job first to build the index:")
            print("  RAG_CONFIG=configs/aws.yaml uv run python -m pipeline.runners.run_benchmark_aws --config configs/aws.yaml")
            print("=" * 60)

        self._initialized = True
        print("[PipelineService] Ready.")

    @staticmethod
    def _s3_prefix_for(dataset: str) -> str:
        """Derive S3 prefix from dataset name — strips the _aws suffix.

        altumint_aws → altumint
        hotpotqa_aws → hotpotqa
        docvqa_aws   → docvqa
        gqa_aws      → gqa
        """
        return dataset.replace("_aws", "")

    def _get_or_create_retriever(self, retriever_type: str, method: str, dataset: str) -> Any:
        """Return a cached retriever instance or create and cache a new one.

        Key: "{retriever_type}_{method}_{dataset}"
        First call pays the init cost (connection setup, _indexed_ids scroll).
        Subsequent calls with the same key return the existing instance instantly.
        """
        key = f"{retriever_type}_{method}_{dataset}"
        if key not in self._retriever_cache:
            print(f"[PipelineService] Initializing retriever (cache miss): {key}")
            cfg = dict(self.config)
            cfg["retrieval"] = {**self.config["retrieval"]}
            # Always derive s3_prefix from dataset name so image fetching uses
            # the correct S3 path regardless of what the yaml default says.
            cfg["dataset"] = {
                **self.config["dataset"],
                "name": dataset,
                "s3_prefix": self._s3_prefix_for(dataset),
            }
            if retriever_type == "text":
                cfg["retrieval"]["text"] = {**self.config["retrieval"]["text"], "method": method}
            else:
                cfg["retrieval"]["image"] = {**self.config["retrieval"]["image"], "method": method}
            self._retriever_cache[key] = get_retriever(cfg, retriever_type)
        else:
            print(f"[PipelineService] Retriever cache hit: {key}")
        return self._retriever_cache[key]

    def _swap_text_retriever(self, method: str) -> None:
        """Swap text retriever — uses cache, no re-init if already seen."""
        print(f"[PipelineService] Swapping text retriever: {self._active_text_method} → {method}")
        text_retriever = self._get_or_create_retriever("text", method, self._active_dataset)
        image_retriever = self._retriever.image_retriever if self._retriever else None
        self._retriever = HybridRetriever(text_retriever, image_retriever)
        self._active_text_method = method

    def _swap_image_retriever(self, method: str) -> None:
        """Swap image retriever — uses cache, no re-init if already seen."""
        print(f"[PipelineService] Swapping image retriever: {self._active_image_method} → {method}")
        image_retriever = self._get_or_create_retriever("image", method, self._active_dataset)
        text_retriever = self._retriever.text_retriever if self._retriever else None
        if text_retriever is None:
            raise RuntimeError("Cannot swap image retriever: no active text retriever")
        self._retriever = HybridRetriever(text_retriever, image_retriever)
        self._active_image_method = method

    def _swap_dataset(self, dataset_name: str) -> None:
        """Switch to a different dataset — uses cache for both retrievers."""
        print(f"[PipelineService] Swapping dataset: {self._active_dataset} → {dataset_name}")
        text_retriever = self._get_or_create_retriever("text", self._active_text_method, dataset_name)
        image_retriever = self._get_or_create_retriever("image", self._active_image_method, dataset_name)
        self._retriever = HybridRetriever(text_retriever, image_retriever)
        self._active_dataset = dataset_name
        self.config = {**self.config, "dataset": {
            **self.config["dataset"],
            "name": dataset_name,
            "s3_prefix": self._s3_prefix_for(dataset_name),
        }}

    def _get_or_create_model(self, model_name: str) -> Any:
        """Return a cached model instance or create and cache a new one."""
        if model_name not in self._model_cache:
            print(f"[PipelineService] Initializing model (cache miss): {model_name}")
            cfg = {**self.config, "model": {**self.config["model"], "name": model_name}}
            self._model_cache[model_name] = get_model(cfg)
        else:
            print(f"[PipelineService] Model cache hit: {model_name}")
        return self._model_cache[model_name]

    def _swap_model(self, model_name: str) -> None:
        """Swap the generative model — uses cache, no re-init if already seen."""
        print(f"[PipelineService] Swapping model: {self._active_model} → {model_name}")
        self._model = self._get_or_create_model(model_name)
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

        # Apply config overrides before starting the latency timer — swap cost
        # (retriever re-init, _indexed_ids scroll) is infrastructure overhead,
        # not part of retrieval/generation latency.
        self._ensure_config(request)

        start = time.monotonic()

        # Load query image if provided — query_image_path is always an S3 key
        # uploaded via POST /upload-query-image.
        query_image = None
        if request.query_image_path:
            from io import BytesIO
            from pipeline.utils.s3 import S3Client
            s3 = S3Client(self.config)
            data = s3.download_bytes(request.query_image_path)
            query_image = Image.open(BytesIO(data)).convert("RGB")

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

        token_usage = model_result.token_usage or {}
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
            token_usage=token_usage if token_usage else None,
            cost_usd=self._compute_cost(self._active_model, token_usage),
            storage_info=self._retriever.storage_info() if self._retriever else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def all_storage_info(self) -> StorageOverview:
        """Return all Qdrant collections with sizes and active markers."""
        # Collect active collection names from the current retriever
        active_names: set = set()
        if self._retriever:
            info = self._retriever.storage_info()
            for section in [info.get("text", {}), info.get("image", {})]:
                if isinstance(section, dict) and section.get("collection"):
                    active_names.add(section["collection"])
            # hybrid: nested bm25/dense under text
            text_info = info.get("text", {})
            if isinstance(text_info, dict):
                for sub in [text_info.get("bm25", {}), text_info.get("dense", {})]:
                    if isinstance(sub, dict) and sub.get("collection"):
                        active_names.add(sub["collection"])

        # Get Qdrant client from active retriever
        qdrant_client = None
        if self._retriever:
            for r in [self._retriever.text_retriever, self._retriever.image_retriever]:
                if r and hasattr(r, "qdrant"):
                    qdrant_client = r.qdrant
                    break

        collections: list = []
        if qdrant_client:
            try:
                all_cols = qdrant_client.get_collections().collections
                for col in all_cols:
                    try:
                        col_info = qdrant_client.get_collection(col.name)
                        vectors = col_info.points_count or 0
                        dim = 0
                        try:
                            vec_cfg = col_info.config.params.vectors
                            dim = getattr(vec_cfg, "size", 0) or 0
                        except Exception:
                            pass
                        estimated_mb = round((vectors * dim * 4) / (1024 * 1024), 2) if dim else 0
                        collections.append(QdrantCollectionStats(
                            name=col.name,
                            vectors=vectors,
                            dimension=dim,
                            estimated_mb=estimated_mb,
                            active=col.name in active_names,
                        ))
                    except Exception:
                        pass
            except Exception:
                pass

        # Sort: active first, then by size descending
        collections.sort(key=lambda c: (not c.active, -c.estimated_mb))
        return StorageOverview(collections=collections, active_names=list(active_names))

    def _compute_cost(self, model_name: str, token_usage: dict) -> Optional[float]:
        """Estimate API cost in USD from token counts and pricing table."""
        pricing = _TOKEN_COST_PER_1M.get(model_name)
        if not pricing or not token_usage:
            return None
        input_cost  = (token_usage.get("input_tokens")  or 0) * pricing["input"]  / 1_000_000
        output_cost = (token_usage.get("output_tokens") or 0) * pricing["output"] / 1_000_000
        return round(input_cost + output_cost, 6)

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def index_ready(self) -> bool:
        """Return True if the active retriever has a valid index built by the batch job."""
        if not self._retriever:
            return False
        return self._retriever.is_indexed()

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
