"""Request and response schemas for the RAG API."""

from typing import Dict, List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Single query request to the RAG pipeline.

    Attributes:
        query:              The question to answer.
        ground_truth:       Optional ground truth answer — if provided, metrics are computed.
        query_image_path:   Optional path to a query image (for visual queries).
        model:              Override model (e.g. "gpt", "gemini"). Uses config default if omitted.
        text_method:        Override text retriever (e.g. "hybrid_elastic_qdrant"). Uses config default if omitted.
        image_method:       Override image retriever (e.g. "colpali_qdrant"). Uses config default if omitted.
    """
    query: str
    ground_truth: Optional[str] = None
    query_image_path: Optional[str] = None
    model: Optional[str] = None
    text_method: Optional[str] = None
    image_method: Optional[str] = None
    dataset: Optional[str] = None


class RetrievedTextChunk(BaseModel):
    """A single retrieved text chunk."""
    page_id: str
    text: str
    score: float


class RetrievedImage(BaseModel):
    """A single retrieved image result."""
    page_id: str
    score: float


class QueryResponse(BaseModel):
    """Full response from the RAG pipeline for a single query.

    Attributes:
        query:                  The original query.
        answer:                 Model's predicted answer (cleaned, no Sources line).
        sources:                Source page IDs cited by the model.
        retrieved_text:         Top-k text chunks with scores.
        retrieved_images:       Top-k retrieved image page IDs with scores.
        metrics:                Evaluation metrics — only present if ground_truth was provided.
        latency_ms:             Total end-to-end latency in milliseconds.
        latency_breakdown:      Per-stage latencies in ms: retrieval_ms, generation_ms, evaluation_ms.
        token_usage:            Input and output token counts from the model API.
        cost_usd:               Estimated API cost in USD for this query.
        storage_info:           Active retriever index stats (vectors, size, collection name).
    """
    query: str
    answer: str
    sources: List[str]
    retrieved_text: List[RetrievedTextChunk]
    retrieved_images: List[RetrievedImage]
    metrics: Optional[Dict[str, float]] = None
    latency_ms: float
    latency_breakdown: Optional[Dict[str, float]] = None
    token_usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None
    storage_info: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    initialized: bool
    index_ready: bool
    dataset: str
    text_retriever: str
    image_retriever: str
    model: str


class FeedbackRequest(BaseModel):
    """User feedback on a single RAG response."""
    query: str
    answer: str
    rating: str                          # "positive" | "negative"
    feedback_text: Optional[str] = None
    sources: List[str] = []
    config: Optional[dict] = None        # active dataset/model/retrievers
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class QdrantCollectionStats(BaseModel):
    """Stats for a single Qdrant collection."""
    name: str
    vectors: int
    dimension: int
    estimated_mb: float
    active: bool

class StorageOverview(BaseModel):
    """All Qdrant collections with active markers."""
    collections: List[QdrantCollectionStats]
    active_names: List[str]

class ConfigOptions(BaseModel):
    """Available options for each configurable pipeline component."""
    datasets: List[str]
    text_methods: List[str]
    image_methods: List[str]
    models: List[str]
    active_dataset: str
    active_text_method: str
    active_image_method: str
    active_model: str
