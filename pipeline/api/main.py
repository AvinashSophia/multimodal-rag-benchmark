"""FastAPI application for the multimodal RAG pipeline.

Exposes a single /query endpoint that runs retrieve → generate → evaluate
for one query at a time. Index is built once on startup and reused.

Usage:
    uvicorn pipeline.api.main:app --reload
    uvicorn pipeline.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from datetime import datetime
import json
from pathlib import Path
import uuid
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response

from pipeline.api.schemas import ConfigOptions, FeedbackRequest, HealthResponse, QueryRequest, QueryResponse
from pipeline.api.pipeline_service import PipelineService
from pipeline.utils.s3 import S3Client

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Service singleton — initialized once on startup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_config_path = os.environ.get("RAG_CONFIG", str(_PROJECT_ROOT / "configs" / "aws.yaml"))
service = PipelineService(config_path=_config_path)


def _get_s3() -> S3Client:
    """Return an S3Client configured from the active service config."""
    return S3Client(service.config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline service before serving requests."""
    service.initialize()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal RAG Benchmark API",
    description="Single-query RAG pipeline: retrieve → generate → evaluate.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check whether the pipeline is initialized and ready."""
    return HealthResponse(
        status="ok" if service.is_ready else "initializing",
        initialized=service.is_ready,
        index_ready=service.index_ready,
        dataset=service.dataset_name,
        text_retriever=service.text_retriever_name,
        image_retriever=service.image_retriever_name,
        model=service.model_name,
    )


@app.get("/config/options", response_model=ConfigOptions)
def config_options() -> ConfigOptions:
    """Return available and currently active configuration options."""
    return service.config_options()


@app.post("/upload-query-image")
async def upload_query_image(file: UploadFile = File(...)) -> dict:
    """Upload a query image to S3 and return its S3 key.

    The returned key is passed as query_image_path in a subsequent /query call.
    Accepted formats: JPEG, PNG, WEBP, GIF.
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

    suffix = Path(file.filename or "upload").suffix or ".png"
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    data = await file.read()

    s3 = _get_s3()
    s3_key = s3.query_upload_key(unique_name)
    s3.upload_bytes(data, s3_key, content_type=file.content_type or "image/png")
    return {"path": s3_key}


@app.post("/feedback", status_code=201)
def submit_feedback(request: FeedbackRequest) -> dict:
    """Append a user feedback record to S3 feedback JSONL."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **request.model_dump(),
    }
    s3 = _get_s3()
    s3.append_jsonl(record, s3.feedback_key())
    return {"status": "ok"}


@app.get("/image/{page_id}")
def get_image(page_id: str) -> Response:
    """Fetch a page screenshot from S3 by page_id and stream it back as PNG."""
    s3 = _get_s3()
    s3_prefix = service.config.get("dataset", {}).get("s3_prefix", "altumint")
    s3_key = s3.image_key(s3_prefix, f"figures/{page_id}_page.png")
    try:
        data = s3.download_bytes(s3_key)
        return Response(content=data, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {page_id} ({e})")


@app.get("/storage")
def storage_overview():
    """Return all Qdrant collections with vector counts, sizes, and active markers."""
    return service.all_storage_info()


@app.get("/heatmap")
def heatmap(query: str, page_id: str) -> dict:
    """Generate a ColPali similarity heatmap for a query + retrieved page.

    Returns a base64-encoded PNG showing which image regions were most relevant
    to the query. Only works when the active image retriever is colpali_qdrant_aws.
    """
    from pipeline.retrieval.colpali_qdrant_aws import ColPaliQdrantAWSRetriever
    retriever = service.image_retriever
    if not isinstance(retriever, ColPaliQdrantAWSRetriever):
        raise HTTPException(
            status_code=400,
            detail=f"Heatmap only available with colpali_qdrant_aws retriever. Active: {service.image_retriever_name}",
        )
    heatmap_b64 = retriever.generate_heatmap(query=query, page_id=page_id)
    if heatmap_b64 is None:
        raise HTTPException(status_code=404, detail=f"Heatmap not available for page '{page_id}'. The page may not be indexed or the image could not be fetched.")
    return {"page_id": page_id, "heatmap": heatmap_b64}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run a single query through the RAG pipeline.

    - **query**: The question to answer.
    - **ground_truth**: Optional. If provided, evaluation metrics are returned.
    - **query_image_path**: Optional. S3 key of a query image for visual queries.
    """
    try:
        return service.query(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Image file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
