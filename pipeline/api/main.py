"""FastAPI application for the multimodal RAG pipeline.

Exposes a single /query endpoint that runs retrieve → generate → evaluate
for one query at a time. Index is built once on startup and reused.

Usage:
    uvicorn pipeline.api.main:app --reload
    uvicorn pipeline.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles

from pipeline.api.schemas import ConfigOptions, FeedbackRequest, HealthResponse, QueryRequest, QueryResponse
from pipeline.api.pipeline_service import PipelineService

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Service singleton — initialized once on startup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_UPLOAD_DIR = _PROJECT_ROOT / "data" / "query_uploads"
_FEEDBACK_FILE = _PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"
_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

service = PipelineService(config_path=str(_PROJECT_ROOT / "configs" / "default.yaml"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build retrieval index and load model before serving requests."""
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

# Serve page screenshots so the frontend can display retrieved images.
# Images are accessible at: GET /images/{filename}
_data_dir = _PROJECT_ROOT / "data"
if _data_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_data_dir)), name="images")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check whether the pipeline is initialized and ready."""
    return HealthResponse(
        status="ok" if service._initialized else "initializing",
        initialized=service._initialized,
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
    """Save an uploaded query image and return its server-side path.

    The returned path is passed as query_image_path in a subsequent /query call.
    Accepted formats: JPEG, PNG, WEBP, GIF.
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

    suffix = Path(file.filename or "upload").suffix or ".png"
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"path": str(dest)}


@app.post("/feedback", status_code=201)
def submit_feedback(request: FeedbackRequest) -> dict:
    """Append a user feedback record to data/feedback/feedback.jsonl."""
    import json
    from datetime import datetime
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **request.model_dump(),
    }
    with _FEEDBACK_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "ok"}


@app.get("/heatmap")
def heatmap(query: str, page_id: str) -> dict:
    """Generate a ColPali similarity heatmap for a query + retrieved page.

    Returns a base64-encoded PNG showing which image regions were most relevant
    to the query. Only works when the active image retriever is colpali_qdrant.
    """
    from pipeline.retrieval.colpali_qdrant import ColPaliQdrantRetriever
    retriever = service.image_retriever
    if not isinstance(retriever, ColPaliQdrantRetriever):
        raise HTTPException(
            status_code=400,
            detail=f"Heatmap only available with colpali_qdrant retriever. Active: {service.image_retriever_name}",
        )
    heatmap_b64 = retriever.generate_heatmap(query=query, page_id=page_id)
    if heatmap_b64 is None:
        raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found in image store.")
    return {"page_id": page_id, "heatmap": heatmap_b64}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run a single query through the RAG pipeline.

    - **query**: The question to answer.
    - **ground_truth**: Optional. If provided, evaluation metrics are returned.
    - **query_image_path**: Optional. Path to a query image for visual queries.
    """
    try:
        return service.query(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Image file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
