"""FastAPI application for the multimodal RAG pipeline.

Exposes a single /query endpoint that runs retrieve → generate → evaluate
for one query at a time. Index is built once on startup and reused.

Usage:
    uvicorn pipeline.api.main:app --reload
    uvicorn pipeline.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pipeline.api.schemas import ConfigOptions, HealthResponse, QueryRequest, QueryResponse
from pipeline.api.pipeline_service import PipelineService

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Service singleton — initialized once on startup
# ---------------------------------------------------------------------------

service = PipelineService(config_path="configs/default.yaml")


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
_data_dir = Path("data")
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
