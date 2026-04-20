# Multimodal RAG Benchmark

A benchmarking pipeline for evaluating multimodal Retrieval-Augmented Generation (RAG) systems that combine text and image retrieval with LLM/VLM generation. Runs entirely on AWS — all data lives in S3, all ML inference services run on EKS.

The pipeline is split into two completely separate phases:

- **Phase 1 — Offline Indexing (once per dataset):** Parse documents, encode corpus, store in Qdrant/Elasticsearch. Done once. Survives pod restarts, port-forward drops, and node replacements because Qdrant and ES are backed by persistent EBS volumes on EKS.
- **Phase 2 — Benchmark Loop (every run):** Load QA pairs, retrieve from existing index, generate answers, evaluate, upload results to S3. Fast and repeatable — only `max_samples` rows are processed each run.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Two-Phase Design](#two-phase-design)
- [EKS Services](#eks-services)
- [S3 Layout](#s3-layout)
- [Setup](#setup)
- [Data Pipeline](#data-pipeline)
- [Running the Benchmark](#running-the-benchmark)
- [UI Server](#ui-server)
- [Flow Diagrams](#flow-diagrams)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Adding New Components](#adding-new-components)

---

## Architecture Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        UI["React Frontend\n(Vite · localhost:5173)"]
    end

    subgraph API["API Layer"]
        FastAPI["FastAPI Backend\n(uvicorn · localhost:8000)"]
    end

    subgraph Pipeline["Pipeline Layer"]
        HR["HybridRetriever"]
        TR["Text Retriever\nbm25_elastic_aws\ndense_qdrant_aws\nhybrid_elastic_qdrant_aws"]
        IR["Image Retriever\ncolpali_qdrant_aws\ncolqwen2_qdrant_aws"]
        Model["Generative Model\nGPT-4o · Gemini · QwenVL"]
        Eval["Evaluator\nRetrieval · Answer · Grounding · Multimodal"]
    end

    subgraph EKS["EKS Services (port-forwarded · persistent EBS volumes)"]
        ES["Elasticsearch\n:9200 — BM25 index"]
        Qdrant["Qdrant\n:6333 — Vector store"]
        BGE["BGE-large TEI\n:8112 — Text embeddings"]
        ColPali["ColPali service\n:8110 — Patch embeddings"]
        ColQwen2["ColQwen2 service\n:8111 — Patch embeddings"]
    end

    subgraph S3["S3 — spatial-ai-staging-processing-632872792182"]
        DS["benchmarking/datasets/\nPDFs · parsed JSONs · QA pairs · corpus"]
        IMG["benchmarking/images/\nPage screenshots · figure crops · query uploads"]
        RES["benchmarking/results/\nconfig · results · metrics per run"]
        FB["benchmarking/feedback/\nfeedback.jsonl"]
    end

    UI -->|"HTTP"| FastAPI
    FastAPI --> HR
    HR --> TR & IR
    TR -->|"BM25 search"| ES
    TR -->|"dense ANN"| Qdrant
    TR -->|"embed text"| BGE
    IR -->|"patch ANN"| Qdrant
    IR -->|"embed image/text"| ColPali & ColQwen2
    HR --> Model
    Model -->|"OpenAI / Google / Qwen API"| Model
    Model --> Eval
    FastAPI -->|"images on demand"| S3
    FastAPI -->|"feedback"| S3
    Eval -->|"results upload"| S3
```

---

## Two-Phase Design

### Phase 1 — Offline Indexing (run once per dataset)

```mermaid
flowchart LR
    S3[(S3\nDataset)] --> Load["dataset.load()\nFull corpus — never truncated\nby max_samples"]
    Load --> Index["HybridRetriever.index()\nEncode via BGE / ColPali\nStore in Qdrant + Elastic"]
    Index --> Sentinel["Write metadata sentinel\ncorpus_count / page_count\n→ marks index complete"]
    Sentinel --> Done[("EKS\nPersistent index\nSurvives pod restarts")]
```

Key properties:
- `max_samples` **never affects** what gets indexed — always the full corpus
- A metadata sentinel is written only after the full corpus is successfully indexed
- `is_indexed()` checks `sentinel exists AND count > 0` — partial indexes (killed mid-run) are detected and re-indexed cleanly
- docvqa/gqa build the full image corpus from all QA pairs before applying `max_samples` to eval samples

### Phase 2 — Benchmark Loop (every run)

```mermaid
flowchart LR
    Check{"is_indexed()?"}
    Check -->|True| QAOnly["dataset.load_qa_only()\nQA pairs only — tiny JSON\nno corpus in memory"]
    Check -->|False| Full["dataset.load()\n→ index() → write sentinel"]
    QAOnly --> Eval["Retrieve → Generate → Evaluate\nonly max_samples rows"]
    Full --> Eval
    Eval --> S3[(S3\nResults)]
```

From the second run onwards, for any `max_samples` value:
- Index is reused as-is (persistent EBS)
- Only QA pairs are loaded into memory
- Retrieval hits the existing Qdrant/ES collections directly
- Results are uploaded to S3 per run

This means you can freely change `max_samples`, the model, or the evaluation config and re-run in seconds without ever paying the indexing cost again.

---

## EKS Services

| Service | Port | Purpose | Model |
|---|---|---|---|
| Elasticsearch | `9200` | BM25 full-text index | — |
| Qdrant | `6333` | Vector ANN store (text + image) | — |
| BGE-large TEI | `8112` | Text embeddings for dense retrieval | `BAAI/bge-large-en-v1.5` (1024-dim) |
| ColPali | `8110` | Patch-level image embeddings | `vidore/colpali-v1.3` (128-dim) |
| ColQwen2 | `8111` | Patch-level image embeddings | `vidore/colqwen2-v1.0` (128-dim) |

All services are backed by persistent EBS volumes — data survives pod restarts. Access via `kubectl port-forward` to localhost:

```bash
bash scripts/port-forward-benchmarking.sh
```

---

## S3 Layout

```
spatial-ai-staging-processing-632872792182/
└── benchmarking/
    ├── datasets/
    │   └── {dataset}/
    │       ├── pdfs/           ← raw PDFs (upload source for parse_documents_aws.py)
    │       ├── parsed/         ← per-page JSONs (text + table + figure metadata)
    │       ├── corpus.jsonl    ← deduplicated text passages (hotpotqa only)
    │       └── qa_pairs.json   ← question/answer pairs
    ├── images/
    │   └── {dataset}/
    │       └── figures/        ← page screenshots (*_page.png) + figure crops
    ├── results/
    │   └── {run_name}/
    │       ├── config.json
    │       ├── results.json
    │       └── metrics.json
    └── feedback/
        └── feedback.jsonl
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure API keys

```bash
# .env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...                      # Gemini via Google AI Studio
GOOGLE_APPLICATION_CREDENTIALS=...     # Gemini via Vertex AI
QWEN_VL_API_KEY=...                    # Self-hosted Qwen endpoint
```

### 3. Port-forward EKS services

```bash
bash scripts/port-forward-benchmarking.sh
```

---

## Data Pipeline

Run once per dataset to populate S3. After this the benchmark runner never needs local data.

```mermaid
flowchart LR
    subgraph Altumint
        direction TB
        A1["Upload PDFs to S3\naws s3 cp data/altumint/ s3://.../pdfs/"]
        A2["Parse PDFs\nparse_documents_aws.py\n(Docling · runs on EKS)"]
        A3["Generate QA pairs\ngenerate_qa_aws.py\n(GPT-4o)"]
        A1 --> A2 --> A3
    end

    subgraph Public["HotpotQA / DocVQA / GQA"]
        direction TB
        B1["Download from HuggingFace\n+ upload to S3"]
        B1
    end

    A3 -->|"S3: parsed/ + qa_pairs.json + images/"| Ready[("S3\nReady for benchmark")]
    B1 -->|"S3: corpus.jsonl / qa_pairs.json / images/"| Ready
```

### Altumint (proprietary PDFs)

```bash
# 1. Upload PDFs to S3
aws s3 cp data/altumint/ s3://spatial-ai-staging-processing-632872792182/benchmarking/datasets/altumint/pdfs/ \
  --recursive --include "*.pdf"

# 2. Parse (Docling) — outputs per-page JSONs + images to S3
uv run python scripts/parse_documents_aws.py --dataset altumint --config configs/aws.yaml

# 3. Generate QA pairs — outputs qa_pairs.json to S3
uv run python scripts/generate_qa_aws.py --dataset altumint --config configs/aws.yaml
```

### HotpotQA / DocVQA / GQA (public datasets)

```bash
uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml
uv run python -m scripts.upload_docvqa_to_s3   --config configs/aws.yaml
uv run python -m scripts.upload_gqa_to_s3      --config configs/aws.yaml
```

---

## Running the Benchmark

```mermaid
flowchart TD
    Start([run_benchmark_aws.py]) --> Connect["Connect to EKS services\n(Qdrant · Elasticsearch · BGE · ColPali)"]
    Connect --> Check{"is_indexed()?"}

    Check -->|"False — first run"| Load["dataset.load()\nFull corpus + images from S3\n(max_samples ignored for corpus)"]
    Load --> Index["HybridRetriever.index()\nEncode + store in Qdrant / Elastic\nWrite metadata sentinel"]
    Index --> QA

    Check -->|"True — any subsequent run"| QAOnly["dataset.load_qa_only()\nQA pairs only — no corpus in memory"]
    QAOnly --> QA

    QA["For each QA sample\n(up to max_samples)"] --> Retrieve["HybridRetriever.retrieve()\ntext top-k + image top-k"]
    Retrieve --> Generate["Model.run_model()\nGPT-4o / Gemini / QwenVL"]
    Generate --> Evaluate["Evaluator.evaluate_sample()\n5 metric categories"]
    Evaluate --> More{more samples?}
    More -->|yes| QA
    More -->|no| Aggregate["Aggregate metrics\nUpload results to S3"]
```

```bash
RAG_CONFIG=configs/aws.yaml uv run python -m pipeline.runners.run_benchmark_aws --config configs/aws.yaml
```

**First run** — builds index from full corpus (slow, encoding all pages/chunks). Subsequent runs reuse the same index.

**Subsequent runs** — `is_indexed()` finds the metadata sentinel → skips corpus load and encoding entirely → straight to retrieval and evaluation. Fast regardless of `max_samples`.

**Changing `max_samples`** — never triggers re-indexing. The index is always the full corpus; `max_samples` only controls how many QA pairs are evaluated each run.

---

## UI Server

```mermaid
sequenceDiagram
    actor User
    participant Frontend as React Frontend<br/>:5173
    participant API as FastAPI<br/>:8000
    participant Retriever as HybridRetriever
    participant EKS as EKS Services
    participant S3
    participant Model as LLM/VLM

    User->>Frontend: Type query + select config
    Frontend->>API: POST /query
    API->>Retriever: retrieve(query, top_k)
    Retriever->>EKS: BM25 search (Elasticsearch)
    Retriever->>EKS: Dense ANN (Qdrant)
    Retriever->>EKS: Patch embedding (ColPali/ColQwen2)
    EKS-->>Retriever: text chunks + image page_ids
    Retriever-->>API: RetrievalResult
    API->>Model: run_model(query, text_context, images)
    Model-->>API: answer + sources
    API-->>Frontend: QueryResponse (answer, retrieved, metrics, latency)

    Frontend->>API: GET /image/{page_id}
    API->>S3: download_bytes(image key)
    S3-->>API: PNG bytes
    API-->>Frontend: image/png

    opt ColPali retriever active
        User->>Frontend: Click heatmap button
        Frontend->>API: GET /heatmap?query=...&page_id=...
        API->>EKS: embed text + embed image patches
        EKS-->>API: token vectors + patch vectors
        API-->>Frontend: base64 PNG heatmap overlay
    end

    opt User submits feedback
        User->>Frontend: Rate answer
        Frontend->>API: POST /feedback
        API->>S3: append_jsonl(feedback.jsonl)
    end
```

```bash
# Terminal 1 — FastAPI backend
RAG_CONFIG=configs/aws.yaml uvicorn pipeline.api.main:app --reload --port 8000

# Terminal 2 — React frontend
cd frontend && npm run dev
```

Open `http://localhost:5173`

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Pipeline status and active component names |
| `GET` | `/config/options` | Available datasets, retrievers, models |
| `POST` | `/query` | Run full RAG pipeline for one query |
| `GET` | `/image/{page_id}` | Fetch page screenshot from S3 |
| `GET` | `/heatmap` | ColPali similarity heatmap (PNG base64) |
| `POST` | `/upload-query-image` | Upload visual query image → S3 key |
| `POST` | `/feedback` | Append feedback record to S3 JSONL |
| `GET` | `/storage` | Qdrant collection stats |

---

## Flow Diagrams

### Offline Indexing vs Benchmark Loop

```mermaid
sequenceDiagram
    participant Runner as run_benchmark_aws.py
    participant Dataset as Dataset Loader
    participant S3
    participant Retriever as HybridRetriever
    participant EKS as EKS (Qdrant / Elastic / BGE / ColPali)
    participant Model as LLM/VLM

    Runner->>EKS: Connect retrievers
    Runner->>EKS: is_indexed()? (check metadata sentinel)

    alt First run — sentinel does not exist
        EKS-->>Runner: False
        Runner->>Dataset: load()
        Dataset->>S3: download full corpus + all images
        Note over Dataset: max_samples ignored for corpus
        S3-->>Dataset: corpus texts + PIL images
        Dataset-->>Runner: full corpus, all images, QA samples
        Runner->>EKS: index(corpus, images)
        Note over EKS: Encode via BGE / ColPali<br/>Store in Qdrant + Elastic<br/>Write metadata sentinel
    else Any subsequent run — sentinel exists
        EKS-->>Runner: True
        Runner->>Dataset: load_qa_only()
        Dataset->>S3: download qa_pairs.json only
        S3-->>Dataset: QA pairs (tiny)
        Dataset-->>Runner: QA samples only (no corpus in memory)
    end

    loop For each QA sample (up to max_samples)
        Runner->>EKS: retrieve(query, top_k)
        EKS-->>Runner: text chunks + image page_ids
        Runner->>S3: fetch images by page_id (on demand)
        S3-->>Runner: PIL images
        Runner->>Model: run_model(query, context, images)
        Model-->>Runner: answer + sources
        Runner->>Runner: evaluate_sample(prediction, ground_truth)
    end

    Runner->>S3: upload results (config + results + metrics)
```

### Hybrid Retrieval — RRF Fusion

```mermaid
flowchart TD
    Q["Query text"] --> BM25["BM25ElasticAWS\nkeyword matching"]
    Q --> Dense["DenseQdrantAWS\nBGE-large embed → ANN"]
    Q --> ColPali["ColPaliQdrantAWS\nper-token MaxSim over patches"]

    BM25 -->|"top-20 text chunks"| RRF["RRF Fusion\n1/(k + rank)"]
    Dense -->|"top-20 text chunks"| RRF
    RRF -->|"top-5 text chunks"| Merge

    ColPali -->|"top-5 page images"| Overlap["Cross-modal overlap boost\npage_id in both → promoted to front"]
    RRF --> Overlap
    Overlap --> Merge["RetrievalResult\ntext_chunks + images + scores"]
```

### ColPali Heatmap Generation

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant S3
    participant ColPali as ColPali Service (EKS :8110)

    API->>S3: fetch page image by page_id
    S3-->>API: PNG bytes → PIL Image
    API->>ColPali: POST /embed/images [base64 PNG]
    ColPali-->>API: patch embeddings (n_patches × 128)
    API->>ColPali: POST /embed/text [query]
    ColPali-->>API: token embeddings (n_tokens × 128)
    Note over API: similarity = token_vecs @ patch_vecs.T<br/>aggregated = max over tokens (MaxSim)<br/>reshape to √n × √n grid
    API->>API: normalize + overlay on original image
    API-->>API: base64 PNG heatmap
```

### Sentinel-Based Index State Detection

```mermaid
flowchart LR
    subgraph Qdrant["Qdrant Collection"]
        P1["patch_0 · page_id=p1"]
        P2["patch_1 · page_id=p1"]
        Pn["... N patches ..."]
        S["SENTINEL\ntype=metadata\npage_count=K"]
    end

    subgraph ES["Elasticsearch Index"]
        D1["doc · text=... · doc_id=c1"]
        Dn["... M docs ..."]
        SE["SENTINEL _id=__bm25_index_metadata__\ncorpus_count=M"]
    end

    Check["is_indexed()"] -->|"retrieve sentinel"| Qdrant
    Check -->|"get sentinel doc"| ES
    Qdrant -->|"page_count > 0 → True"| Result["Index ready\nSkip to retrieval"]
    ES -->|"corpus_count > 0 → True"| Result
```

---

## Configuration

All pipeline behavior is controlled by `configs/aws.yaml`.

```yaml
dataset:
  name: "altumint_aws"          # altumint_aws | hotpotqa_aws | docvqa_aws | gqa_aws
  s3_prefix: "altumint"         # S3 key prefix (matches parse/generate scripts)
  max_samples: null             # null = evaluate full QA set
                                # integer = evaluate N QA pairs (index always full corpus)

retrieval:
  text:
    method: "hybrid_elastic_qdrant_aws"   # bm25_elastic_aws | dense_qdrant_aws | hybrid_elastic_qdrant_aws
    top_k: 5
  image:
    method: "colpali_qdrant_aws"          # colpali_qdrant_aws | colqwen2_qdrant_aws
    top_k: 5

model:
  name: "gemini_vertex"         # gpt | gemini | gemini_vertex | qwen_vl_aws

evaluation:
  backend: "production"         # custom (fast, no deps) | production (ranx + HF evaluate + RAGAS)
```

**`max_samples` semantics:** controls how many QA pairs are evaluated per run. The corpus indexed is always 100% of the dataset — changing `max_samples` never triggers re-indexing.

---

## Evaluation Metrics

| Category | Metrics |
|---|---|
| **Retrieval** | Recall@k, MRR, nDCG |
| **Answer** | Exact Match, F1, ANLS |
| **Grounding** | Faithfulness (RAGAS LLM-as-judge), Attribution Accuracy |
| **Multimodal** | VQA Accuracy (standard VQA protocol), Cross-modal Consistency |

Two backends selectable via `evaluation.backend`:
- `custom` — fast numpy/regex implementations, no API calls, good for iteration
- `production` — ranx (IR metrics) + HuggingFace evaluate (EM/F1) + RAGAS (faithfulness)

---

## Adding New Components

All components use a **registry/factory pattern** — add a subclass, register it, done.

### New Dataset

```python
from pipeline.datasets.base import BaseDataset, register_dataset

@register_dataset("my_dataset_aws")
class MyDatasetAWS(BaseDataset):
    def load(self) -> None:
        # Load full corpus (ALL items — never truncate by max_samples here)
        # Apply max_samples only to self.samples for evaluation
        ...
    def load_qa_only(self) -> None:
        # Load only QA pairs JSON — no corpus, no images in memory
        ...
    def get_corpus(self): ...
    def get_images(self): ...
```

Import in `pipeline/datasets/__init__.py`. Add to `_AVAILABLE_DATASETS` in `pipeline_service.py`.

### New Retriever

```python
from pipeline.retrieval.base import BaseRetriever, register_retriever

@register_retriever("my_retriever_aws")
class MyRetrieverAWS(BaseRetriever):
    _METADATA_ID = ...  # unique sentinel ID

    def is_indexed(self) -> bool:
        # Check sentinel exists AND count > 0
        # Do NOT load corpus to check this
        ...
    def index(self, corpus, corpus_ids):
        # Encode and store full corpus
        # Write sentinel ONLY after successful completion
        ...
    def retrieve(self, query, top_k, query_image):
        # Exclude sentinel from results (must_not type=metadata)
        ...
    def storage_info(self): ...
```

Import in `pipeline/retrieval/__init__.py`. Add to `_AVAILABLE_TEXT_METHODS` or `_AVAILABLE_IMAGE_METHODS` in `pipeline_service.py`.

### New Model

```python
from pipeline.models.base import BaseModel, register_model

@register_model("my_model_aws")
class MyModelAWS(BaseModel):
    def run_model(self, question, text_context, image_context, ...) -> ModelResult: ...
```

Import in `pipeline/models/__init__.py`. Add to `_AVAILABLE_MODELS` in `pipeline_service.py`.
