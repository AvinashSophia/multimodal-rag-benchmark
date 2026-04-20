# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A benchmarking pipeline for evaluating multimodal RAG systems that combine text and image retrieval with LLM/VLM generation. Runs entirely on AWS — all data lives in S3, all ML inference services run on EKS.

## Setup

```bash
# Install dependencies (uses uv)
uv sync

# Create .env with required API keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...         # Gemini via Google AI
GOOGLE_APPLICATION_CREDENTIALS=...  # Gemini via Vertex AI

# Port-forward EKS services before running anything
bash scripts/port-forward-benchmarking.sh
```

## Running the Benchmark

```bash
# First run (builds index from S3 data, then evaluates)
RAG_CONFIG=configs/aws.yaml uv run python -m pipeline.runners.run_benchmark_aws --config configs/aws.yaml

# Subsequent runs (index already exists — skips corpus build, evaluates directly)
RAG_CONFIG=configs/aws.yaml uv run python -m pipeline.runners.run_benchmark_aws --config configs/aws.yaml
```

Results are uploaded to S3: `benchmarking/results/{dataset}_{retriever}_{model}_{timestamp}/`

## Data Setup (one-time per dataset)

```bash
# Altumint — parse PDFs and generate QA pairs
aws s3 cp data/altumint/ s3://spatial-ai-staging-processing-632872792182/benchmarking/datasets/altumint/pdfs/ --recursive --include "*.pdf"
uv run python scripts/parse_documents_aws.py --dataset altumint --config configs/aws.yaml
uv run python scripts/generate_qa_aws.py --dataset altumint --config configs/aws.yaml

# HotpotQA / DocVQA / GQA — upload from HuggingFace to S3
uv run python -m scripts.upload_hotpotqa_to_s3 --config configs/aws.yaml
uv run python -m scripts.upload_docvqa_to_s3 --config configs/aws.yaml
uv run python -m scripts.upload_gqa_to_s3 --config configs/aws.yaml
```

## Configuration

All pipeline behavior is controlled by `configs/aws.yaml`:
- **dataset**: `altumint_aws` | `hotpotqa_aws` | `docvqa_aws` | `gqa_aws`
- **retrieval.text.method**: `bm25_elastic_aws` | `dense_qdrant_aws` | `hybrid_elastic_qdrant_aws`
- **retrieval.image.method**: `colpali_qdrant_aws` | `colqwen2_qdrant_aws`
- **model**: `gpt` | `gemini` | `gemini_vertex` | `qwen_vl_aws`

## UI Server

```bash
# Start FastAPI backend
RAG_CONFIG=configs/aws.yaml uvicorn pipeline.api.main:app --reload --port 8000

# Start React frontend (separate terminal)
cd frontend && npm run dev
```

## Architecture

The pipeline runs in 2 paths depending on index state:

**First run:** Load dataset → Build corpus → Index into EKS Qdrant/Elastic → Retrieve → Generate → Evaluate → Upload results to S3

**Subsequent runs:** `is_indexed()` → True → Load QA pairs only → Retrieve from existing index → Generate → Evaluate → Upload results to S3

All major components use a **registry/factory pattern** via `@register_*` decorators. Adding new datasets, retrievers, or models just requires subclassing the base class and registering.

```
pipeline/
├── datasets/       # BaseDataset + 4 AWS loaders (altumint, hotpotqa, docvqa, gqa)
├── retrieval/      # BaseRetriever + 5 AWS retrievers + HybridRetriever
├── models/         # BaseModel + GPT, Gemini, GeminiVertex, QwenVL-AWS
├── evaluation/     # Evaluator + retrieval/answer/grounding/multimodal metrics
├── api/            # FastAPI backend (main.py, pipeline_service.py, schemas.py)
├── runners/        # run_benchmark_aws.py — main orchestrator
└── utils/          # Config loading, S3 client, data classes
```

**Data flow per sample:** `UnifiedSample` → `HybridRetriever` (text + image top-k) → `BaseModel` (answer + sources) → `Evaluator` (5 metric categories) → S3 results.

## Key Data Classes (`pipeline/utils/__init__.py`)

- `UnifiedSample` — standardized format all datasets convert to
- `RetrievalResult` — output from retrievers (text chunks + images with scores)
- `ModelResult` — model output (answer string + source citations)
- `BenchmarkResult` — complete per-sample result including all metrics

## Evaluation Metrics

Five categories computed per sample and aggregated:
- **Retrieval:** Recall@k, MRR, nDCG
- **Answer:** Exact Match, F1, ANLS
- **Grounding:** Faithfulness, Attribution Accuracy
- **Multimodal:** VQA Accuracy (standard VQA protocol), Cross-modal Consistency
