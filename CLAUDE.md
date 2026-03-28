# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A benchmarking pipeline for evaluating multimodal RAG systems that combine text and image retrieval with LLM/VLM generation. Tests how well systems retrieve and answer questions using both text and visual evidence.

## Setup

```bash
# Install dependencies (uses uv)
uv sync

# Create .env with required API keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...   # Only if using Gemini
```

## Running the Benchmark

```bash
# Run with default config
python -m pipeline.runners.run_benchmark

# Run with custom config
python -m pipeline.runners.run_benchmark --config configs/default.yaml
```

For quick validation, set `max_samples: 5` in `configs/default.yaml` before running.

## Configuration

All pipeline behavior is controlled by `configs/default.yaml`:
- **dataset**: `hotpotqa` | `docvqa` | `gqa`
- **retrieval.text_method**: `bm25` | `dense`
- **retrieval.image_method**: `clip`
- **model**: `gpt` | `gemini` | `qwen_vl`

Results are saved to `pipeline/outputs/logs/{dataset}_{retriever}_{model}_{timestamp}/` with `config.json`, `results.json`, and `metrics.json`.

## Architecture

The pipeline runs in 5 stages: Load Dataset → Build Retrieval Index → Load Model → Retrieve+Generate+Evaluate per sample → Aggregate Results.

All major components use a **registry/factory pattern** via `@register_*` decorators. Adding new datasets, retrievers, or models just requires subclassing the base class and registering.

```
pipeline/
├── datasets/       # BaseDataset + HotpotQA, DocVQA, GQA loaders
├── retrieval/      # BaseRetriever + BM25, Dense, CLIP, HybridRetriever
├── models/         # BaseModel + GPT, Gemini, QwenVL wrappers
├── evaluation/     # Evaluator + retrieval/answer/grounding/multimodal metrics
├── runners/        # run_benchmark.py — main orchestrator
└── utils/          # Config loading, data classes (UnifiedSample, RetrievalResult, ModelResult, BenchmarkResult)
```

**Data flow per sample:** `UnifiedSample` → `HybridRetriever` (text + image top-k) → `BaseModel` (answer + sources) → `Evaluator` (5 metric categories) → saved result.

## Key Data Classes (`pipeline/utils/__init__.py`)

- `UnifiedSample` — standardized format all datasets convert to
- `RetrievalResult` — output from retrievers (text chunks + images with scores)
- `ModelResult` — model output (answer string + source citations)
- `BenchmarkResult` — complete per-sample result including all metrics


## Evaluation Metrics

Five categories computed per sample and aggregated:
- **Retrieval:** Recall@k, MRR, nDCG
- **Answer:** Exact Match, F1
- **Grounding:** Faithfulness, Attribution Accuracy
- **Multimodal:** VQA Accuracy (standard VQA protocol), Cross-modal Consistency
