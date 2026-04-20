"""AWS benchmark runner — reads data from S3, writes results to S3.

AWS variant of pipeline/runners/run_benchmark.py. Pipeline flow, evaluation
logic, and result structure are identical. The only difference is that
results (config.json, results.json, metrics.json) are uploaded to S3
instead of written to local disk.

Smart indexing:
  First run  — dataset.load() + corpus build + index (slow, one-time)
  Subsequent — is_indexed() True → dataset.load_qa_only() (fast, no corpus in memory)

S3 output path:
    benchmarking/results/{run_name}/config.json
    benchmarking/results/{run_name}/results.json
    benchmarking/results/{run_name}/metrics.json

Usage:
    python -m pipeline.runners.run_benchmark_aws --config configs/aws.yaml
"""

import argparse
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from pipeline.utils import load_config, BenchmarkResult
from pipeline.datasets import get_dataset
from pipeline.retrieval import get_retriever, HybridRetriever
from pipeline.models import get_model
from pipeline.evaluation import Evaluator
from pipeline.utils.s3 import S3Client

load_dotenv()


def _make_run_name(config: Dict[str, Any], has_images: bool) -> str:
    """Generate a run directory name — same naming convention as setup_output_dirs()."""
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset      = config["dataset"]["name"]
    text_method  = config["retrieval"]["text"]["method"].replace("_", "")
    model        = config["model"]["name"]

    if has_images:
        image_method = config["retrieval"]["image"]["method"].replace("_", "")
        return f"{dataset}_{text_method}_{image_method}_{model}_{timestamp}"
    return f"{dataset}_{text_method}_{model}_{timestamp}"


def run_benchmark_aws(config: Dict[str, Any]) -> None:
    """Execute the full benchmarking pipeline — results uploaded to S3.

    Smart two-path orchestration:
      - First run:      dataset.load() → build corpus → index → retrieve → evaluate
      - Subsequent run: is_indexed() → dataset.load_qa_only() → retrieve → evaluate

    Steps:
    1. Instantiate retrievers (connect to EKS services; do not index yet)
    2. Check is_indexed() — determines which load path to take
    3a. Already indexed: load_qa_only() — tiny JSON only, no corpus in memory
    3b. Not indexed:     load() + get_corpus() + get_images() + index()
    4. Load model
    5. For each sample: retrieve → model → evaluate
    6. Aggregate and upload results to S3
    """
    print("=" * 60)
    print("Multimodal RAG Benchmark (AWS)")
    print("=" * 60)

    random.seed(config["run"]["seed"])
    np.random.seed(config["run"]["seed"])

    s3 = S3Client(config)

    # 1. Instantiate retrievers (no indexing yet — just connect to EKS services)
    print(f"\n[1/5] Connecting to retrieval services...")
    text_retriever   = get_retriever(config, "text")
    image_retriever  = get_retriever(config, "image")
    hybrid_retriever = HybridRetriever(text_retriever, image_retriever)

    # 2. Check if index already exists (skip-if-indexed path)
    already_indexed = hybrid_retriever.is_indexed()

    if already_indexed:
        print(f"  [Skip] Index exists for "
              f"text={config['retrieval']['text']['method']}, "
              f"image={config['retrieval']['image']['method']}, "
              f"dataset={config['dataset']['name']}")

    # 3. Load dataset — full load or QA-only depending on index state
    print(f"\n[2/5] Loading dataset: {config['dataset']['name']}")
    dataset = get_dataset(config)

    images: List[Any] = []
    image_ids: List[str] = []

    if already_indexed:
        # Fast path: only load QA pairs (tiny JSON), no corpus/images in memory
        dataset.load_qa_only()
        has_images = image_retriever.is_indexed()
    else:
        # First-run path: full load, build corpus, index everything
        dataset.load()
        images, image_ids = dataset.get_images()
        corpus, corpus_ids = dataset.get_corpus()
        has_images = bool(images)

        print(f"\n[2b/5] Building retrieval index "
              f"({len(corpus)} text chunks, {len(images)} images)...")
        hybrid_retriever.index(
            corpus, corpus_ids,
            images    if images    else None,
            image_ids if image_ids else None,
        )

    print(f"  Loaded {len(dataset)} samples")

    # Run name and S3 config upload
    run_name = _make_run_name(config, has_images=has_images)
    print(f"  Run name: {run_name}")
    print(f"  Results will be uploaded to: s3://{s3.bucket}/{s3.results_key(run_name, '')}")
    s3.upload_json(config, s3.results_key(run_name, "config.json"))

    # 4. Load model
    print(f"\n[3/5] Loading model: {config['model']['name']}")
    model = get_model(config)

    # 5. Run benchmark
    print(f"\n[4/5] Running benchmark...")
    evaluator   = Evaluator(config)
    all_results = []
    all_metrics = []

    text_top_k  = config["retrieval"]["text"].get("top_k", 5)
    image_top_k = config["retrieval"]["image"].get("top_k", 5)

    for sample in tqdm(dataset, desc="Benchmarking"):
        query_image = sample.images[0] if sample.images else None
        retrieved = hybrid_retriever.retrieve(
            sample.question,
            text_top_k=text_top_k,
            image_top_k=image_top_k,
            query_image=query_image,
        )

        model_result = model.run_model(
            question=sample.question,
            text_context=retrieved.text_chunks,
            image_context=retrieved.images,
            text_ids=retrieved.text_ids,
            image_ids=retrieved.image_ids,
        )

        relevant_texts:     List[str] = []
        relevant_text_ids:  List[str] = []
        relevant_images:    List[Any] = []

        if "supporting_facts" in sample.metadata:
            titles = sample.metadata["supporting_facts"].get("titles", [])
            for doc in sample.text_corpus:
                if any(doc.startswith(f"[{title}]") for title in titles):
                    relevant_texts.append(doc)
                    relevant_text_ids.append(doc.split("]")[0].replace("[", "").strip())
        elif "relevant_text_ids" in sample.metadata:
            relevant_text_ids = list(sample.metadata["relevant_text_ids"])

        if sample.images:
            relevant_images = sample.images

        relevant_image_ids = sample.image_ids

        relevant_sources = relevant_text_ids.copy()
        if sample.images:
            relevant_sources += relevant_image_ids

        metrics = evaluator.evaluate_sample(
            prediction=model_result.answer,
            ground_truth=sample.ground_truth,
            retrieved_texts=retrieved.text_chunks,
            retrieved_text_ids=retrieved.text_ids if retrieved.text_ids else None,
            relevant_texts=relevant_texts if relevant_texts else None,
            relevant_text_ids=relevant_text_ids if relevant_text_ids else None,
            retrieved_images=retrieved.images if retrieved.images else None,
            retrieved_image_ids=retrieved.image_ids if retrieved.image_ids else None,
            relevant_images=relevant_images if relevant_images else None,
            relevant_image_ids=relevant_image_ids if relevant_image_ids else None,
            used_sources=model_result.sources,
            relevant_sources=relevant_sources if relevant_sources else None,
            all_ground_truths=sample.metadata.get("all_answers", None),
            has_images=bool(sample.images),
        )
        all_metrics.append(metrics)

        result = BenchmarkResult(
            sample_id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            raw_answer=model_result.raw_response,
            predicted_answer=model_result.answer,
            retrieved_context={
                "text_chunks":  retrieved.text_chunks,
                "text_ids":     retrieved.text_ids,
                "text_scores":  retrieved.text_scores,
                "num_images":   len(retrieved.images),
                "image_ids":    retrieved.image_ids,
                "image_scores": retrieved.image_scores,
            },
            attribution={
                "used_sources":        model_result.sources,
                "relevant_sources":    relevant_sources,
                "model_cited_sources": model_result.sources,
            },
            metrics=metrics,
            metadata={"model_metadata": model_result.metadata},
        )
        all_results.append(result)

    # 6. Upload results to S3
    print(f"\n[5/5] Uploading results to S3...")

    results_data = [r.to_dict() for r in all_results]
    s3.upload_json(results_data, s3.results_key(run_name, "results.json"))

    aggregated = evaluator.aggregate_metrics(all_metrics)
    s3.upload_json(aggregated, s3.results_key(run_name, "metrics.json"))

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset:         {config['dataset']['name']}")
    print(f"Text Retriever:  {config['retrieval']['text']['method']}")
    if has_images:
        print(f"Image Retriever: {config['retrieval']['image']['method']}")
    print(f"Model:           {config['model']['name']}")
    print(f"Samples:         {len(dataset)}")
    print(f"Index reused:    {'Yes (skipped corpus build)' if already_indexed else 'No (built from scratch)'}")
    print("-" * 40)
    for metric, value in sorted(aggregated.items()):
        print(f"  {metric:30s}: {value:.4f}")
    print("=" * 60)
    print(f"\nResults uploaded to: s3://{s3.bucket}/{s3.results_key(run_name, '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal RAG benchmark (AWS — results to S3)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/aws.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_benchmark_aws(config)


if __name__ == "__main__":
    main()
