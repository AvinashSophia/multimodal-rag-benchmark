"""Main benchmark runner - end-to-end pipeline execution.

Usage:
    python -m pipeline.runners.run_benchmark --config configs/default.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from networkx import config
from tqdm import tqdm

from pipeline.utils import load_config, setup_output_dirs, save_json, BenchmarkResult
from pipeline.datasets import get_dataset
from pipeline.retrieval import get_retriever, HybridRetriever
from pipeline.models import get_model
from pipeline.evaluation import Evaluator

import random
import numpy as np

from dotenv import load_dotenv
load_dotenv()



def run_benchmark(config: Dict[str, Any]) -> None:
    """Execute the full benchmarking pipeline.

    1. Load dataset → unified format
    2. Build retrieval index
    3. For each sample: retrieve → model → evaluate
    4. Aggregate and save results
    """
    print("=" * 60)
    print("Multimodal RAG Benchmark - Stage 1")
    print("=" * 60)

    random.seed(config["run"]["seed"])
    np.random.seed(config["run"]["seed"])

    # Setup output directory
    run_dir = setup_output_dirs(config)
    save_json(config, run_dir / "config.json")
    print(f"Results will be saved to: {run_dir}")

    # 1. Load dataset
    print(f"\n[1/5] Loading dataset: {config['dataset']['name']}")
    dataset = get_dataset(config)
    dataset.load()
    print(f"  Loaded {len(dataset)} samples")

    # 2. Build retrieval index
    print(f"\n[2/5] Building retrieval index...")
    text_retriever = get_retriever(config, "text")

    image_retriever = None
    if dataset.get_images():
        image_retriever = get_retriever(config, "image")

    hybrid_retriever = HybridRetriever(text_retriever, image_retriever)

    corpus, corpus_ids = dataset.get_corpus()
    images = dataset.get_images()
    hybrid_retriever.index(corpus,  corpus_ids, images if images else None)
    print(f"  Indexed {len(corpus)} text chunks, {len(images)} images")

    # 3. Load model
    print(f"\n[3/5] Loading model: {config['model']['name']}")
    model = get_model(config)

    # 4. Run benchmark
    print(f"\n[4/5] Running benchmark...")
    evaluator = Evaluator(config)
    all_results = []
    all_metrics = []

    text_top_k = config["retrieval"]["text"].get("top_k", 5)
    image_top_k = config["retrieval"]["image"].get("top_k", 5)

    for sample in tqdm(dataset, desc="Benchmarking"):
        # Retrieve
        retrieved = hybrid_retriever.retrieve(
            sample.question, text_top_k=text_top_k, image_top_k=image_top_k
        )

        # Generate answer
        model_result = model.run_model(
            question=sample.question,
            text_context=retrieved.text_chunks,
            image_context=retrieved.images,
            text_ids=retrieved.text_ids,
            image_ids=retrieved.image_ids,
        )

        # Get relevant docs for retrieval metrics 
        relevant_texts = []
        relevant_images = []
        if "supporting_facts" in sample.metadata:
            titles = sample.metadata["supporting_facts"].get("titles", [])
            relevant_texts = [
                doc for doc in sample.text_corpus
                if any(title in doc for title in titles)
            ]

        if sample.images:
            relevant_images  = sample.images

        # Build relevant_sources: ground truth for attribution
        relevant_sources = relevant_texts.copy()
        if sample.images:
            relevant_sources += [f"image_{sample.id}_{i}" for i in range(len(sample.images))]
                
        
        # Evaluate
        metrics = evaluator.evaluate_sample(
            prediction=model_result.answer,
            ground_truth=sample.ground_truth,
            retrieved_texts=retrieved.text_chunks,
            retrieved_text_ids=retrieved.text_ids if retrieved.text_ids else None,
            relevant_texts=relevant_texts if relevant_texts else None,
            relevant_text_ids= None,
            retrieved_images=retrieved.images if retrieved.images else None,
            retrieved_image_ids= None,
            relevant_images=relevant_images if relevant_images else None,
            relevant_image_ids= None,
            used_sources=model_result.sources,
            relevant_sources=relevant_sources if relevant_sources else None,
            all_ground_truths=sample.metadata.get("all_answers", None),
            #text_only_answer=model_result.text_only_answer,
            #image_only_answer=model_result.image_only_answer,
            has_images=bool(sample.images),
        )
        all_metrics.append(metrics)
        print("all_metrics claculated")

        # Store complete result
        result = BenchmarkResult(
            sample_id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            predicted_answer=model_result.answer,
            retrieved_context={
                "text_chunks": retrieved.text_chunks,
                "text_scores": retrieved.text_scores,
                "num_images": len(retrieved.images),
                "image_scores": retrieved.image_scores,
            },
            attribution={
                "used_sources": model_result.sources,
                "relevant_sources": relevant_sources,
                "model_cited_sources": model_result.sources,
            },
            metrics=metrics,
            metadata={"model_metadata": model_result.metadata},
        )
        all_results.append(result)
        print("all_results calculated")

    # 5. Aggregate and save
    print(f"\n[5/5] Saving results...")

    # Save per-sample results
    results_data = [r.to_dict() for r in all_results]
    save_json(results_data, run_dir / "results.json")

    # Aggregate metrics
    aggregated = evaluator.aggregate_metrics(all_metrics)
    save_json(aggregated, run_dir / "metrics.json")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Retriever: {config['retrieval']['text']['method']}")
    print(f"Model: {config['model']['name']}")
    print(f"Samples: {len(dataset)}")
    print("-" * 40)
    for metric, value in sorted(aggregated.items()):
        print(f"  {metric:30s}: {value:.4f}")
    print("=" * 60)
    print(f"\nFull results saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run multimodal RAG benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_benchmark(config)


if __name__ == "__main__":
    main()
