"""Retrieval evaluation metrics: Recall@k, MRR, nDCG."""

import numpy as np
from typing import Dict, List, Any


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Recall@k - fraction of relevant docs found in top-k retrieved.

    Args:
        retrieved: List of retrieved document texts/ids (ordered by rank).
        relevant: List of ground-truth relevant document texts/ids.
        k: Number of top results to consider.

    Returns:
        Recall@k score between 0 and 1.
    """
    if not relevant:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_at_k & relevant_set) / len(relevant_set)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Mean Reciprocal Rank.

    MRR = 1 / rank_of_first_relevant_result.
    If no relevant result is found, returns 0.

    Args:
        retrieved: List of retrieved document texts/ids (ordered by rank).
        relevant: List of ground-truth relevant document texts/ids.

    Returns:
        MRR score between 0 and 1.
    """
    relevant_set = set(relevant)
    for i, doc in enumerate(retrieved):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Measures ranking quality - relevant docs ranked higher get more credit.

    Args:
        retrieved: List of retrieved document texts/ids (ordered by rank).
        relevant: List of ground-truth relevant document texts/ids.
        k: Number of top results to consider.

    Returns:
        nDCG@k score between 0 and 1.
    """
    relevant_set = set(relevant)

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = 1.0 if doc in relevant_set else 0.0
        dcg += rel / np.log2(i + 2)  # +2 because rank starts at 1, log2(1)=0

    # Ideal DCG: all relevant docs ranked first
    ideal_length = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved: List[str],
    relevant: List[str],
    k_values: List[int] = [5],
) -> Dict[str, float]:
    """Compute all retrieval metrics.

    Args:
        retrieved: List of retrieved document texts/ids.
        relevant: List of ground-truth relevant document texts/ids.
        k_values: List of k values for Recall@k and nDCG@k.

    Returns:
        Dictionary of metric_name -> score.
    """
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
        metrics[f"mrr@{k}"] = mrr(retrieved, relevant)

    return metrics
