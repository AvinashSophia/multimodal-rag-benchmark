"""Production retrieval metrics using ranx (industry-standard IR library).

ranx provides highly optimized implementations of standard IR metrics.
Drop-in alternative to retrieval_metrics.py with identical interface.

Install: pip install ranx
Docs: https://amenra.github.io/ranx/
"""

from typing import Dict, List

try:
    from ranx import Qrels, Run, evaluate as ranx_evaluate
    RANX_AVAILABLE = True
except ImportError:
    RANX_AVAILABLE = False


def compute_retrieval_metrics_ranx(
    retrieved: List[str],
    relevant: List[str],
    k_values: List[int] = [5],
    query_id: str = "q0",
) -> Dict[str, float]:
    """Compute retrieval metrics using ranx.

    Args:
        retrieved: Ordered list of retrieved document IDs (best first).
        relevant: List of ground-truth relevant document IDs.
        k_values: List of k values for Recall@k and nDCG@k.
        query_id: Internal query identifier (used by ranx internals).

    Returns:
        Dictionary of metric_name -> score, same keys as compute_retrieval_metrics().
    """
    if not RANX_AVAILABLE:
        raise ImportError(
            "ranx is not installed. Run: pip install ranx"
        )

    if not relevant or not retrieved:
        result = {"mrr": 0.0}
        for k in k_values:
            result[f"recall@{k}"] = 0.0
            result[f"ndcg@{k}"] = 0.0
        return result

    # ranx expects dicts: {query_id: {doc_id: relevance_score}}
    qrels_dict = {query_id: {doc_id: 1 for doc_id in relevant}}
    # Scores = inverse rank (ranx ranks by score descending)
    run_dict = {
        query_id: {doc_id: float(len(retrieved) - i) for i, doc_id in enumerate(retrieved)}
    }

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    # Build metric strings ranx understands: "mrr", "recall@5", "ndcg@5"
    metric_names = ["mrr"]
    for k in k_values:
        metric_names.append(f"recall@{k}")
        metric_names.append(f"ndcg@{k}")

    scores = ranx_evaluate(qrels, run, metric_names, make_comparable=True)

    metrics: Dict[str, float] = {}
    for name in metric_names:
        # ranx returns averages across queries; for single query this is the raw score
        metrics[name] = float(scores[name])

    return metrics
