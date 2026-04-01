"""Production grounding metrics using RAGAS (LLM-as-judge faithfulness).

RAGAS uses an LLM to evaluate whether the generated answer is faithful to
the retrieved context — far more accurate than token-overlap heuristics.

Drop-in alternative to grounding_metrics.py with identical interface.

Install: pip install ragas
Docs: https://docs.ragas.io
"""

import os
from typing import Dict, List, Optional

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def compute_grounding_metrics_ragas(
    answer: str,
    retrieved_context: List[str],
    used_sources: Optional[List[str]] = None,
    relevant_sources: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute grounding metrics using RAGAS LLM-as-judge faithfulness.

    RAGAS decomposes the answer into atomic statements and checks each one
    against the retrieved context using an LLM judge — much more reliable
    than the heuristic token-overlap approach.

    Attribution accuracy falls back to the standard F1 implementation since
    RAGAS does not provide a source-citation metric out of the box.

    Args:
        answer: Model's generated answer.
        retrieved_context: Retrieved text chunks (must be non-empty for faithfulness).
        used_sources: Sources cited by the model.
        relevant_sources: Ground-truth relevant sources.

    Returns:
        Dictionary with "faithfulness" and optionally "attribution_accuracy".

    Requires:
        OPENAI_API_KEY env var set (RAGAS uses OpenAI by default as the judge LLM).
        Set RAGAS_LLM env var to override (e.g., "gpt-4o-mini" for cheaper evals).
    """
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS is not installed. Run: pip install ragas"
        )

    metrics: Dict[str, float] = {}

    # --- Faithfulness via RAGAS ---
    if answer and retrieved_context:
        sample = SingleTurnSample(
            user_input="",          # Not needed for faithfulness
            response=answer,
            retrieved_contexts=retrieved_context,
        )
        dataset = EvaluationDataset(samples=[sample])
        result = ragas_evaluate(dataset=dataset, metrics=[faithfulness])
        if not hasattr(result, "to_pandas"):
            raise RuntimeError(f"ragas_evaluate returned unexpected type: {type(result)}")
        df = result.to_pandas()
        metrics["faithfulness"] = float(df["faithfulness"].iloc[0]) if not df.empty else 0.0
    else:
        metrics["faithfulness"] = 0.0

    # --- Attribution accuracy (F1 of cited vs relevant sources) ---
    # RAGAS has no built-in citation metric; reuse our implementation.
    if used_sources and relevant_sources:
        from pipeline.evaluation.grounding_metrics import attribution_accuracy
        metrics["attribution_accuracy"] = attribution_accuracy(used_sources, relevant_sources)

    return metrics
