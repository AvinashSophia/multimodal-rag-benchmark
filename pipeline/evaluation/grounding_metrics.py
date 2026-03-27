"""Grounding evaluation metrics: Faithfulness, Attribution accuracy."""

from typing import Dict, List, Any


def faithfulness_score(
    answer: str,
    retrieved_context: List[str],
) -> float:
    """Simple heuristic faithfulness check.

    Measures what fraction of answer tokens appear in the retrieved context.
    A faithful answer should be grounded in the retrieved documents.

    For production use, consider RAGAS faithfulness or LLM-as-judge.

    Args:
        answer: Model's generated answer.
        retrieved_context: List of retrieved text chunks.

    Returns:
        Faithfulness score between 0 and 1.
    """
    if not answer or not retrieved_context:
        return 0.0

    answer_tokens = set(answer.lower().split())
    context_text = " ".join(retrieved_context).lower()
    context_tokens = set(context_text.split())

    # Remove common stop words from consideration
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "its", "this", "that", "these", "those", "and", "or",
        "but", "not", "no", "yes",
    }

    meaningful_answer_tokens = answer_tokens - stop_words
    if not meaningful_answer_tokens:
        return 1.0  # Answer is all stop words, consider faithful

    grounded_tokens = meaningful_answer_tokens & context_tokens
    return len(grounded_tokens) / len(meaningful_answer_tokens)


def attribution_accuracy(
    used_sources: List[str],
    relevant_sources: List[str],
) -> float:
    """Measure if the model attributed its answer to the correct sources.

    Args:
        used_sources: Sources the model claims to have used.
        relevant_sources: Ground-truth relevant sources.

    Returns:
        Attribution accuracy between 0 and 1.
    """
    if not relevant_sources:
        return 0.0
    if not used_sources:
        return 0.0

    used_set = set(used_sources)
    relevant_set = set(relevant_sources)

    correct = len(used_set & relevant_set)
    # Precision: of sources cited, how many are actually relevant
    precision = correct / len(used_set) if used_set else 0.0
    # Recall: of relevant sources, how many were cited
    recall = correct / len(relevant_set) if relevant_set else 0.0

    if precision + recall == 0:
        return 0.0
    # F1 of attribution
    return 2 * precision * recall / (precision + recall)


def compute_grounding_metrics(
    answer: str,
    retrieved_context: List[str],
    used_sources: List[str] = None,
    relevant_sources: List[str] = None,
) -> Dict[str, float]:
    """Compute all grounding metrics.

    Args:
        answer: Model's generated answer.
        retrieved_context: Retrieved text chunks.
        used_sources: Sources cited by model.
        relevant_sources: Ground-truth relevant sources.

    Returns:
        Dictionary of metric_name -> score.
    """
    metrics = {
        "faithfulness": faithfulness_score(answer, retrieved_context),
    }

    if used_sources and relevant_sources:
        metrics["attribution_accuracy"] = attribution_accuracy(
            used_sources, relevant_sources
        )

    return metrics
