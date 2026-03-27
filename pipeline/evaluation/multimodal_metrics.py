"""Multimodal evaluation metrics: VQA accuracy, Cross-modal consistency."""

from typing import Dict, List, Any
from pipeline.evaluation.answer_metrics import _normalize_answer


def vqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """VQA accuracy metric following the standard VQA evaluation protocol.

    VQA accuracy = min(#humans_who_gave_that_answer / 3, 1)
    Simplified version: checks if prediction matches any ground truth.

    For datasets with multiple answers, gives partial credit based
    on how many annotators agree with the prediction.

    Args:
        prediction: Model's predicted answer.
        ground_truths: List of ground truth answers from annotators.

    Returns:
        VQA accuracy score between 0 and 1.
    """
    if not ground_truths:
        return 0.0

    normalized_pred = _normalize_answer(prediction)
    normalized_gts = [_normalize_answer(gt) for gt in ground_truths]

    # Count how many ground truths match the prediction
    match_count = sum(1 for gt in normalized_gts if gt == normalized_pred)

    # Standard VQA accuracy: min(match_count / 3, 1)
    return min(match_count / 3.0, 1.0)


def cross_modal_consistency(
    text_answer: str,
    image_answer: str,
) -> float:
    """Measure consistency between answers derived from text vs image context.

    Tests if the model gives the same answer regardless of whether
    it uses text or image evidence. High consistency = reliable system.

    Args:
        text_answer: Answer generated from text context only.
        image_answer: Answer generated from image context only.

    Returns:
        Consistency score between 0 and 1.
    """
    if not text_answer or not image_answer:
        return 0.0

    text_tokens = set(_normalize_answer(text_answer).split())
    image_tokens = set(_normalize_answer(image_answer).split())

    if not text_tokens and not image_tokens:
        return 1.0
    if not text_tokens or not image_tokens:
        return 0.0

    # Jaccard similarity between token sets
    intersection = text_tokens & image_tokens
    union = text_tokens | image_tokens
    return len(intersection) / len(union)


def compute_multimodal_metrics(
    prediction: str,
    ground_truths: List[str] = None,
    text_only_answer: str = None,
    image_only_answer: str = None,
) -> Dict[str, float]:
    """Compute all multimodal metrics.

    Args:
        prediction: Model's predicted answer.
        ground_truths: List of ground truth answers.
        text_only_answer: Answer from text-only context (for consistency check).
        image_only_answer: Answer from image-only context (for consistency check).

    Returns:
        Dictionary of metric_name -> score.
    """
    metrics = {}

    if ground_truths:
        metrics["vqa_accuracy"] = vqa_accuracy(prediction, ground_truths)

    if text_only_answer and image_only_answer:
        metrics["cross_modal_consistency"] = cross_modal_consistency(
            text_only_answer, image_only_answer
        )

    return metrics
