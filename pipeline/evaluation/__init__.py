"""Evaluation module - combines all metric categories."""

from typing import Dict, List, Any
from pipeline.evaluation.retrieval_metrics import compute_retrieval_metrics
from pipeline.evaluation.answer_metrics import compute_answer_metrics
from pipeline.evaluation.grounding_metrics import compute_grounding_metrics
from pipeline.evaluation.multimodal_metrics import compute_multimodal_metrics


class Evaluator:
    """Unified evaluator that computes all metrics for a benchmark run.

    Combines retrieval, answer quality, grounding, and multimodal metrics
    into a single evaluation call.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k_values = config["evaluation"].get("recall_k_values", [1, 3, 5, 10])

    def evaluate_sample(
        self,
        prediction: str,
        ground_truth: str,
        retrieved_texts: List[str] = None,
        retrieved_text_ids: List[str] = None,
        relevant_texts: List[str] = None,
        relevant_text_ids: List[str] = None,
        retrieved_images: List[str] = None,
        retrieved_image_ids: List[str] = None,
        relevant_images: List[str] = None,
        relevant_image_ids: List[str] = None,
        used_sources: List[str] = None,
        relevant_sources: List[str] = None,
        all_ground_truths: List[str] = None,
        text_only_answer: str = None,
        image_only_answer: str = None,
        has_images: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a single sample across all metric categories.

        Returns:
            Dictionary of all metrics for this sample.
        """
        metrics = {}

        # Retrieval metrics - text
        if retrieved_texts and relevant_texts:
            text_metrics = compute_retrieval_metrics(
                retrieved_texts, relevant_texts, self.k_values
            )
            metrics.update({f"text_{k}": v for k, v in text_metrics.items()})

        # Retrieval metrics - image
        if retrieved_image_ids and relevant_image_ids:
            image_metrics = compute_retrieval_metrics(
                retrieved_image_ids, relevant_image_ids, self.k_values
            )
            metrics.update({f"image_{k}": v for k, v in image_metrics.items()})


        # Answer quality metrics
        metrics.update(
            compute_answer_metrics(prediction, ground_truth, all_ground_truths)
        )

        # Grounding metrics
        if retrieved_texts:
            metrics.update(
                compute_grounding_metrics(
                    prediction, retrieved_texts, used_sources, relevant_sources
                )
            )

        # Multimodal metrics
        mm_metrics = compute_multimodal_metrics(
            prediction, all_ground_truths, text_only_answer, image_only_answer
        )
        metrics.update(mm_metrics)

        return metrics

    def aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all samples (compute averages).

        Args:
            all_metrics: List of per-sample metric dictionaries.

        Returns:
            Averaged metrics across all samples.
        """
        if not all_metrics:
            return {}

        aggregated = {}
        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated
