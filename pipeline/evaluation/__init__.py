"""Evaluation module - combines all metric categories.

Backend selection via config:
    evaluation:
      backend: "custom"       # default — fast, no external deps
      backend: "production"   # ranx + HuggingFace evaluate + RAGAS (LLM-as-judge)
"""

from typing import Callable, Dict, List, Any, Optional, Set
from pipeline.evaluation.retrieval_metrics import compute_retrieval_metrics
from pipeline.evaluation.answer_metrics import compute_answer_metrics
from pipeline.evaluation.grounding_metrics import compute_grounding_metrics
from pipeline.evaluation.multimodal_metrics import compute_multimodal_metrics


class Evaluator:
    """Unified evaluator that computes all metrics for a benchmark run.

    Combines retrieval, answer quality, grounding, and multimodal metrics
    into a single evaluation call.

    backend="custom" (default): fast, numpy/regex implementations, no API calls.
    backend="production": ranx (IR metrics), HuggingFace evaluate (SQuAD EM/F1),
                          RAGAS (LLM-as-judge faithfulness). Requires extra deps.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k_values = config["evaluation"].get("recall_k_values", [1, 3, 5, 10])
        self.backend = config["evaluation"].get("backend", "custom")

        self._compute_retrieval: Any
        self._compute_answer: Any
        self._compute_grounding: Any

        if self.backend == "production":
            from pipeline.evaluation.retrieval_metrics_ranx import compute_retrieval_metrics_ranx
            from pipeline.evaluation.answer_metrics_hf import compute_answer_metrics_hf
            from pipeline.evaluation.grounding_metrics_ragas import compute_grounding_metrics_ragas
            self._compute_retrieval = compute_retrieval_metrics_ranx
            self._compute_answer = compute_answer_metrics_hf
            self._compute_grounding = compute_grounding_metrics_ragas
            print("  [Evaluator] Using production backend: ranx + HF evaluate + RAGAS")
        else:
            self._compute_retrieval = compute_retrieval_metrics
            self._compute_answer = compute_answer_metrics
            self._compute_grounding = compute_grounding_metrics

    def evaluate_sample(
        self,
        prediction: str,
        ground_truth: str,
        retrieved_texts: Optional[List[str]] = None,
        retrieved_text_ids: Optional[List[str]] = None,
        relevant_texts: Optional[List[str]] = None,
        relevant_text_ids: Optional[List[str]] = None,
        retrieved_images: Optional[List[str]] = None,
        retrieved_image_ids: Optional[List[str]] = None,
        relevant_images: Optional[List[str]] = None,
        relevant_image_ids: Optional[List[str]] = None,
        used_sources: Optional[List[str]] = None,
        relevant_sources: Optional[List[str]] = None,
        all_ground_truths: Optional[List[str]] = None,
        text_only_answer: Optional[str] = None,
        image_only_answer: Optional[str] = None,
        has_images: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a single sample across all metric categories.

        Returns:
            Dictionary of all metrics for this sample.
        """
        metrics = {}

        # Retrieval metrics - text (prefer IDs over full text for comparison)
        if retrieved_text_ids and relevant_text_ids:
            text_metrics = self._compute_retrieval(
                retrieved_text_ids, relevant_text_ids, self.k_values
            )
            metrics.update({f"text_{k}": v for k, v in text_metrics.items()})
        elif retrieved_texts and relevant_texts:
            text_metrics = self._compute_retrieval(
                retrieved_texts, relevant_texts, self.k_values
            )
            metrics.update({f"text_{k}": v for k, v in text_metrics.items()})

        # Retrieval metrics - image
        if retrieved_image_ids and relevant_image_ids:
            image_metrics = self._compute_retrieval(
                retrieved_image_ids, relevant_image_ids, self.k_values
            )
            metrics.update({f"image_{k}": v for k, v in image_metrics.items()})

        # Answer quality metrics
        metrics.update(
            self._compute_answer(prediction, ground_truth, all_ground_truths)
        )

        # Grounding metrics
        # Faithfulness requires retrieved text — text datasets only.
        # Attribution accuracy only needs source ID lists — works for image datasets too.
        if retrieved_texts:
            grounding = self._compute_grounding(prediction, retrieved_texts)
            metrics["faithfulness"] = grounding["faithfulness"]
        if used_sources and relevant_sources:
            from pipeline.evaluation.grounding_metrics import attribution_accuracy
            metrics["attribution_accuracy"] = attribution_accuracy(used_sources, relevant_sources)

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
        all_keys: Set[str] = set()
        for m in all_metrics:
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated
