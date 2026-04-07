"""Production answer quality metrics using HuggingFace evaluate library.

HuggingFace evaluate provides standardized, peer-reviewed implementations of
SQuAD exact_match and F1 — the same metrics used in official leaderboards.

Drop-in alternative to answer_metrics.py with identical interface.

Install: pip install evaluate
Docs: https://huggingface.co/docs/evaluate
"""

from typing import Dict, List, Optional
from pipeline.evaluation.answer_metrics import anls_score

try:
    import evaluate as hf_evaluate
    _squad_metric = hf_evaluate.load("squad")
    HF_EVALUATE_AVAILABLE = True
except Exception:
    HF_EVALUATE_AVAILABLE = False
    _squad_metric = None


def compute_answer_metrics_hf(
    prediction: str,
    ground_truth: str,
    all_ground_truths: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute answer quality metrics using HuggingFace evaluate (SQuAD protocol).

    Uses the official SQuAD metric which applies the same normalization
    (lowercase, strip articles/punctuation) as the original paper.

    Args:
        prediction: Model's predicted answer.
        ground_truth: Primary ground truth answer.
        all_ground_truths: Optional list of all acceptable answers.
            SQuAD takes the best match across all answers automatically.

    Returns:
        Dictionary with "exact_match" and "f1" keys, same as compute_answer_metrics().
    """
    if not HF_EVALUATE_AVAILABLE or _squad_metric is None:
        raise ImportError(
            "HuggingFace evaluate is not installed or failed to load. "
            "Run: pip install evaluate"
        )

    answers = all_ground_truths if all_ground_truths else [ground_truth]

    # SQuAD metric expects a list of predictions and a list of references.
    # Each reference is {"id": str, "answers": {"text": [...], "answer_start": [...]}}.
    # We use dummy answer_start=0 since we only care about text matching.
    predictions = [{"id": "0", "prediction_text": prediction}]
    references = [
        {
            "id": "0",
            "answers": {
                "text": answers,
                "answer_start": [0] * len(answers),
            },
        }
    ]

    scores = _squad_metric.compute(predictions=predictions, references=references)

    anls = max(anls_score(prediction, gt) for gt in answers)

    return {
        # SQuAD returns 0-100 scale; normalize to 0-1 to match our convention
        "exact_match": (scores["exact_match"] / 100.0) if scores else 0.0,
        "f1": (scores["f1"] / 100.0) if scores else 0.0,
        "anls": anls,
    }
