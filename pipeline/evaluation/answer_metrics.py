"""Answer quality evaluation metrics: Exact Match, F1, ANLS."""

import re
import string
from typing import Dict, List, Optional
from collections import Counter


def _normalize_answer(text: str) -> str:
    """Normalize answer text for fair comparison.

    Lowercases, removes articles, punctuation, and extra whitespace.
    Standard normalization used in SQuAD and HotpotQA evaluation.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM) - binary, 1 if normalized texts match exactly.

    Args:
        prediction: Model's predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth.

    Measures overlap of words between predicted and true answer.
    More forgiving than EM - gives partial credit.

    Args:
        prediction: Model's predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        F1 score between 0 and 1.
    """
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ANLS).

    Measures character-level similarity between prediction and ground truth.
    More forgiving than EM for OCR outputs, unit variations, and partial matches.
    Scores below the threshold are set to 0 to avoid rewarding near-misses on
    completely wrong answers.

    Standard metric for document VQA tasks (DocVQA, InfographicVQA).

    Args:
        prediction:   Model's predicted answer.
        ground_truth: Ground truth answer.
        threshold:    Minimum similarity to score above 0 (default 0.5).

    Returns:
        ANLS score between 0 and 1.
    """
    pred = _normalize_answer(prediction)
    gt   = _normalize_answer(ground_truth)

    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    # Character-level edit distance via dynamic programming
    len_p, len_g = len(pred), len(gt)
    dp = list(range(len_g + 1))
    for i in range(1, len_p + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len_g + 1):
            temp = dp[j]
            if pred[i - 1] == gt[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    edit_dist = dp[len_g]
    nls = 1.0 - edit_dist / max(len_p, len_g)
    return nls if nls >= threshold else 0.0


def compute_answer_metrics(
    prediction: str,
    ground_truth: str,
    all_ground_truths: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute all answer quality metrics.

    If multiple ground truth answers exist (e.g., DocVQA),
    takes the max score across all ground truths.

    Args:
        prediction: Model's predicted answer.
        ground_truth: Primary ground truth answer.
        all_ground_truths: Optional list of all acceptable answers.

    Returns:
        Dictionary of metric_name -> score.
    """
    if all_ground_truths and len(all_ground_truths) > 1:
        em   = max(exact_match(prediction, gt) for gt in all_ground_truths)
        f1   = max(f1_score(prediction, gt) for gt in all_ground_truths)
        anls = max(anls_score(prediction, gt) for gt in all_ground_truths)
    else:
        em   = exact_match(prediction, ground_truth)
        f1   = f1_score(prediction, ground_truth)
        anls = anls_score(prediction, ground_truth)

    return {
        "exact_match": em,
        "f1":          f1,
        "anls":        anls,
    }
