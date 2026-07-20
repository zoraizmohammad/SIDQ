"""Additional evaluation metrics and score-direction safeguards."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from sidq.constants import LABEL_BONAFIDE, LABEL_SPOOFED
from sidq.evaluation.eer import compute_eer


def compute_all_metrics(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Compute all evaluation metrics for a set of predictions.

    Convention: higher score = more likely bonafide (label=1).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)

    eer_result = compute_eer(scores, labels)

    # ROC-AUC: sklearn expects positive class = 1, higher score → more positive
    auc = roc_auc_score(labels, scores)

    # Accuracy at EER threshold
    predictions_at_thr = (scores >= eer_result.threshold).astype(int)
    accuracy = float((predictions_at_thr == labels).mean())

    # Macro F1 at EER threshold
    f1 = _macro_f1(predictions_at_thr, labels)

    return {
        "eer": eer_result.eer,
        "eer_threshold": eer_result.threshold,
        "roc_auc": float(auc),
        "accuracy_at_eer": accuracy,
        "macro_f1_at_eer": f1,
        "num_bonafide": eer_result.num_bonafide,
        "num_spoofed": eer_result.num_spoofed,
    }


def _macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute macro-averaged F1 score."""
    f1_scores = []
    for cls in [LABEL_BONAFIDE, LABEL_SPOOFED]:
        tp = int(((predictions == cls) & (labels == cls)).sum())
        fp = int(((predictions == cls) & (labels != cls)).sum())
        fn = int(((predictions != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def validate_score_direction(
    scores: np.ndarray,
    labels: np.ndarray,
    fail_loudly: bool = True,
) -> bool:
    """Validate that scores follow the correct direction (higher=bonafide).

    This is a critical safety check to prevent submitting inverted scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)

    bonafide_mask = labels == LABEL_BONAFIDE
    spoofed_mask = labels == LABEL_SPOOFED

    mean_bonafide = scores[bonafide_mask].mean()
    mean_spoofed = scores[spoofed_mask].mean()

    correct_direction = mean_bonafide > mean_spoofed

    if not correct_direction and fail_loudly:
        raise ValueError(
            f"SCORE DIRECTION ERROR: Mean bonafide score ({mean_bonafide:.4f}) "
            f"is lower than mean spoofed score ({mean_spoofed:.4f}). "
            f"Expected higher scores for bonafide (label=1). "
            f"Check that you are outputting bonafide-class logits."
        )

    return correct_direction


def bootstrap_eer(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for EER."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    eers = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_scores = scores[indices]
        boot_labels = labels[indices]
        if len(set(boot_labels)) < 2:
            continue
        try:
            result = compute_eer(boot_scores, boot_labels)
            eers.append(result.eer)
        except ValueError:
            continue

    if not eers:
        return {"eer_mean": float("nan"), "eer_ci_lower": float("nan"), "eer_ci_upper": float("nan")}

    eers_arr = np.array(eers)
    alpha = 1 - confidence
    return {
        "eer_mean": float(eers_arr.mean()),
        "eer_ci_lower": float(np.percentile(eers_arr, 100 * alpha / 2)),
        "eer_ci_upper": float(np.percentile(eers_arr, 100 * (1 - alpha / 2))),
    }
