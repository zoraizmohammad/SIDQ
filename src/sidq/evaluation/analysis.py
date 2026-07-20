"""Failure analysis and score-drift reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sidq.evaluation.eer import compute_eer


@dataclass
class FailureCase:
    """A single failure case for analysis."""

    audio_id: str
    score: float
    label: int
    margin: float
    condition: str = "clean"


def find_low_margin_cases(
    scores: np.ndarray,
    labels: np.ndarray,
    audio_ids: list[str],
    threshold: float | None = None,
    top_k: int = 50,
) -> list[FailureCase]:
    """Find cases closest to the decision boundary."""
    if threshold is None:
        result = compute_eer(scores, labels)
        threshold = result.threshold

    margins = np.abs(scores - threshold)
    sorted_idx = np.argsort(margins)[:top_k]

    cases = []
    for idx in sorted_idx:
        cases.append(FailureCase(
            audio_id=audio_ids[idx],
            score=float(scores[idx]),
            label=int(labels[idx]),
            margin=float(margins[idx]),
        ))
    return cases


def compute_score_drift(
    clean_scores: np.ndarray,
    corrupted_scores: np.ndarray,
) -> dict[str, float]:
    """Compute score drift between clean and corrupted predictions."""
    diff = corrupted_scores - clean_scores
    return {
        "mean_drift": float(diff.mean()),
        "std_drift": float(diff.std()),
        "max_abs_drift": float(np.abs(diff).max()),
        "rank_correlation": float(np.corrcoef(clean_scores, corrupted_scores)[0, 1]),
    }
