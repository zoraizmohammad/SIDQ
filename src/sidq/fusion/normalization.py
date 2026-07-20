"""Score normalization for model fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizationStats:
    """Statistics for score normalization."""

    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    iqr: float = 1.0
    method: str = "zscore"


def fit_normalization(scores: np.ndarray, method: str = "zscore") -> NormalizationStats:
    """Fit normalization statistics from validation scores."""
    stats = NormalizationStats(method=method)
    stats.mean = float(scores.mean())
    stats.std = float(scores.std())
    stats.median = float(np.median(scores))
    q75, q25 = np.percentile(scores, [75, 25])
    stats.iqr = float(q75 - q25) if q75 != q25 else 1.0
    return stats


def normalize_scores(scores: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Normalize scores using fitted statistics."""
    if stats.method == "none":
        return scores
    elif stats.method == "zscore":
        std = stats.std if stats.std > 1e-8 else 1.0
        return (scores - stats.mean) / std
    elif stats.method == "robust":
        iqr = stats.iqr if stats.iqr > 1e-8 else 1.0
        return (scores - stats.median) / iqr
    elif stats.method == "rank":
        ranks = np.argsort(np.argsort(scores)).astype(float)
        return ranks / (len(scores) - 1) if len(scores) > 1 else ranks
    else:
        raise ValueError(f"Unknown normalization method: {stats.method}")
