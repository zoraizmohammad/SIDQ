"""Equal Error Rate computation — the primary competition metric.

Convention (from official ArA-DF 2026 specification):
- Label 1 = bonafide (positive class)
- Label 0 = spoofed
- Higher score = more likely bonafide
- Lower EER is better
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sidq.constants import LABEL_BONAFIDE, LABEL_SPOOFED


@dataclass
class EERResult:
    """Result of an EER computation."""

    eer: float
    threshold: float
    num_bonafide: int
    num_spoofed: int


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> EERResult:
    """Compute Equal Error Rate.

    Args:
        scores: Continuous bonafide scores. Higher = more likely bonafide.
        labels: Ground truth labels. 1 = bonafide, 0 = spoofed.

    Returns:
        EERResult with EER value and the operating threshold.

    Raises:
        ValueError: on invalid inputs (NaN, Inf, wrong labels, single class).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)

    _validate_inputs(scores, labels)

    bonafide_scores = scores[labels == LABEL_BONAFIDE]
    spoofed_scores = scores[labels == LABEL_SPOOFED]

    # FRR: fraction of bonafide rejected (score < threshold)
    # FAR: fraction of spoofed accepted (score >= threshold)
    # Sweep candidate thresholds over all unique scores.
    all_scores = np.concatenate([bonafide_scores, spoofed_scores])
    thresholds = np.unique(all_scores)

    # Add sentinel thresholds below min and above max
    eps = 1e-8
    thresholds = np.concatenate(
        [[thresholds[0] - eps], thresholds, [thresholds[-1] + eps]]
    )

    frr = np.array([(bonafide_scores < t).mean() for t in thresholds])
    far = np.array([(spoofed_scores >= t).mean() for t in thresholds])

    # Find point where FRR and FAR cross
    diff = frr - far
    # Index of smallest absolute difference
    idx = int(np.argmin(np.abs(diff)))

    # Interpolate around the crossing point when possible for a smoother EER
    if 0 < idx < len(thresholds) - 1 and diff[idx] != 0:
        # Look for adjacent sign change
        for j in (idx - 1, idx):
            if j >= 0 and j + 1 < len(diff) and np.sign(diff[j]) != np.sign(diff[j + 1]):
                # Linear interpolation between j and j+1
                d0, d1 = diff[j], diff[j + 1]
                if d1 != d0:
                    frac = -d0 / (d1 - d0)
                    eer = float(frr[j] + frac * (frr[j + 1] - frr[j]))
                    thr = float(
                        thresholds[j] + frac * (thresholds[j + 1] - thresholds[j])
                    )
                    return EERResult(
                        eer=eer,
                        threshold=thr,
                        num_bonafide=len(bonafide_scores),
                        num_spoofed=len(spoofed_scores),
                    )

    eer = float((frr[idx] + far[idx]) / 2)
    return EERResult(
        eer=eer,
        threshold=float(thresholds[idx]),
        num_bonafide=len(bonafide_scores),
        num_spoofed=len(spoofed_scores),
    )


def _validate_inputs(scores: np.ndarray, labels: np.ndarray) -> None:
    """Validate score and label arrays."""
    if scores.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: scores {scores.shape} vs labels {labels.shape}"
        )
    if scores.size == 0:
        raise ValueError("Empty score array")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or Inf")

    unique_labels = set(np.unique(labels).tolist())
    allowed = {LABEL_BONAFIDE, LABEL_SPOOFED}
    if not unique_labels.issubset(allowed):
        raise ValueError(f"Labels must be subset of {allowed}, got {unique_labels}")
    if len(unique_labels) < 2:
        raise ValueError(
            f"EER requires both classes present, got only {unique_labels}"
        )


def detect_score_inversion(scores: np.ndarray, labels: np.ndarray) -> bool:
    """Detect if scores appear to be inverted (higher = spoofed).

    Returns True if the score direction looks wrong: i.e., mean bonafide
    score is lower than mean spoofed score AND inverting improves EER
    substantially.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    _validate_inputs(scores, labels)

    result = compute_eer(scores, labels)
    inverted = compute_eer(-scores, labels)

    # If inverting cuts EER by more than 20 absolute points, direction is wrong
    return inverted.eer < result.eer - 0.20
