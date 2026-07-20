"""Multi-crop inference for robust scoring."""

from __future__ import annotations

import numpy as np
import torch


def compute_crop_positions(
    total_samples: int,
    crop_samples: int,
    overlap: float = 0.5,
) -> list[int]:
    """Compute deterministic overlapping crop start positions.

    Args:
        total_samples: Total waveform length.
        crop_samples: Crop size in samples.
        overlap: Fraction of overlap between adjacent crops.

    Returns:
        List of start sample positions.
    """
    if total_samples <= crop_samples:
        return [0]

    stride = int(crop_samples * (1 - overlap))
    if stride <= 0:
        stride = 1

    positions = list(range(0, total_samples - crop_samples + 1, stride))
    if not positions:
        positions = [0]

    last_possible = total_samples - crop_samples
    if positions[-1] < last_possible:
        positions.append(last_possible)

    return positions


def extract_crops(
    waveform: torch.Tensor,
    crop_duration_sec: float,
    sample_rate: int = 16000,
    overlap: float = 0.5,
    include_full: bool = True,
) -> list[torch.Tensor]:
    """Extract overlapping crops from a waveform.

    Args:
        waveform: (num_samples,) tensor.
        crop_duration_sec: Duration of each crop.
        sample_rate: Audio sample rate.
        overlap: Overlap fraction.
        include_full: Whether to include the full utterance.

    Returns:
        List of crop tensors.
    """
    crop_samples = int(crop_duration_sec * sample_rate)
    total_samples = waveform.shape[-1]

    crops: list[torch.Tensor] = []

    if include_full:
        crops.append(waveform)

    if total_samples <= crop_samples:
        if not include_full:
            padded = torch.nn.functional.pad(waveform, (0, crop_samples - total_samples))
            crops.append(padded)
        return crops

    positions = compute_crop_positions(total_samples, crop_samples, overlap)
    for start in positions:
        crops.append(waveform[start : start + crop_samples])

    return crops


def aggregate_scores(
    scores: list[float],
    method: str = "mean",
) -> float:
    """Aggregate multiple crop scores into a single score.

    Args:
        scores: Per-crop bonafide scores.
        method: One of 'mean', 'median', 'logsumexp'.

    Returns:
        Aggregated bonafide score.
    """
    arr = np.array(scores)

    if method == "mean":
        return float(arr.mean())
    elif method == "median":
        return float(np.median(arr))
    elif method == "logsumexp":
        max_val = arr.max()
        return float(max_val + np.log(np.exp(arr - max_val).sum()) - np.log(len(arr)))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
