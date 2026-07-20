"""Ensemble inference: load multiple models, fuse scores, produce predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sidq.fusion.optimize import uniform_fusion, weighted_fusion
from sidq.inference.multicrop import aggregate_scores, extract_crops


@dataclass
class EnsembleConfig:
    """Configuration for ensemble inference."""

    model_paths: list[Path]
    model_names: list[str]
    weights: dict[str, float] | None = None
    fusion_method: str = "uniform"
    normalize: str = "zscore"
    multi_crop: bool = False
    crop_duration_sec: float = 4.0
    crop_overlap: float = 0.5
    aggregation: str = "mean"
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class EnsemblePrediction:
    """Prediction from ensemble inference for one sample."""

    audio_id: str
    fused_score: float
    per_model_scores: dict[str, float]


def run_ensemble_inference(
    config: EnsembleConfig,
    audio_ids: list[str],
    waveforms: list[torch.Tensor],
    models: dict[str, torch.nn.Module] | None = None,
) -> pd.DataFrame:
    """Run ensemble inference over a set of waveforms.

    Args:
        config: Ensemble configuration.
        audio_ids: List of audio IDs.
        waveforms: List of waveform tensors.
        models: Pre-loaded models (name -> module). If None, loads from paths.

    Returns:
        DataFrame with columns: audio_id, logit (fused bonafide score).
    """
    if models is None:
        raise NotImplementedError(
            "Automatic model loading from paths requires checkpoint format "
            "specification. Pass pre-loaded models dict instead."
        )

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    all_model_scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        model = model.to(device)
        model.eval()
        scores = _score_all(model, waveforms, config, device)
        all_model_scores[name] = np.array(scores)

    # Fuse scores
    if config.fusion_method == "uniform":
        fused = uniform_fusion(all_model_scores)
    elif config.fusion_method == "weighted" and config.weights:
        fused = weighted_fusion(all_model_scores, config.weights)
    else:
        fused = uniform_fusion(all_model_scores)

    return pd.DataFrame({"audio_id": audio_ids, "logit": fused.tolist()})


@torch.no_grad()
def _score_all(
    model: torch.nn.Module,
    waveforms: list[torch.Tensor],
    config: EnsembleConfig,
    device: torch.device,
) -> list[float]:
    """Score all waveforms with one model."""
    scores: list[float] = []

    for wav in waveforms:
        if config.multi_crop:
            crops = extract_crops(
                wav,
                crop_duration_sec=config.crop_duration_sec,
                overlap=config.crop_overlap,
                include_full=True,
            )
            crop_scores = []
            for crop in crops:
                logits = model(crop.unsqueeze(0).to(device))
                crop_scores.append(logits[0, 1].item())
            score = aggregate_scores(crop_scores, config.aggregation)
        else:
            logits = model(wav.unsqueeze(0).to(device))
            score = logits[0, 1].item()
        scores.append(score)

    return scores
