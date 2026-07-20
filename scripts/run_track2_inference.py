#!/usr/bin/env python3
"""Run Track 2 inference using the baseline model with enhancements.

Produces predictions for Track 2 evaluation phase submission.
Enhancements over raw baseline:
- Multi-crop scoring (multiple overlapping windows)
- Score aggregation for more robust predictions

Usage:
    python scripts/run_track2_inference.py \
        --audio-dir ArA-DF-2026/data/track-2_test \
        --config ArA-DF-2026/models/ArA-DF-Baseline/config.yaml \
        --checkpoint ArA-DF-2026/models/ArA-DF-Baseline/best_model.pth \
        --output track2_preds.csv \
        --multi-crop
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LEN = 64600  # ~4s at 16kHz
BONAFIDE_CLASS_IDX = 1


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file to numpy array."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy().astype(np.float32)


def pad_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    """Repeat-pad or trim waveform."""
    if len(x) >= max_len:
        return x[:max_len]
    reps = (max_len // len(x)) + 1
    return np.tile(x, reps)[:max_len]


def get_crops(x: np.ndarray, max_len: int, overlap: float = 0.5) -> list[np.ndarray]:
    """Get multiple overlapping crops from audio."""
    if len(x) <= max_len:
        return [pad_or_trim(x, max_len)]

    stride = int(max_len * (1 - overlap))
    crops = []
    for start in range(0, len(x) - max_len + 1, stride):
        crops.append(x[start:start + max_len])

    # Always include the last segment
    last_start = len(x) - max_len
    if not crops or crops[-1] is not x[last_start:last_start + max_len]:
        crops.append(x[last_start:last_start + max_len])

    return crops


def model_scores(outputs: dict) -> np.ndarray:
    """Extract bonafide scores from model outputs."""
    scores = outputs.get("scores")
    if scores is not None and torch.is_tensor(scores):
        scores = scores.detach().cpu()
        if scores.ndim == 1:
            return scores.numpy()
        if scores.ndim == 2 and scores.shape[1] >= 2:
            return scores[:, BONAFIDE_CLASS_IDX].numpy()

    logits = outputs.get("logits")
    if logits is not None and torch.is_tensor(logits):
        if logits.ndim == 2 and logits.shape[1] >= 2:
            return logits[:, BONAFIDE_CLASS_IDX].detach().cpu().numpy()

    raise ValueError("Model outputs contain neither usable scores nor logits.")


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load DeepFense model."""
    from omegaconf import OmegaConf
    from deepfense.models import *  # noqa: F401,F403
    from deepfense.utils.registry import build_detector

    cfg = OmegaConf.load(config_path)

    # Resolve ckpt_path
    config_dir = Path(config_path).resolve().parent
    ckpt = str(cfg.model.frontend.args.ckpt_path)
    path = Path(ckpt).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    cfg.model.frontend.args.ckpt_path = str(path)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_detector(cfg.model.type, model_cfg)
    model.to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    audio_dir: Path,
    config_path: str,
    checkpoint_path: str,
    output_csv: Path,
    batch_size: int = 32,
    multi_crop: bool = False,
    overlap: float = 0.5,
):
    """Run inference with optional multi-crop scoring."""
    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available()
                         else "cpu")
    logger.info(f"Device: {device}")

    model = load_model(config_path, checkpoint_path, device)

    audio_files = sorted(audio_dir.rglob("*.flac"))
    if not audio_files:
        raise FileNotFoundError(f"No .flac files found under {audio_dir}")
    logger.info(f"Found {len(audio_files)} audio files")
    logger.info(f"Multi-crop: {multi_crop}, overlap: {overlap}")

    rows: list[tuple[str, float]] = []

    for i in tqdm(range(0, len(audio_files), batch_size), desc="Inference"):
        batch_files = audio_files[i:i + batch_size]

        for fpath in batch_files:
            try:
                x = load_audio(str(fpath))

                if multi_crop and len(x) > MAX_LEN:
                    crops = get_crops(x, MAX_LEN, overlap)
                    crop_scores = []
                    for crop in crops:
                        tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).to(device)
                        outputs = model(tensor)
                        score = model_scores(outputs)
                        crop_scores.append(float(score[0]))
                    final_score = float(np.mean(crop_scores))
                else:
                    x_padded = pad_or_trim(x, MAX_LEN)
                    tensor = torch.tensor(x_padded, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = model(tensor)
                    score = model_scores(outputs)
                    final_score = float(score[0])

                rows.append((fpath.stem, final_score))
            except Exception as e:
                logger.warning(f"Skipping {fpath.name}: {e}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_id", "logit"])
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Track 2 inference")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("track2_preds.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--multi-crop", action="store_true")
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    run_inference(
        audio_dir=args.audio_dir,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        batch_size=args.batch_size,
        multi_crop=args.multi_crop,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
