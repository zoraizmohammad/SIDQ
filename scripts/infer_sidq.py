#!/usr/bin/env python3
"""Native SIDQ inference: score audio with a ``train_sidq.py`` checkpoint.

Unlike ``run_track2_inference.py`` (which needs DeepFense + fairseq) and
``standalone_inference.py`` (which emits a *proxy* score, not usable for
submission), this script loads the exact HF-XLS-R + AASIST model that
``train_sidq.py`` produces and writes real bonafide-class logits.

Preprocessing mirrors training (``scripts/train_sidq.py``): mono 16 kHz,
repeat-pad/trim to 64600 samples (~4 s). ``--multi-crop`` averages the bonafide
score over overlapping windows for longer clips (more robust, a bit slower).

Usage:
    python scripts/infer_sidq.py \
        --audio-dir ArA-DF-2026/data/track-2_test \
        --checkpoint checkpoints/best_sidq.pt \
        --output track2_preds_sidq.csv \
        --batch-size 32

    # More robust (recommended for the acoustic-robustness track):
    python scripts/infer_sidq.py ... --multi-crop
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sidq.constants import LABEL_BONAFIDE, SAMPLE_RATE
from sidq.models.sidq_xlsr import load_from_checkpoint

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LEN = 64600  # ~4.04 s at 16 kHz — matches train_sidq.py


def load_audio_np(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio to a mono float32 numpy array at target sample rate."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy().astype(np.float32)


def pad_or_trim(x: np.ndarray, max_len: int) -> np.ndarray:
    """Repeat-pad or trim to exactly max_len (matches training)."""
    if len(x) >= max_len:
        return x[:max_len]
    reps = (max_len // len(x)) + 1
    return np.tile(x, reps)[:max_len]


def get_crops(x: np.ndarray, max_len: int, overlap: float = 0.5) -> list[np.ndarray]:
    """Overlapping fixed-length crops; always includes the final window."""
    if len(x) <= max_len:
        return [pad_or_trim(x, max_len)]
    stride = max(1, int(max_len * (1 - overlap)))
    crops = [x[s:s + max_len] for s in range(0, len(x) - max_len + 1, stride)]
    last = x[len(x) - max_len:]
    if not crops or not np.array_equal(crops[-1], last):
        crops.append(last)
    return crops


@torch.no_grad()
def run(
    audio_dir: Path,
    checkpoint_path: str,
    output_csv: Path,
    batch_size: int = 32,
    multi_crop: bool = False,
    overlap: float = 0.5,
    device_name: str = "auto",
    limit: int | None = None,
) -> None:
    if device_name == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_name)
    logger.info("Device: %s", device)

    model, ckpt = load_from_checkpoint(checkpoint_path, device)
    if "best_eer" in ckpt:
        logger.info("Loaded checkpoint (val EER %.2f%%)", ckpt["best_eer"] * 100)

    audio_files = sorted(audio_dir.rglob("*.flac"))
    if not audio_files:
        raise FileNotFoundError(f"No .flac files under {audio_dir}")
    if limit:
        audio_files = audio_files[:limit]
    logger.info("Found %d audio files | multi_crop=%s", len(audio_files), multi_crop)

    rows: list[tuple[str, float]] = []

    if multi_crop:
        # Per-file variable crop counts — score each file's crops as one batch.
        for fpath in tqdm(audio_files, desc="Inference (multi-crop)"):
            try:
                x = load_audio_np(str(fpath))
                crops = get_crops(x, MAX_LEN, overlap)
                batch = torch.tensor(np.stack(crops), dtype=torch.float32).to(device)
                logits = model(batch)
                score = float(logits[:, LABEL_BONAFIDE].mean().cpu())
                rows.append((fpath.stem, score))
            except Exception as e:  # noqa: BLE001
                logger.warning("Skipping %s: %s", fpath.name, e)
    else:
        for i in tqdm(range(0, len(audio_files), batch_size), desc="Inference"):
            batch_files = audio_files[i:i + batch_size]
            arrays, ids = [], []
            for fpath in batch_files:
                try:
                    x = pad_or_trim(load_audio_np(str(fpath)), MAX_LEN)
                    arrays.append(x)
                    ids.append(fpath.stem)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Skipping %s: %s", fpath.name, e)
            if not arrays:
                continue
            batch = torch.tensor(np.stack(arrays), dtype=torch.float32).to(device)
            logits = model(batch)
            scores = logits[:, LABEL_BONAFIDE].cpu().numpy()
            rows.extend(zip(ids, (float(s) for s in scores), strict=True))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_id", "logit"])
        writer.writerows(rows)
    logger.info("Saved %d predictions to %s", len(rows), output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Native SIDQ Track-2 inference")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("track2_preds_sidq.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--multi-crop", action="store_true")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Score only first N files (debug)")
    args = parser.parse_args()

    run(
        audio_dir=args.audio_dir,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        batch_size=args.batch_size,
        multi_crop=args.multi_crop,
        overlap=args.overlap,
        device_name=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
