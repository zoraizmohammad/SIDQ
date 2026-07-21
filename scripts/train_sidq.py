#!/usr/bin/env python3
"""Train SIDQ model using transformers XLS-R + AASIST on ArA-DF 2026.

This is the native SIDQ training script that doesn't require DeepFense.
Uses HuggingFace transformers for XLS-R and our own AASIST implementation.

Usage:
    # Smoke test (CPU, small subset)
    python scripts/train_sidq.py --smoke

    # Full training on MPS
    python scripts/train_sidq.py --data-root ArA-DF-2026/data --epochs 30

    # With augmentation
    python scripts/train_sidq.py --data-root ArA-DF-2026/data --augmentation robust
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sidq.audio.io import load_audio
from sidq.augment.curriculum import build_pipeline
from sidq.constants import LABEL_BONAFIDE, SAMPLE_RATE
from sidq.evaluation.eer import compute_eer
from sidq.models.sidq_xlsr import SIDQXLSRModel
from sidq.reproducibility import seed_everything


MAX_LEN = 64600  # 4.0375s at 16kHz


class AraDFDataset(Dataset):
    """ArA-DF 2026 dataset for training."""

    def __init__(
        self,
        data_dir: Path,
        metadata: pd.DataFrame,
        max_len: int = MAX_LEN,
        augmentation_profile: str = "none",
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.metadata = metadata.reset_index(drop=True)
        self.max_len = max_len
        self.augmentation = build_pipeline(augmentation_profile, seed=seed)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        audio_id = row["id"]
        label = int(row["label"])

        # Find the audio file
        flac_path = self.data_dir / f"{audio_id}.flac"
        if not flac_path.exists():
            # Try recursive search
            matches = list(self.data_dir.rglob(f"{audio_id}.flac"))
            if matches:
                flac_path = matches[0]
            else:
                raise FileNotFoundError(f"Audio not found: {audio_id}")

        waveform, sr = load_audio(flac_path, target_sr=SAMPLE_RATE)

        # Apply augmentation
        result = self.augmentation(waveform)
        waveform = result.waveform

        # Pad or trim
        waveform_np = waveform.numpy()
        if len(waveform_np) >= self.max_len:
            waveform_np = waveform_np[:self.max_len]
        else:
            reps = (self.max_len // len(waveform_np)) + 1
            waveform_np = np.tile(waveform_np, reps)[:self.max_len]

        return {
            "waveform": torch.tensor(waveform_np, dtype=torch.float32),
            "label": label,
            "audio_id": audio_id,
        }


def collate_fn(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"waveforms": waveforms, "labels": labels}


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []

    for batch in loader:
        waveforms = batch["waveforms"].to(device)
        labels = batch["labels"]

        logits = model(waveforms)
        scores = logits[:, LABEL_BONAFIDE].cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.numpy().tolist())

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    eer_result = compute_eer(scores_arr, labels_arr)
    return eer_result.eer, eer_result.threshold


def main():
    parser = argparse.ArgumentParser(description="Train SIDQ model")
    parser.add_argument("--data-root", type=Path, default=Path("ArA-DF-2026/data"))
    parser.add_argument("--meta-root", type=Path, default=Path("ArA-DF-2026/metadata"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--augmentation", type=str, default="none",
                       choices=["none", "mild", "robust", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--patience", type=int, default=7)
    # --- Credit / compute optimization (matters most on metered Colab GPU) ---
    parser.add_argument("--val-subset", type=int, default=3000,
                        help="Validate on a fixed random N-clip dev subset each epoch "
                             "(0 = full dev). Full dev is always evaluated once at the end. "
                             "Halves per-epoch GPU time on Colab.")
    parser.add_argument("--limit-train", type=int, default=0,
                        help="Cap training clips (0 = all 22,500). Use for quick experiments.")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="Wall-clock budget guard: stop before starting an epoch that "
                             "would exceed this many hours (0 = no limit). Protects credits.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (Colab T4: 2; high-core CPU: more).")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from a checkpoint (restores model+optimizer+best EER).")
    parser.add_argument("--attn", type=str, default="auto",
                        choices=["auto", "eager", "sdpa"],
                        help="XLS-R attention impl. 'auto' = sdpa on CUDA, eager otherwise.")
    args = parser.parse_args()

    seed_everything(args.seed)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    attn = ("sdpa" if device.type == "cuda" else "eager") if args.attn == "auto" else args.attn

    # Load metadata
    train_meta = pd.read_parquet(args.meta_root / "train.parquet")
    val_meta = pd.read_parquet(args.meta_root / "dev.parquet")

    if args.smoke:
        train_meta = train_meta.head(64)
        val_meta = val_meta.head(64)
        args.epochs = 2
        args.batch_size = 4
        logger.info("SMOKE TEST MODE: using 64 samples, 2 epochs")

    if args.limit_train and not args.smoke:
        train_meta = train_meta.sample(
            n=min(args.limit_train, len(train_meta)), random_state=args.seed
        ).reset_index(drop=True)
        logger.info("Limiting training set to %d clips", len(train_meta))

    # Fixed random dev subset for per-epoch validation (cheap, deterministic).
    # Full dev is still evaluated once at the very end for an honest final number.
    if args.val_subset and not args.smoke and args.val_subset < len(val_meta):
        val_sub_meta = val_meta.sample(n=args.val_subset, random_state=args.seed).reset_index(drop=True)
        logger.info("Per-epoch validation on fixed %d-clip dev subset (full dev at end)",
                    len(val_sub_meta))
    else:
        val_sub_meta = val_meta

    # Datasets
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "dev"

    train_dataset = AraDFDataset(train_dir, train_meta,
                                 augmentation_profile=args.augmentation, seed=args.seed)
    val_dataset = AraDFDataset(val_dir, val_sub_meta,
                               augmentation_profile="none", seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                           shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Model
    model = SIDQXLSRModel(freeze_frontend=True, hidden_dim=args.hidden_dim,
                          attn_implementation=attn)
    model = model.to(device)

    # Only train backend parameters
    trainable_params = model.trainable_parameters()
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class-weighted CE (2:1 spoofed:bonafide imbalance)
    weights = torch.tensor([0.33, 0.67]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optional resume (restores backend weights, optimizer, best EER, epoch offset)
    best_eer = 1.0
    start_epoch = 0
    patience_counter = 0
    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_eer = ckpt.get("best_eer", 1.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info("Resumed at epoch %d (best EER %.2f%%)", start_epoch, best_eer * 100)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop with wall-clock budget guard
    t_start = time.time()
    epoch_times: list[float] = []

    for epoch in range(start_epoch, args.epochs):
        # Budget guard: don't start an epoch we can't finish within --max-hours.
        if args.max_hours > 0 and epoch_times:
            avg = sum(epoch_times) / len(epoch_times)
            elapsed_h = (time.time() - t_start) / 3600
            if elapsed_h + avg / 3600 > args.max_hours:
                logger.info("Stopping: next epoch (~%.2fh) would exceed --max-hours=%.2f "
                            "(elapsed %.2fh). Credit guard.", avg / 3600, args.max_hours, elapsed_h)
                break

        t_epoch = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_eer, threshold = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_times.append(time.time() - t_epoch)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val EER: {val_eer*100:.2f}% | "
            f"Threshold: {threshold:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{epoch_times[-1]:.0f}s"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            patience_counter = 0
            ckpt_path = args.output_dir / "best_sidq.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer": best_eer,
                "threshold": threshold,
                "config": vars(args),
            }, ckpt_path)
            logger.info(f"  ★ New best EER: {best_eer*100:.2f}% (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final honest evaluation on the FULL dev set with the best checkpoint.
    if not args.smoke and len(val_sub_meta) < len(val_meta):
        best_ckpt = args.output_dir / "best_sidq.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=False)
        full_val_ds = AraDFDataset(val_dir, val_meta, augmentation_profile="none", seed=args.seed)
        full_val_loader = DataLoader(full_val_ds, batch_size=args.batch_size * 2,
                                     shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
        full_eer, full_thr = evaluate(model, full_val_loader, device)
        logger.info("FULL dev EER (best checkpoint): %.2f%% (threshold %.4f)",
                    full_eer * 100, full_thr)

    logger.info(f"Training complete. Best (subset) EER: {best_eer*100:.2f}%")


if __name__ == "__main__":
    main()
