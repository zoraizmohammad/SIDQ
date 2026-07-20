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
from sidq.constants import LABEL_BONAFIDE, LABEL_SPOOFED, SAMPLE_RATE
from sidq.evaluation.eer import compute_eer
from sidq.models.backends.aasist import AASISTBackend
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


class SIDQXLSRModel(nn.Module):
    """XLS-R 300M + AASIST for anti-spoofing."""

    def __init__(self, freeze_frontend: bool = True, hidden_dim: int = 160):
        super().__init__()
        from transformers import Wav2Vec2Model

        logger.info("Loading XLS-R 300M from HuggingFace...")
        self.frontend = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        logger.info("XLS-R loaded!")

        if freeze_frontend:
            for param in self.frontend.parameters():
                param.requires_grad = False

        self.backend = AASISTBackend(
            input_dim=1024,
            hidden_dim=hidden_dim,
            num_graph_layers=2,
            num_classes=2,
        )

    def forward(self, waveform: torch.Tensor, lengths=None) -> torch.Tensor:
        with torch.no_grad() if not any(p.requires_grad for p in self.frontend.parameters()) else torch.enable_grad():
            outputs = self.frontend(waveform)
            features = outputs.last_hidden_state  # (batch, time, 1024)

        logits = self.backend(features, lengths=lengths)
        return logits


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

    # Load metadata
    train_meta = pd.read_parquet(args.meta_root / "train.parquet")
    val_meta = pd.read_parquet(args.meta_root / "dev.parquet")

    if args.smoke:
        train_meta = train_meta.head(64)
        val_meta = val_meta.head(64)
        args.epochs = 2
        args.batch_size = 4
        logger.info("SMOKE TEST MODE: using 64 samples, 2 epochs")

    # Datasets
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "dev"

    train_dataset = AraDFDataset(train_dir, train_meta,
                                 augmentation_profile=args.augmentation, seed=args.seed)
    val_dataset = AraDFDataset(val_dir, val_meta,
                               augmentation_profile="none", seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                           shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Model
    model = SIDQXLSRModel(freeze_frontend=True, hidden_dim=args.hidden_dim)
    model = model.to(device)

    # Only train backend parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Class-weighted CE (2:1 spoofed:bonafide imbalance)
    weights = torch.tensor([0.33, 0.67]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training loop
    best_eer = 1.0
    patience_counter = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_eer, threshold = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val EER: {val_eer*100:.2f}% | "
            f"Threshold: {threshold:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
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

    logger.info(f"Training complete. Best EER: {best_eer*100:.2f}%")


if __name__ == "__main__":
    main()
