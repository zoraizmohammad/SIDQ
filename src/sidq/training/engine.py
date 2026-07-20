"""Training engine with mixed precision, checkpointing, and EER validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from sidq.evaluation.eer import compute_eer


@dataclass
class TrainState:
    """Mutable training state for checkpointing."""

    epoch: int = 0
    global_step: int = 0
    best_eer: float = 1.0
    best_epoch: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_eers: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    """Final result from training run."""

    best_eer: float
    best_epoch: int
    total_epochs: int
    checkpoint_path: Path | None = None


class Trainer:
    """Training engine for anti-spoofing models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        criterion: nn.Module | None = None,
        device: torch.device | None = None,
        mixed_precision: bool = True,
        grad_clip: float = 1.0,
        grad_accumulation: int = 1,
        checkpoint_dir: Path | None = None,
        early_stopping_patience: int = 5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.grad_clip = grad_clip
        self.grad_accumulation = grad_accumulation
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience

        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.state = TrainState()

        self.model.to(self.device)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            waveforms = batch["waveforms"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(self.device)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(waveforms, lengths=lengths)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accumulation

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.state.global_step += 1

            total_loss += loss.item() * self.grad_accumulation
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        self.state.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation, computing EER and loss."""
        self.model.eval()
        all_scores: list[float] = []
        all_labels: list[int] = []
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            waveforms = batch["waveforms"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(self.device)

            logits = self.model(waveforms, lengths=lengths)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            bonafide_scores = logits[:, 1].cpu().numpy()
            all_scores.extend(bonafide_scores.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        scores_arr = np.array(all_scores)
        labels_arr = np.array(all_labels)

        eer_result = compute_eer(scores_arr, labels_arr)

        metrics = {
            "val_loss": total_loss / max(num_batches, 1),
            "val_eer": eer_result.eer,
            "val_threshold": eer_result.threshold,
        }

        self.state.val_eers.append(eer_result.eer)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
    ) -> TrainResult:
        """Full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(epochs):
            self.state.epoch = epoch
            self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            val_eer = val_metrics["val_eer"]

            if val_eer < self.state.best_eer:
                self.state.best_eer = val_eer
                self.state.best_epoch = epoch
                patience_counter = 0
                if self.checkpoint_dir:
                    self._save_checkpoint("best.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

        return TrainResult(
            best_eer=self.state.best_eer,
            best_epoch=self.state.best_epoch,
            total_epochs=self.state.epoch + 1,
            checkpoint_path=(
                self.checkpoint_dir / "best.pt" if self.checkpoint_dir else None
            ),
        )

    def _save_checkpoint(self, filename: str) -> None:
        """Save model and optimizer state."""
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.state.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_eer": self.state.best_eer,
                "global_step": self.state.global_step,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Resume from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.best_eer = checkpoint.get("best_eer", 1.0)
        self.state.global_step = checkpoint.get("global_step", 0)
