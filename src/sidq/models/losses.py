"""Loss functions for anti-spoofing training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Reduces loss for well-classified examples, focusing on hard cases.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class ConsistencyLoss(nn.Module):
    """Consistency regularization between clean and corrupted views.

    Encourages the model to produce similar scores for clean and
    corrupted versions of the same utterance.
    """

    def __init__(self, method: str = "mse", weight: float = 0.1):
        super().__init__()
        self.method = method
        self.weight = weight

    def forward(
        self, logits_clean: torch.Tensor, logits_corrupted: torch.Tensor
    ) -> torch.Tensor:
        if self.method == "mse":
            return self.weight * F.mse_loss(logits_corrupted, logits_clean.detach())
        elif self.method == "kl":
            p = F.log_softmax(logits_corrupted, dim=-1)
            q = F.softmax(logits_clean.detach(), dim=-1)
            return self.weight * F.kl_div(p, q, reduction="batchmean")
        elif self.method == "cosine":
            cos_sim = F.cosine_similarity(logits_clean.detach(), logits_corrupted, dim=-1)
            return self.weight * (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown consistency method: {self.method}")


class MultiViewLoss(nn.Module):
    """Combined loss for multi-view (clean + corrupted) training.

    L_total = L_ce(clean) + lambda_aug * L_ce(corrupted)
              + lambda_consistency * L_consistency(clean, corrupted)
    """

    def __init__(
        self,
        lambda_aug: float = 1.0,
        lambda_consistency: float = 0.1,
        consistency_method: str = "mse",
    ):
        super().__init__()
        self.lambda_aug = lambda_aug
        self.ce = nn.CrossEntropyLoss()
        self.consistency = ConsistencyLoss(method=consistency_method, weight=lambda_consistency)

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_corrupted: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_clean = self.ce(logits_clean, targets)
        loss_aug = self.ce(logits_corrupted, targets)
        loss_consistency = self.consistency(logits_clean, logits_corrupted)

        total = loss_clean + self.lambda_aug * loss_aug + loss_consistency

        return {
            "total": total,
            "clean": loss_clean,
            "augmented": loss_aug,
            "consistency": loss_consistency,
        }
