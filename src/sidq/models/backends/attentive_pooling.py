"""Attentive statistics pooling backend."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling for variable-length sequences."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Pool sequence with attention weights.

        Args:
            x: (batch, time, dim)
            lengths: optional (batch,) true lengths

        Returns:
            (batch, dim * 2) — concatenation of weighted mean and std.
        """
        attn_weights = self.attention(x).squeeze(-1)

        if lengths is not None:
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)

        mean = (x * attn_weights).sum(dim=1)
        var = ((x - mean.unsqueeze(1)) ** 2 * attn_weights).sum(dim=1)
        std = (var + 1e-8).sqrt()

        return torch.cat([mean, std], dim=-1)


class AttentiveBackend(nn.Module):
    """Attentive statistics pooling + classifier backend."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.pool = AttentiveStatisticsPooling(input_dim, hidden_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pool(features, lengths)
        return self.classifier(pooled)
