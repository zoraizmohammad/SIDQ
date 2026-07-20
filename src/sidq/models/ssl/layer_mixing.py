"""Learned weighted layer mixing for SSL frontends."""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedLayerMixing(nn.Module):
    """Learns a weighted combination of SSL transformer layer outputs."""

    def __init__(self, num_layers: int, normalize: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.normalize = normalize
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Combine hidden states with learned weights.

        Args:
            hidden_states: Tuple of (num_layers,) tensors each (batch, time, dim).

        Returns:
            Weighted combination (batch, time, dim).
        """
        stacked = torch.stack(hidden_states, dim=0)
        w = torch.softmax(self.weights, dim=0) if self.normalize else self.weights
        mixed = (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)
        return mixed
