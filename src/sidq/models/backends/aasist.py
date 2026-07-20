"""AASIST-inspired spectro-temporal graph attention backend.

Adapted from the AASIST architecture for anti-spoofing detection.
Implements graph attention over spectral and temporal feature groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """Apply graph attention.

        Args:
            x: (batch, nodes, in_dim)
            adj: optional adjacency mask (batch, nodes, nodes)
        """
        h = self.fc(x)
        batch, nodes, dim = h.shape

        hi = h.unsqueeze(2).expand(-1, -1, nodes, -1)
        hj = h.unsqueeze(1).expand(-1, nodes, -1, -1)
        attn_input = torch.cat([hi, hj], dim=-1)
        e = F.leaky_relu(self.attn_fc(attn_input).squeeze(-1), negative_slope=0.2)

        if adj is not None:
            e = e.masked_fill(adj == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.bmm(alpha, h)
        return out


class HeterogeneousGraphLayer(nn.Module):
    """Heterogeneous graph attention with spectral and temporal nodes."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.spec_gat = GraphAttentionLayer(in_dim, hidden_dim, dropout)
        self.temp_gat = GraphAttentionLayer(in_dim, hidden_dim, dropout)
        self.cross_gat = GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, spec_nodes: torch.Tensor, temp_nodes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process spectral and temporal node groups.

        Args:
            spec_nodes: (batch, n_spec, dim)
            temp_nodes: (batch, n_temp, dim)
        """
        spec_out = self.spec_gat(spec_nodes)
        temp_out = self.temp_gat(temp_nodes)

        combined = torch.cat([spec_out, temp_out], dim=1)
        cross_out = self.cross_gat(combined)

        n_spec = spec_nodes.shape[1]
        spec_final = self.norm(cross_out[:, :n_spec, :])
        temp_final = self.norm(cross_out[:, n_spec:, :])

        return spec_final, temp_final


class AASISTBackend(nn.Module):
    """AASIST-style backend classifier for anti-spoofing.

    Takes SSL features and classifies via graph attention over
    spectral and temporal representations.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 160,
        num_graph_layers: int = 2,
        num_spec_nodes: int = 4,
        num_temp_nodes: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_spec_nodes = num_spec_nodes
        self.num_temp_nodes = num_temp_nodes

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.spec_queries = nn.Parameter(torch.randn(num_spec_nodes, hidden_dim) * 0.02)
        self.temp_queries = nn.Parameter(torch.randn(num_temp_nodes, hidden_dim) * 0.02)

        self.graph_layers = nn.ModuleList([
            HeterogeneousGraphLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_graph_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Classify features.

        Args:
            features: (batch, time, input_dim) from SSL frontend.
            lengths: optional (batch,) actual lengths for masking.

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.input_proj(features)

        if lengths is not None:
            mask = self._length_mask(lengths, x.shape[1])
            x = x * mask.unsqueeze(-1)

        # Cross-attention: queries attend to feature sequence x
        # spec_queries: (num_spec_nodes, hidden_dim), x: (batch, time, hidden_dim)
        spec_attn = torch.matmul(
            self.spec_queries.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)
        )
        spec_attn = torch.softmax(spec_attn / (self.hidden_dim ** 0.5), dim=-1)
        spec_nodes = torch.bmm(spec_attn, x)

        temp_attn = torch.matmul(
            self.temp_queries.unsqueeze(0).expand(x.shape[0], -1, -1), x.transpose(1, 2)
        )
        temp_attn = torch.softmax(temp_attn / (self.hidden_dim ** 0.5), dim=-1)
        temp_nodes = torch.bmm(temp_attn, x)

        for graph_layer in self.graph_layers:
            spec_nodes, temp_nodes = graph_layer(spec_nodes, temp_nodes)

        spec_pool = spec_nodes.mean(dim=1)
        temp_pool = temp_nodes.mean(dim=1)
        combined = torch.cat([spec_pool, temp_pool], dim=-1)

        logits = self.readout(combined)
        return logits

    def _length_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create binary mask from lengths."""
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return mask.float()
