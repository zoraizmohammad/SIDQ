"""Model registry — unified model construction."""

from __future__ import annotations

import torch
import torch.nn as nn

from sidq.types import ModelConfig


class SIDQModel(nn.Module):
    """Complete SIDQ model: SSL frontend + backend classifier."""

    def __init__(self, frontend: nn.Module, backend: nn.Module, frontend_dim: int):
        super().__init__()
        self.frontend = frontend
        self.backend = backend
        self.frontend_dim = frontend_dim

    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """End-to-end forward: waveform -> logits.

        Args:
            waveform: (batch, num_samples) raw audio.
            lengths: optional (batch,) sample counts.

        Returns:
            Logits (batch, num_classes). Index 1 = bonafide score.
        """
        features = self.frontend(waveform)
        logits = self.backend(features, lengths=lengths)
        return logits

    def get_bonafide_score(self, waveform: torch.Tensor) -> torch.Tensor:
        """Get continuous bonafide score (higher = more bonafide).

        Returns the logit for class index 1 (bonafide).
        """
        logits = self.forward(waveform)
        return logits[:, 1]


def build_model(config: ModelConfig) -> SIDQModel:
    """Build a model from configuration."""
    from sidq.models.backends.aasist import AASISTBackend
    from sidq.models.ssl.xlsr import XLSRFrontend

    if config.frontend.value == "xlsr_300m":
        frontend = XLSRFrontend(
            freeze=config.freeze_frontend,
            layer_weights=config.layer_weights,
        )
        frontend_dim = frontend.output_dim
    else:
        raise ValueError(f"Unsupported frontend: {config.frontend}")

    if config.backend.value == "aasist":
        backend = AASISTBackend(
            input_dim=frontend_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
        )
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    return SIDQModel(frontend=frontend, backend=backend, frontend_dim=frontend_dim)
