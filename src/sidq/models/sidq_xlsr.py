"""Shared SIDQ model: HuggingFace XLS-R 300M frontend + AASIST backend.

This module is the single source of truth for the native (non-DeepFense) SIDQ
model, imported by both the training script (``scripts/train_sidq.py``) and the
inference script (``scripts/infer_sidq.py``) so the two can never drift apart.

The frontend is a frozen ``facebook/wav2vec2-xls-r-300m`` (315M params) and the
only trainable component is the small AASIST backend (~374K params). Keeping the
frontend frozen is what makes training cheap enough to iterate on a single GPU.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sidq.models.backends.aasist import AASISTBackend

logger = logging.getLogger(__name__)

XLSR_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
XLSR_FEATURE_DIM = 1024


class SIDQXLSRModel(nn.Module):
    """XLS-R 300M (frozen) + AASIST backend for anti-spoofing.

    Args:
        freeze_frontend: If True (default) the XLS-R frontend is frozen and only
            the AASIST backend is trained.
        hidden_dim: AASIST hidden dimension (must match at train and inference).
        attn_implementation: HF attention backend. "eager" is the safe default
            (works on CPU/MPS/CUDA and matches historical training); pass "sdpa"
            on CUDA for a modest speedup.
    """

    def __init__(
        self,
        freeze_frontend: bool = True,
        hidden_dim: int = 160,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        from transformers import Wav2Vec2Model

        logger.info("Loading XLS-R 300M from HuggingFace (%s)...", XLSR_MODEL_NAME)
        self.frontend = Wav2Vec2Model.from_pretrained(
            XLSR_MODEL_NAME,
            attn_implementation=attn_implementation,
        )
        logger.info("XLS-R loaded.")

        self.freeze_frontend = freeze_frontend
        if freeze_frontend:
            self.frontend.eval()
            for param in self.frontend.parameters():
                param.requires_grad = False

        self.backend = AASISTBackend(
            input_dim=XLSR_FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_graph_layers=2,
            num_classes=2,
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return 2-class logits. ``logits[:, LABEL_BONAFIDE]`` is the bonafide score."""
        if self.freeze_frontend:
            with torch.no_grad():
                features = self.frontend(waveform).last_hidden_state
        else:
            features = self.frontend(waveform).last_hidden_state
        return self.backend(features, lengths=lengths)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters that require gradients (the AASIST backend when frozen)."""
        return [p for p in self.parameters() if p.requires_grad]


def build_model(
    hidden_dim: int = 160,
    freeze_frontend: bool = True,
    attn_implementation: str = "eager",
) -> SIDQXLSRModel:
    """Construct a fresh SIDQ model."""
    return SIDQXLSRModel(
        freeze_frontend=freeze_frontend,
        hidden_dim=hidden_dim,
        attn_implementation=attn_implementation,
    )


def load_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    attn_implementation: str = "eager",
) -> tuple[SIDQXLSRModel, dict]:
    """Load a SIDQ model from a ``train_sidq.py`` checkpoint.

    The checkpoint carries ``config`` (the training args) so we can reconstruct
    the exact ``hidden_dim`` — this prevents silent shape mismatches when the
    backend was trained with a non-default width.

    Returns:
        (model in eval mode on ``device``, checkpoint dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = int(config.get("hidden_dim", 160))

    model = build_model(
        hidden_dim=hidden_dim,
        freeze_frontend=True,
        attn_implementation=attn_implementation,
    )
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The frozen frontend weights come from HF, so only backend keys must match;
    # surface anything unexpected in the backend to catch real mismatches.
    backend_missing = [k for k in missing if k.startswith("backend.")]
    if backend_missing:
        raise RuntimeError(
            f"Checkpoint is missing backend weights: {backend_missing[:5]} "
            f"(hidden_dim={hidden_dim}). Was it trained with a different config?"
        )
    model.to(device)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}
