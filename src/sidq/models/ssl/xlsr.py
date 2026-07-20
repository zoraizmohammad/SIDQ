"""XLS-R 300M frontend for audio feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn


class XLSRFrontend(nn.Module):
    """XLS-R 300M self-supervised frontend.

    Loads pretrained wav2vec2 XLS-R 300M and extracts hidden-state features.
    Supports frozen and unfrozen operation, and learned layer mixing.
    """

    MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
    HIDDEN_DIM = 1024
    NUM_LAYERS = 24

    def __init__(
        self,
        model_name: str | None = None,
        freeze: bool = True,
        layer_weights: bool = False,
        output_layer: int = -1,
    ):
        super().__init__()
        self.model_name = model_name or self.MODEL_NAME
        self.freeze = freeze
        self.use_layer_weights = layer_weights
        self.output_layer = output_layer

        self._model: nn.Module | None = None
        self._loaded = False

        if layer_weights:
            self.layer_weights = nn.Parameter(
                torch.ones(self.NUM_LAYERS + 1) / (self.NUM_LAYERS + 1)
            )

    def load_pretrained(self) -> None:
        """Load pretrained weights from Hugging Face."""
        try:
            from transformers import Wav2Vec2Model

            self._model = Wav2Vec2Model.from_pretrained(self.model_name)
            self._loaded = True

            if self.freeze:
                self._freeze_parameters()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load XLS-R model '{self.model_name}'. "
                f"Ensure transformers is installed and model is accessible. "
                f"Error: {e}"
            ) from e

    def _freeze_parameters(self) -> None:
        """Freeze all frontend parameters."""
        if self._model is not None:
            for param in self._model.parameters():
                param.requires_grad = False

    def unfreeze_top_layers(self, n_layers: int = 4) -> None:
        """Unfreeze the top N transformer layers for fine-tuning."""
        if self._model is None:
            return
        for param in self._model.parameters():
            param.requires_grad = False
        if hasattr(self._model, "encoder") and hasattr(self._model.encoder, "layers"):
            layers = self._model.encoder.layers
            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from raw waveform.

        Args:
            waveform: (batch, num_samples) raw audio at 16kHz.

        Returns:
            Features tensor of shape (batch, time_steps, hidden_dim).
        """
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call load_pretrained() first, or ensure "
                "the checkpoint is available."
            )

        assert self._model is not None
        outputs = self._model(waveform, output_hidden_states=self.use_layer_weights)

        if self.use_layer_weights:
            hidden_states = outputs.hidden_states
            stacked = torch.stack(hidden_states, dim=0)
            weights = torch.softmax(self.layer_weights, dim=0)
            features = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        elif self.output_layer == -1:
            features = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            features = hidden_states[self.output_layer]

        return features

    @property
    def output_dim(self) -> int:
        return self.HIDDEN_DIM

    @property
    def is_loaded(self) -> bool:
        return self._loaded
