"""WavLM frontend for audio feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from sidq.models.ssl.layer_mixing import LearnedLayerMixing


class WavLMFrontend(nn.Module):
    """WavLM self-supervised frontend.

    Supports WavLM Base+ (768-dim, 12 layers) and WavLM Large (1024-dim, 24 layers).
    """

    CONFIGS = {
        "wavlm_base_plus": {
            "model_name": "microsoft/wavlm-base-plus",
            "hidden_dim": 768,
            "num_layers": 12,
        },
        "wavlm_large": {
            "model_name": "microsoft/wavlm-large",
            "hidden_dim": 1024,
            "num_layers": 24,
        },
    }

    def __init__(
        self,
        variant: str = "wavlm_base_plus",
        freeze: bool = True,
        layer_weights: bool = True,
    ):
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unknown WavLM variant: {variant}. Choose from {list(self.CONFIGS)}")

        self.variant = variant
        self.config = self.CONFIGS[variant]
        self.freeze = freeze
        self.use_layer_weights = layer_weights

        self._model: nn.Module | None = None
        self._loaded = False

        if layer_weights:
            self.layer_mixer = LearnedLayerMixing(self.config["num_layers"] + 1)

    @property
    def output_dim(self) -> int:
        return self.config["hidden_dim"]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_pretrained(self) -> None:
        """Load pretrained WavLM from Hugging Face."""
        try:
            from transformers import WavLMModel

            self._model = WavLMModel.from_pretrained(self.config["model_name"])
            self._loaded = True

            if self.freeze:
                self._freeze_parameters()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load WavLM '{self.config['model_name']}'. "
                f"Error: {e}"
            ) from e

    def _freeze_parameters(self) -> None:
        if self._model is not None:
            for param in self._model.parameters():
                param.requires_grad = False

    def unfreeze_top_layers(self, n_layers: int = 4) -> None:
        """Unfreeze the top N encoder layers."""
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
            waveform: (batch, num_samples) at 16kHz.

        Returns:
            Features (batch, time_steps, hidden_dim).
        """
        if not self._loaded:
            raise RuntimeError("WavLM not loaded. Call load_pretrained() first.")

        assert self._model is not None
        outputs = self._model(waveform, output_hidden_states=True)

        if self.use_layer_weights:
            features = self.layer_mixer(outputs.hidden_states)
        else:
            features = outputs.last_hidden_state

        return features
