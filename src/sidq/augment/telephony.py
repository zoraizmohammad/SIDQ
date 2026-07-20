"""Telephony channel simulation: bandwidth reduction, quantization."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from sidq.augment.base import AudioTransform, TransformMetadata


class TelephonyTransform(AudioTransform):
    """Simulate telephony channel effects."""

    def __init__(
        self,
        mode: str = "narrowband",
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)
        self.mode = mode

    @property
    def name(self) -> str:
        return "telephony"

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        mode = rng.choice(["narrowband", "wideband"]) if self.mode == "random" else self.mode

        if mode == "narrowband":
            target_sr = 8000
            lowpass = 3400
            highpass = 300
        else:
            target_sr = 16000
            lowpass = 7000
            highpass = 50

        result = self._apply_bandwidth(waveform, sample_rate, target_sr, lowpass, highpass)

        # Optional: apply mu-law or A-law quantization
        apply_companding = rng.random() > 0.5
        if apply_companding:
            law = rng.choice(["mu", "a"])
            result = self._apply_companding(result, law)
        else:
            law = "none"

        return result, TransformMetadata(
            transform_name=self.name,
            parameters={"mode": mode, "target_sr": target_sr, "companding": law},
        )

    def _apply_bandwidth(
        self,
        waveform: torch.Tensor,
        source_sr: int,
        target_sr: int,
        lowpass: int,
        highpass: int,
    ) -> torch.Tensor:
        """Apply bandwidth restriction via resampling."""
        # Downsample to target SR
        if source_sr != target_sr:
            waveform_2d = waveform.unsqueeze(0)
            down = torchaudio.functional.resample(waveform_2d, source_sr, target_sr)
            # Upsample back
            up = torchaudio.functional.resample(down, target_sr, source_sr)
            result = up.squeeze(0)
            # Trim or pad to match original length
            if result.shape[-1] > waveform.shape[-1]:
                result = result[: waveform.shape[-1]]
            elif result.shape[-1] < waveform.shape[-1]:
                result = torch.nn.functional.pad(
                    result, (0, waveform.shape[-1] - result.shape[-1])
                )
            return result
        return waveform

    def _apply_companding(self, waveform: torch.Tensor, law: str) -> torch.Tensor:
        """Apply mu-law or A-law companding."""
        x = waveform.clamp(-1.0, 1.0)
        if law == "mu":
            mu = 255.0
            compressed = torch.sign(x) * torch.log1p(mu * x.abs()) / np.log(1 + mu)
            expanded = torch.sign(compressed) * (
                (1 + mu) ** compressed.abs() - 1
            ) / mu
        else:
            # A-law approximation
            a = 87.6
            abs_x = x.abs()
            mask = abs_x < (1.0 / a)
            compressed = torch.where(
                mask,
                a * abs_x / (1 + np.log(a)),
                (1 + torch.log(a * abs_x.clamp(min=1e-10))) / (1 + np.log(a)),
            )
            compressed = torch.sign(x) * compressed
            expanded = compressed  # Simplified: expand approximation
        return expanded
