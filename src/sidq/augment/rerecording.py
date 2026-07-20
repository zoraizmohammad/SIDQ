"""Re-recording simulation: speaker-room-microphone chain."""

from __future__ import annotations

import numpy as np
import torch

from sidq.augment.base import AudioTransform, TransformMetadata
from sidq.augment.noise import AdditiveNoiseTransform
from sidq.augment.reverberation import ReverbTransform


class RerecordingTransform(AudioTransform):
    """Simulate re-recording through speaker-room-mic chain."""

    def __init__(
        self,
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)

    @property
    def name(self) -> str:
        return "rerecording"

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        params = {}

        # 1. Speaker frequency response (mild low-pass)
        speaker_cutoff = rng.uniform(4000, 12000)
        result = self._apply_lowpass(waveform, sample_rate, speaker_cutoff)
        params["speaker_cutoff"] = speaker_cutoff

        # 2. Room impulse response
        reverb = ReverbTransform(room_profiles=["small", "medium"], seed=int(rng.integers(1e6)))
        result, _ = reverb.apply(result, sample_rate, rng)

        # 3. Environmental noise
        noise_snr = rng.uniform(20, 40)
        noise_t = AdditiveNoiseTransform(
            noise_types=["white"], snr_range=(noise_snr, noise_snr), seed=int(rng.integers(1e6))
        )
        result, _ = noise_t.apply(result, sample_rate, rng)
        params["noise_snr"] = noise_snr

        # 4. Mild nonlinear distortion
        distortion_amount = rng.uniform(0.01, 0.1)
        result = self._soft_clip(result, distortion_amount)
        params["distortion"] = distortion_amount

        # 5. Microphone frequency response variation
        mic_gain = rng.uniform(0.7, 1.3)
        result = result * mic_gain
        params["mic_gain"] = mic_gain

        return result, TransformMetadata(
            transform_name=self.name, parameters=params
        )

    def _apply_lowpass(
        self, waveform: torch.Tensor, sample_rate: int, cutoff: float
    ) -> torch.Tensor:
        """Simple first-order IIR lowpass approximation."""
        alpha = cutoff / (cutoff + sample_rate / (2 * np.pi))
        result = torch.zeros_like(waveform)
        result[0] = waveform[0]
        for i in range(1, len(waveform)):
            result[i] = alpha * waveform[i] + (1 - alpha) * result[i - 1]
        return result

    def _soft_clip(self, waveform: torch.Tensor, amount: float) -> torch.Tensor:
        """Apply soft clipping distortion."""
        return torch.tanh(waveform * (1 + amount * 10)) / (1 + amount * 5)
