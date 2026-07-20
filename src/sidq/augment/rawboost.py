"""RawBoost-style augmentation: convolutive and impulsive noise.

Inspired by the RawBoost augmentation technique for anti-spoofing.
Reference: Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method"
"""

from __future__ import annotations

import numpy as np
import torch

from sidq.augment.base import AudioTransform, TransformMetadata


class RawBoostTransform(AudioTransform):
    """RawBoost-inspired augmentation combining multiple noise types."""

    def __init__(
        self,
        modes: list[str] | None = None,
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)
        self.modes = modes or ["linear_convolutive", "impulsive", "stationary", "combined"]

    @property
    def name(self) -> str:
        return "rawboost"

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        mode = rng.choice(self.modes)

        if mode == "linear_convolutive":
            result = self._linear_convolutive(waveform, rng)
        elif mode == "impulsive":
            result = self._impulsive_noise(waveform, rng)
        elif mode == "stationary":
            result = self._stationary_noise(waveform, rng)
        elif mode == "combined":
            result = self._linear_convolutive(waveform, rng)
            result = self._impulsive_noise(result, rng)
        else:
            result = waveform

        return result, TransformMetadata(
            transform_name=self.name, parameters={"mode": mode}
        )

    def _linear_convolutive(
        self, waveform: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        """Apply linear convolutive noise via random FIR filter."""
        filt_len = rng.integers(5, 50)
        filt = rng.standard_normal(filt_len).astype(np.float32)
        filt[0] = 1.0
        filt = filt / (np.abs(filt).sum() + 1e-8)

        filt_tensor = torch.from_numpy(filt).unsqueeze(0).unsqueeze(0)
        wav_2d = waveform.unsqueeze(0).unsqueeze(0)

        result = torch.nn.functional.conv1d(wav_2d, filt_tensor, padding=filt_len // 2)
        result = result.squeeze(0).squeeze(0)[: waveform.shape[-1]]

        # Mix with original to control severity
        mix = rng.uniform(0.3, 0.8)
        return (1 - mix) * waveform + mix * result

    def _impulsive_noise(
        self, waveform: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        """Add signal-dependent impulsive noise."""
        n_samples = waveform.shape[-1]
        density = rng.uniform(0.001, 0.01)
        n_impulses = int(n_samples * density)

        positions = rng.integers(0, n_samples, size=n_impulses)
        amplitudes = rng.standard_normal(n_impulses).astype(np.float32)

        impulse = torch.zeros_like(waveform)
        for pos, amp in zip(positions, amplitudes, strict=True):
            impulse[pos] += amp * waveform[pos].abs() * rng.uniform(0.1, 0.5)

        return waveform + impulse

    def _stationary_noise(
        self, waveform: torch.Tensor, rng: np.random.Generator
    ) -> torch.Tensor:
        """Add stationary signal-independent noise."""
        snr_db = rng.uniform(15, 35)
        noise = torch.from_numpy(
            rng.standard_normal(waveform.shape[-1]).astype(np.float32)
        )

        signal_power = (waveform**2).mean()
        noise_power = (noise**2).mean()
        if noise_power < 1e-10:
            return waveform

        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        return waveform + noise * scale
