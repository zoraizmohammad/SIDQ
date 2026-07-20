"""Noise augmentation: additive noise at configurable SNR."""

from __future__ import annotations

import numpy as np
import torch

from sidq.augment.base import AudioTransform, TransformMetadata


class AdditiveNoiseTransform(AudioTransform):
    """Add procedural noise (white, pink, brown) at target SNR."""

    def __init__(
        self,
        noise_types: list[str] | None = None,
        snr_range: tuple[float, float] = (5.0, 30.0),
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)
        self.noise_types = noise_types or ["white", "pink", "brown"]
        self.snr_range = snr_range

    @property
    def name(self) -> str:
        return "additive_noise"

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        noise_type = rng.choice(self.noise_types)
        target_snr = rng.uniform(self.snr_range[0], self.snr_range[1])

        noise = self._generate_noise(noise_type, waveform.shape[-1], rng)
        noisy = self._mix_at_snr(waveform, noise, target_snr)

        return noisy, TransformMetadata(
            transform_name=self.name,
            parameters={"noise_type": noise_type, "target_snr_db": target_snr},
        )

    def _generate_noise(
        self, noise_type: str, length: int, rng: np.random.Generator
    ) -> torch.Tensor:
        if noise_type == "white":
            noise = torch.from_numpy(rng.standard_normal(length)).float()
        elif noise_type == "pink":
            noise = torch.from_numpy(self._pink_noise(length, rng)).float()
        elif noise_type == "brown":
            noise = torch.from_numpy(self._brown_noise(length, rng)).float()
        else:
            noise = torch.from_numpy(rng.standard_normal(length)).float()
        return noise

    def _pink_noise(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """Generate pink noise via frequency-domain shaping."""
        white = rng.standard_normal(length)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1.0
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=length)
        return pink / (np.abs(pink).max() + 1e-8)

    def _brown_noise(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """Generate brown noise via cumulative sum of white noise."""
        white = rng.standard_normal(length)
        brown = np.cumsum(white)
        return brown / (np.abs(brown).max() + 1e-8)

    def _mix_at_snr(
        self, signal: torch.Tensor, noise: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        """Mix signal and noise at target SNR in dB."""
        signal_power = (signal**2).mean()
        noise_power = (noise**2).mean()

        if noise_power < 1e-10:
            return signal

        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        return signal + noise * scale
