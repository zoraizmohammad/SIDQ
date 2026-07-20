"""Reverberation augmentation via synthetic room impulse responses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sidq.augment.base import AudioTransform, TransformMetadata

ROOM_PROFILES = {
    "small": {"rt60_range": (0.1, 0.3), "room_dim_range": (3, 6)},
    "medium": {"rt60_range": (0.3, 0.7), "room_dim_range": (6, 15)},
    "large": {"rt60_range": (0.7, 1.5), "room_dim_range": (15, 30)},
}


class ReverbTransform(AudioTransform):
    """Apply synthetic reverberation."""

    def __init__(
        self,
        room_profiles: list[str] | None = None,
        wet_ratio_range: tuple[float, float] = (0.2, 0.8),
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)
        self.room_profiles = room_profiles or list(ROOM_PROFILES.keys())
        self.wet_ratio_range = wet_ratio_range

    @property
    def name(self) -> str:
        return "reverberation"

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        profile_name = rng.choice(self.room_profiles)
        profile = ROOM_PROFILES[profile_name]

        rt60 = rng.uniform(*profile["rt60_range"])
        wet_ratio = rng.uniform(*self.wet_ratio_range)

        rir = self._generate_rir(rt60, sample_rate, rng)
        reverbed = self._apply_rir(waveform, rir, wet_ratio)

        return reverbed, TransformMetadata(
            transform_name=self.name,
            parameters={
                "room_profile": profile_name,
                "rt60": rt60,
                "wet_ratio": wet_ratio,
            },
        )

    def _generate_rir(
        self, rt60: float, sample_rate: int, rng: np.random.Generator
    ) -> torch.Tensor:
        """Generate synthetic room impulse response."""
        length = int(rt60 * sample_rate)
        if length < 1:
            length = 1

        t = np.arange(length) / sample_rate
        decay = np.exp(-6.908 * t / rt60)

        noise = rng.standard_normal(length)
        rir = noise * decay
        rir[0] = 1.0

        rir = rir / (np.abs(rir).max() + 1e-8)
        return torch.from_numpy(rir).float()

    def _apply_rir(
        self, waveform: torch.Tensor, rir: torch.Tensor, wet_ratio: float
    ) -> torch.Tensor:
        """Convolve waveform with RIR and mix."""
        # 1D convolution
        wav_2d = waveform.unsqueeze(0).unsqueeze(0)
        rir_2d = rir.unsqueeze(0).unsqueeze(0)

        reverbed = F.conv1d(wav_2d, rir_2d, padding=rir.shape[-1] - 1)
        reverbed = reverbed.squeeze(0).squeeze(0)[: waveform.shape[-1]]

        # Normalize to prevent amplitude explosion
        if reverbed.abs().max() > 0:
            reverbed = reverbed / reverbed.abs().max() * waveform.abs().max()

        mixed = (1 - wet_ratio) * waveform + wet_ratio * reverbed
        return mixed
