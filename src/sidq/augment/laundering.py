"""Social-media laundering: multi-stage platform re-encoding simulation."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from sidq.augment.base import AudioTransform, TransformMetadata


class LaunderingTransform(AudioTransform):
    """Simulate social-media platform processing chains."""

    def __init__(
        self,
        probability: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(probability=probability, seed=seed)

    @property
    def name(self) -> str:
        return "laundering"

    @property
    def external_requirements(self) -> list[str]:
        return ["ffmpeg"]

    def apply(
        self, waveform: torch.Tensor, sample_rate: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, TransformMetadata]:
        params = {}
        result = waveform

        # 1. Resample (simulate platform intake)
        intermediate_sr = int(rng.choice([22050, 44100, 48000]))
        result = self._resample(result, sample_rate, intermediate_sr)
        result = self._resample(result, intermediate_sr, sample_rate)
        params["intermediate_sr"] = intermediate_sr

        # 2. Loudness normalization (platforms normalize to -14 LUFS approx)
        target_peak = rng.uniform(0.7, 0.95)
        peak = result.abs().max()
        if peak > 0:
            result = result * (target_peak / peak)
        params["target_peak"] = target_peak

        # 3. Dynamic range compression
        threshold = rng.uniform(0.3, 0.7)
        ratio = rng.uniform(2.0, 6.0)
        result = self._dynamic_range_compress(result, threshold, ratio)
        params["drc_threshold"] = threshold
        params["drc_ratio"] = ratio

        # 4. Low-pass filter (platform bandwidth limiting)
        cutoff_hz = int(rng.choice([8000, 12000, 15000]))
        result = self._lowpass_via_resample(result, sample_rate, cutoff_hz)
        params["lowpass_cutoff"] = cutoff_hz

        # 5. Mild clipping (from normalization overshoot)
        if rng.random() > 0.5:
            clip_level = rng.uniform(0.9, 0.99)
            result = result.clamp(-clip_level, clip_level)
            params["clip_level"] = clip_level

        return result, TransformMetadata(
            transform_name=self.name, parameters=params
        )

    def _resample(
        self, waveform: torch.Tensor, from_sr: int, to_sr: int
    ) -> torch.Tensor:
        """Resample via torchaudio."""
        if from_sr == to_sr:
            return waveform
        result = torchaudio.functional.resample(
            waveform.unsqueeze(0), from_sr, to_sr
        ).squeeze(0)
        return result

    def _dynamic_range_compress(
        self, waveform: torch.Tensor, threshold: float, ratio: float
    ) -> torch.Tensor:
        """Simple soft-knee dynamic range compression."""
        abs_wav = waveform.abs()
        mask = abs_wav > threshold
        compressed = torch.where(
            mask,
            torch.sign(waveform) * (threshold + (abs_wav - threshold) / ratio),
            waveform,
        )
        return compressed

    def _lowpass_via_resample(
        self, waveform: torch.Tensor, sample_rate: int, cutoff: int
    ) -> torch.Tensor:
        """Approximate lowpass by downsampling and upsampling."""
        if cutoff >= sample_rate // 2:
            return waveform
        target_sr = cutoff * 2
        down = torchaudio.functional.resample(
            waveform.unsqueeze(0), sample_rate, target_sr
        )
        up = torchaudio.functional.resample(down, target_sr, sample_rate).squeeze(0)
        # Match length
        if up.shape[-1] > waveform.shape[-1]:
            up = up[: waveform.shape[-1]]
        elif up.shape[-1] < waveform.shape[-1]:
            up = torch.nn.functional.pad(up, (0, waveform.shape[-1] - up.shape[-1]))
        return up
