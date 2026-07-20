"""Audio I/O operations: loading, resampling, and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio

from sidq.constants import MONO_CHANNELS, SAMPLE_RATE


def load_audio(
    path: Path | str,
    target_sr: int = SAMPLE_RATE,
    mono: bool = True,
    max_duration_sec: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Load audio file and return waveform tensor and sample rate.

    Returns:
        Tuple of (waveform, sample_rate) where waveform is shape (num_samples,)
        for mono or (channels, num_samples) for multi-channel.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if mono and waveform.shape[0] > MONO_CHANNELS:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    if mono:
        waveform = waveform.squeeze(0)

    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * sr)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

    validate_waveform(waveform)
    return waveform, sr


def validate_waveform(waveform: torch.Tensor) -> None:
    """Validate that waveform contains finite, non-empty values."""
    if waveform.numel() == 0:
        raise ValueError("Waveform is empty (zero samples)")
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform contains NaN or Inf values")


def get_duration_sec(waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> float:
    """Get duration of waveform in seconds."""
    return waveform.shape[-1] / sample_rate


def normalize_amplitude(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Normalize waveform amplitude to target peak level."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform * (target_peak / peak)
    return waveform


def crop_audio(
    waveform: torch.Tensor,
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    offset_sec: float = 0.0,
) -> torch.Tensor:
    """Crop waveform to specified duration starting from offset."""
    start_sample = int(offset_sec * sample_rate)
    num_samples = int(duration_sec * sample_rate)
    end_sample = start_sample + num_samples

    if start_sample >= waveform.shape[-1]:
        raise ValueError(
            f"Offset {offset_sec}s exceeds waveform duration "
            f"{waveform.shape[-1] / sample_rate:.2f}s"
        )

    cropped = waveform[..., start_sample:end_sample]

    if cropped.shape[-1] < num_samples:
        pad_size = num_samples - cropped.shape[-1]
        cropped = torch.nn.functional.pad(cropped, (0, pad_size))

    return cropped


def random_crop(
    waveform: torch.Tensor,
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Take a random crop of the specified duration."""
    num_samples = int(duration_sec * sample_rate)
    total_samples = waveform.shape[-1]

    if total_samples <= num_samples:
        pad_size = num_samples - total_samples
        return torch.nn.functional.pad(waveform, (0, pad_size))

    if rng is None:
        rng = np.random.default_rng()

    max_offset = total_samples - num_samples
    start = rng.integers(0, max_offset + 1)
    return waveform[..., start : start + num_samples]
