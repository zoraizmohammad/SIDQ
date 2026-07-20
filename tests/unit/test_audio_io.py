"""Tests for audio I/O operations."""

import numpy as np
import pytest
import torch

from sidq.audio.io import (
    crop_audio,
    get_duration_sec,
    normalize_amplitude,
    random_crop,
    validate_waveform,
)


class TestValidateWaveform:
    def test_valid(self):
        w = torch.randn(16000)
        validate_waveform(w)

    def test_empty(self):
        w = torch.tensor([])
        with pytest.raises(ValueError, match="empty"):
            validate_waveform(w)

    def test_nan(self):
        w = torch.tensor([1.0, float("nan"), 0.5])
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_waveform(w)

    def test_inf(self):
        w = torch.tensor([1.0, float("inf"), 0.5])
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_waveform(w)


class TestGetDuration:
    def test_one_second(self):
        w = torch.randn(16000)
        assert abs(get_duration_sec(w, 16000) - 1.0) < 1e-6

    def test_half_second(self):
        w = torch.randn(8000)
        assert abs(get_duration_sec(w, 16000) - 0.5) < 1e-6


class TestNormalize:
    def test_normalizes_peak(self):
        w = torch.tensor([0.5, -0.3, 0.2])
        normalized = normalize_amplitude(w, target_peak=0.95)
        assert abs(normalized.abs().max().item() - 0.95) < 1e-6

    def test_silent_audio(self):
        w = torch.zeros(100)
        normalized = normalize_amplitude(w)
        assert (normalized == 0).all()


class TestCropAudio:
    def test_basic_crop(self):
        w = torch.randn(32000)
        cropped = crop_audio(w, duration_sec=1.0, sample_rate=16000)
        assert cropped.shape[-1] == 16000

    def test_crop_with_offset(self):
        w = torch.randn(48000)
        cropped = crop_audio(w, duration_sec=1.0, sample_rate=16000, offset_sec=1.0)
        assert cropped.shape[-1] == 16000

    def test_crop_pads_short(self):
        w = torch.randn(8000)
        cropped = crop_audio(w, duration_sec=2.0, sample_rate=16000)
        assert cropped.shape[-1] == 32000

    def test_offset_beyond_duration(self):
        w = torch.randn(16000)
        with pytest.raises(ValueError, match="exceeds"):
            crop_audio(w, duration_sec=1.0, sample_rate=16000, offset_sec=2.0)


class TestRandomCrop:
    def test_deterministic_with_seed(self):
        w = torch.randn(48000)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        c1 = random_crop(w, duration_sec=1.0, rng=rng1)
        c2 = random_crop(w, duration_sec=1.0, rng=rng2)
        assert torch.allclose(c1, c2)

    def test_pads_short_audio(self):
        w = torch.randn(4000)
        cropped = random_crop(w, duration_sec=1.0, sample_rate=16000)
        assert cropped.shape[-1] == 16000

    def test_correct_length(self):
        w = torch.randn(64000)
        cropped = random_crop(w, duration_sec=2.0, sample_rate=16000)
        assert cropped.shape[-1] == 32000
