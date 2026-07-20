"""Tests for augmentation transforms."""

import torch

from sidq.augment.noise import AdditiveNoiseTransform
from sidq.augment.rawboost import RawBoostTransform
from sidq.augment.reverberation import ReverbTransform
from sidq.augment.telephony import TelephonyTransform


class TestAdditiveNoise:
    def test_output_shape(self):
        t = AdditiveNoiseTransform(seed=42)
        w = torch.randn(16000)
        result, meta = t(w, sample_rate=16000)
        assert result.shape == w.shape

    def test_finite_output(self):
        t = AdditiveNoiseTransform(seed=42)
        w = torch.randn(16000)
        result, _ = t(w)
        assert torch.isfinite(result).all()

    def test_snr_approximate(self):
        t = AdditiveNoiseTransform(snr_range=(20.0, 20.0), noise_types=["white"], seed=42)
        w = torch.randn(32000)
        result, meta = t(w)
        noise_added = result - w
        signal_power = (w**2).mean()
        noise_power = (noise_added**2).mean()
        if noise_power > 0:
            measured_snr = 10 * torch.log10(signal_power / noise_power)
            assert abs(measured_snr.item() - 20.0) < 3.0

    def test_deterministic(self):
        t1 = AdditiveNoiseTransform(seed=42)
        t2 = AdditiveNoiseTransform(seed=42)
        w = torch.randn(16000)
        r1, _ = t1(w)
        r2, _ = t2(w)
        assert torch.allclose(r1, r2)


class TestReverberation:
    def test_output_shape(self):
        t = ReverbTransform(seed=42)
        w = torch.randn(16000)
        result, _ = t(w)
        assert result.shape == w.shape

    def test_finite_output(self):
        t = ReverbTransform(seed=42)
        w = torch.randn(32000)
        result, _ = t(w)
        assert torch.isfinite(result).all()

    def test_metadata(self):
        t = ReverbTransform(seed=42)
        w = torch.randn(16000)
        _, meta = t(w)
        assert "room_profile" in meta.parameters


class TestTelephony:
    def test_output_shape(self):
        t = TelephonyTransform(seed=42)
        w = torch.randn(16000)
        result, _ = t(w)
        assert result.shape == w.shape

    def test_finite_output(self):
        t = TelephonyTransform(seed=42)
        w = torch.randn(16000)
        result, _ = t(w)
        assert torch.isfinite(result).all()


class TestRawBoost:
    def test_output_shape(self):
        t = RawBoostTransform(seed=42)
        w = torch.randn(16000)
        result, _ = t(w)
        assert result.shape == w.shape

    def test_finite_output(self):
        t = RawBoostTransform(seed=42)
        w = torch.randn(16000)
        result, _ = t(w)
        assert torch.isfinite(result).all()

    def test_modes(self):
        for mode in ["linear_convolutive", "impulsive", "stationary", "combined"]:
            t = RawBoostTransform(modes=[mode], seed=42)
            w = torch.randn(16000)
            result, meta = t(w)
            assert result.shape == w.shape
            assert meta.parameters["mode"] == mode
