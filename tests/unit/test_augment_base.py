"""Tests for augmentation base interface and pipeline."""

import torch

from sidq.augment.base import AudioTransform, TransformMetadata
from sidq.augment.pipeline import AugmentationPipeline


class GainTransform(AudioTransform):
    """Simple test transform that applies random gain."""

    @property
    def name(self) -> str:
        return "test_gain"

    def apply(self, waveform, sample_rate, rng):
        gain = rng.uniform(0.5, 2.0)
        result = waveform * gain
        return result, TransformMetadata(
            transform_name=self.name,
            parameters={"gain": gain},
            seed_used=self._base_seed or 0,
        )


class TestAudioTransform:
    def test_deterministic_with_seed(self):
        t1 = GainTransform(seed=42)
        t2 = GainTransform(seed=42)
        w = torch.randn(16000)
        r1, _ = t1(w)
        r2, _ = t2(w)
        assert torch.allclose(r1, r2)

    def test_probability_zero_skips(self):
        t = GainTransform(probability=0.0, seed=1)
        w = torch.randn(16000)
        r, meta = t(w)
        assert torch.allclose(r, w)
        assert "skipped" in meta.transform_name

    def test_nonfinite_protection(self):
        class BadTransform(AudioTransform):
            @property
            def name(self):
                return "bad"

            def apply(self, waveform, sample_rate, rng):
                result = torch.full_like(waveform, float("nan"))
                return result, TransformMetadata(transform_name="bad")

        t = BadTransform(seed=1)
        w = torch.randn(16000)
        r, meta = t(w)
        assert torch.allclose(r, w)
        assert "failed" in meta.transform_name

    def test_metadata_recorded(self):
        t = GainTransform(seed=10)
        w = torch.randn(16000)
        _, meta = t(w)
        assert meta.transform_name == "test_gain"
        assert "gain" in meta.parameters


class TestPipeline:
    def test_clean_passthrough(self):
        pipeline = AugmentationPipeline(
            transforms=[GainTransform(seed=1)],
            clean_probability=1.0,
            seed=42,
        )
        w = torch.randn(16000)
        result = pipeline(w)
        assert torch.allclose(result.waveform, w)
        assert result.num_applied == 0

    def test_applies_transforms(self):
        pipeline = AugmentationPipeline(
            transforms=[GainTransform(seed=i) for i in range(5)],
            clean_probability=0.0,
            min_transforms=1,
            max_transforms=3,
            seed=42,
        )
        w = torch.randn(16000)
        result = pipeline(w)
        assert result.num_applied >= 1
        assert not torch.allclose(result.waveform, w)

    def test_deterministic(self):
        # Each pipeline needs its own transform instances since they track call count
        t1 = [GainTransform(seed=i) for i in range(3)]
        t2 = [GainTransform(seed=i) for i in range(3)]
        p1 = AugmentationPipeline(transforms=t1, clean_probability=0.0, seed=99)
        p2 = AugmentationPipeline(transforms=t2, clean_probability=0.0, seed=99)
        w = torch.randn(16000)
        r1 = p1(w)
        r2 = p2(w)
        assert torch.allclose(r1.waveform, r2.waveform)
