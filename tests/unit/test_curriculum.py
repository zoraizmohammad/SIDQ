"""Tests for augmentation curriculum profiles."""

import torch

from sidq.augment.curriculum import build_pipeline


class TestBuildPipeline:
    def test_none_profile(self):
        pipe = build_pipeline("none")
        w = torch.randn(16000)
        result = pipe(w)
        assert torch.allclose(result.waveform, w)
        assert result.num_applied == 0

    def test_mild_profile(self):
        pipe = build_pipeline("mild", seed=42)
        w = torch.randn(16000)
        # Run multiple times to check some get augmented
        applied_count = 0
        for i in range(20):
            pipe._call_count = i
            result = pipe(w)
            applied_count += result.num_applied
        # At least some should be augmented (prob ~35%)
        assert applied_count > 0

    def test_robust_profile(self):
        pipe = build_pipeline("robust", seed=42)
        assert len(pipe.transforms) >= 4
        assert pipe.clean_probability == 0.30

    def test_hard_profile(self):
        pipe = build_pipeline("hard", seed=42)
        assert len(pipe.transforms) >= 6
        assert pipe.clean_probability == 0.15

    def test_invalid_profile(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown"):
            build_pipeline("extreme")
