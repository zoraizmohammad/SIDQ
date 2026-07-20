"""Tests for typed configuration schemas."""

import math

import pytest
from pydantic import ValidationError

from sidq.types import (
    AudioConfig,
    AugmentationProfile,
    EvaluationConfig,
    ExperimentConfig,
    FusionConfig,
    Label,
    ModelConfig,
    SubmissionRecord,
    TrainingConfig,
)


class TestLabel:
    def test_bonafide_value(self):
        assert Label.BONAFIDE == 1

    def test_spoofed_value(self):
        assert Label.SPOOFED == 0


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16_000
        assert cfg.mono is True
        assert cfg.crop_duration_sec == 4.0

    def test_invalid_sample_rate(self):
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate=12345)

    def test_valid_sample_rates(self):
        for sr in (8000, 16000, 22050, 44100, 48000):
            cfg = AudioConfig(sample_rate=sr)
            assert cfg.sample_rate == sr


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.frontend.value == "xlsr_300m"
        assert cfg.backend.value == "aasist"
        assert cfg.freeze_frontend is True


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 30
        assert cfg.seed == 42
        assert cfg.augmentation == AugmentationProfile.NONE

    def test_custom_values(self):
        cfg = TrainingConfig(epochs=10, seed=123, batch_size=32)
        assert cfg.epochs == 10
        assert cfg.seed == 123
        assert cfg.batch_size == 32


class TestEvaluationConfig:
    def test_defaults(self):
        cfg = EvaluationConfig()
        assert cfg.aggregation == "mean"

    def test_invalid_aggregation(self):
        with pytest.raises(ValidationError):
            EvaluationConfig(aggregation="invalid")


class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.method == "weighted_mean"

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            FusionConfig(method="magic")


class TestExperimentConfig:
    def test_minimal(self):
        cfg = ExperimentConfig(name="test_exp")
        assert cfg.name == "test_exp"
        assert cfg.audio.sample_rate == 16_000


class TestSubmissionRecord:
    def test_valid(self):
        rec = SubmissionRecord(audio_id="test_001", logit=1.5)
        assert rec.audio_id == "test_001"
        assert rec.logit == 1.5

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            SubmissionRecord(audio_id="test_001", logit=math.nan)

    def test_inf_rejected(self):
        with pytest.raises(ValidationError):
            SubmissionRecord(audio_id="test_001", logit=math.inf)

    def test_negative_inf_rejected(self):
        with pytest.raises(ValidationError):
            SubmissionRecord(audio_id="test_001", logit=-math.inf)
