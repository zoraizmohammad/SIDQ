"""Tests for AWS integration (all mocked, no live calls)."""

import pytest

from sidq.aws.config import AWSConfig, CostGuard
from sidq.aws.polly import PollyRequest, estimate_polly_cost, list_arabic_voices, synthesize
from sidq.aws.transcribe import normalize_arabic_text


class TestCostGuard:
    def test_within_budget(self):
        guard = CostGuard(budget_usd=10.0, enabled=True)
        guard.check(5.0)

    def test_exceeds_budget(self):
        guard = CostGuard(budget_usd=1.0, spent_usd=0.8, enabled=True)
        with pytest.raises(RuntimeError, match="exceed"):
            guard.check(0.5)

    def test_disabled_raises(self):
        guard = CostGuard(enabled=False)
        with pytest.raises(RuntimeError, match="disabled"):
            guard.check(0.01)

    def test_zero_budget(self):
        guard = CostGuard(budget_usd=0.0, enabled=True)
        with pytest.raises(RuntimeError, match="No budget"):
            guard.check(0.01)

    def test_remaining(self):
        guard = CostGuard(budget_usd=10.0, spent_usd=3.0, enabled=True)
        assert guard.remaining_usd == 7.0


class TestAWSConfig:
    def test_disabled_by_default(self):
        config = AWSConfig()
        assert config.enabled is False
        assert config.dry_run is True

    def test_validate_disabled_raises(self):
        config = AWSConfig(enabled=False)
        with pytest.raises(RuntimeError, match="disabled"):
            config.validate_enabled()

    def test_validate_enabled_passes(self):
        config = AWSConfig(enabled=True)
        config.validate_enabled()


class TestPolly:
    def test_cost_estimate(self):
        cost = estimate_polly_cost("Hello world", engine="standard")
        assert cost > 0
        assert cost < 0.001

    def test_neural_more_expensive(self):
        text = "test text"
        standard = estimate_polly_cost(text, "standard")
        neural = estimate_polly_cost(text, "neural")
        assert neural > standard

    def test_dry_run(self, tmp_path):
        config = AWSConfig(enabled=True, dry_run=True)
        config.cost_guard = CostGuard(budget_usd=10.0, enabled=True)
        request = PollyRequest(text="مرحبا")
        result = synthesize(request, config, tmp_path)
        assert result.cost_estimate_usd > 0

    def test_disabled_raises(self, tmp_path):
        config = AWSConfig(enabled=False)
        request = PollyRequest(text="test")
        with pytest.raises(RuntimeError, match="disabled"):
            synthesize(request, config, tmp_path)

    def test_list_voices_dry_run(self):
        config = AWSConfig(enabled=True, dry_run=True)
        voices = list_arabic_voices(config)
        assert len(voices) >= 1

    def test_content_hash(self):
        r1 = PollyRequest(text="hello")
        r2 = PollyRequest(text="hello")
        r3 = PollyRequest(text="different")
        assert r1.content_hash == r2.content_hash
        assert r1.content_hash != r3.content_hash


class TestTranscribe:
    def test_normalize_arabic_alef(self):
        text = normalize_arabic_text("أحمد وإبراهيم وآدم")
        assert "أ" not in text
        assert "إ" not in text
        assert "آ" not in text
        assert "ا" in text

    def test_normalize_removes_diacritics(self):
        text = normalize_arabic_text("مَرْحَبًا")
        assert "َ" not in text
        assert "ْ" not in text

    def test_normalize_whitespace(self):
        text = normalize_arabic_text("hello   world  ")
        assert text == "hello world"
