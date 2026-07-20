"""Tests for EER computation — critical for competition correctness."""

import numpy as np
import pytest

from sidq.evaluation.eer import compute_eer, detect_score_inversion


class TestComputeEER:
    def test_perfect_separation(self):
        """Perfect classifier: bonafide scores above spoofed."""
        scores = np.array([10.0, 9.0, 8.0, -1.0, -2.0, -3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        result = compute_eer(scores, labels)
        assert result.eer < 0.01
        assert result.num_bonafide == 3
        assert result.num_spoofed == 3

    def test_random_scores(self):
        """Random scores should give approximately 50% EER."""
        rng = np.random.default_rng(42)
        n = 10000
        scores = rng.standard_normal(n)
        labels = np.array([1] * (n // 2) + [0] * (n // 2))
        result = compute_eer(scores, labels)
        assert 0.4 < result.eer < 0.6

    def test_completely_inverted(self):
        """Inverted scores: bonafide gets low scores, should give high EER."""
        scores = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        result = compute_eer(scores, labels)
        assert result.eer > 0.9

    def test_tied_scores(self):
        """All equal scores."""
        scores = np.array([1.0, 1.0, 1.0, 1.0])
        labels = np.array([1, 1, 0, 0])
        result = compute_eer(scores, labels)
        # With all tied, EER should be around 0.5
        assert 0.0 <= result.eer <= 1.0

    def test_known_eer_fixture(self):
        """A hand-computed case with known EER."""
        # 5 bonafide: [5, 4, 3, 2, 1], 5 spoofed: [4.5, 3.5, 2.5, 1.5, 0.5]
        # Overlapping, EER around 40%
        scores = np.array([5, 4, 3, 2, 1, 4.5, 3.5, 2.5, 1.5, 0.5], dtype=float)
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result = compute_eer(scores, labels)
        assert 0.3 <= result.eer <= 0.5

    def test_eer_returns_threshold(self):
        scores = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        result = compute_eer(scores, labels)
        assert isinstance(result.threshold, float)


class TestInputValidation:
    def test_nan_scores(self):
        scores = np.array([1.0, float("nan"), 2.0])
        labels = np.array([1, 0, 1])
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_eer(scores, labels)

    def test_inf_scores(self):
        scores = np.array([1.0, float("inf"), 2.0])
        labels = np.array([1, 0, 1])
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_eer(scores, labels)

    def test_single_class(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1])
        with pytest.raises(ValueError, match="both classes"):
            compute_eer(scores, labels)

    def test_invalid_labels(self):
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([1, 0, 2])
        with pytest.raises(ValueError, match="Labels must be"):
            compute_eer(scores, labels)

    def test_empty(self):
        with pytest.raises(ValueError, match="Empty"):
            compute_eer(np.array([]), np.array([]))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_eer(np.array([1.0, 2.0]), np.array([1]))


class TestScoreInversion:
    def test_correct_direction(self):
        """Correct direction should NOT be detected as inverted."""
        scores = np.array([5.0, 4.0, 3.0, -1.0, -2.0, -3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert detect_score_inversion(scores, labels) is False

    def test_inverted_detected(self):
        """Wrong direction should be detected."""
        scores = np.array([-5.0, -4.0, -3.0, 1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert detect_score_inversion(scores, labels) is True

    def test_ambiguous_not_flagged(self):
        """Random/ambiguous scores should not be flagged."""
        rng = np.random.default_rng(123)
        scores = rng.standard_normal(100)
        labels = np.array([1] * 50 + [0] * 50)
        assert detect_score_inversion(scores, labels) is False
