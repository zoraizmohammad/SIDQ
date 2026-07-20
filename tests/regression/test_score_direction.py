"""Regression tests for score direction — must never submit inverted scores."""

import numpy as np
import pytest

from sidq.evaluation.metrics import compute_all_metrics, validate_score_direction


class TestScoreDirection:
    """These tests exist to prevent the most catastrophic competition mistake:
    submitting scores where higher means spoofed instead of bonafide."""

    def test_correct_direction_passes(self):
        scores = np.array([3.0, 2.5, 2.0, -1.0, -2.0, -3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert validate_score_direction(scores, labels)

    def test_inverted_direction_raises(self):
        scores = np.array([-3.0, -2.5, -2.0, 1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        with pytest.raises(ValueError, match="SCORE DIRECTION ERROR"):
            validate_score_direction(scores, labels)

    def test_inverted_direction_silent(self):
        scores = np.array([-3.0, -2.5, -2.0, 1.0, 2.0, 3.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        result = validate_score_direction(scores, labels, fail_loudly=False)
        assert not result

    def test_perfect_classifier_has_near_zero_eer(self):
        """A well-separated correct-direction classifier must have low EER."""
        scores = np.array([10.0, 9.0, 8.0, 7.0, -5.0, -6.0, -7.0, -8.0])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        metrics = compute_all_metrics(scores, labels)
        assert metrics["eer"] < 0.01
        assert metrics["roc_auc"] > 0.99

    def test_inverted_classifier_has_high_eer(self):
        """An inverted classifier will show high EER, alerting to the problem."""
        scores = np.array([-10.0, -9.0, -8.0, -7.0, 5.0, 6.0, 7.0, 8.0])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        metrics = compute_all_metrics(scores, labels)
        assert metrics["eer"] > 0.9

    def test_label_mapping_consistency(self):
        """Verify label mapping: 1=bonafide, 0=spoofed."""
        from sidq.constants import LABEL_BONAFIDE, LABEL_SPOOFED

        assert LABEL_BONAFIDE == 1
        assert LABEL_SPOOFED == 0
