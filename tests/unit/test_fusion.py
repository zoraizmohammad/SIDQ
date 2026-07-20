"""Tests for fusion normalization and optimization."""

import numpy as np

from sidq.fusion.normalization import fit_normalization, normalize_scores
from sidq.fusion.optimize import (
    compute_correlation_matrix,
    logistic_fusion,
    optimize_weights,
    uniform_fusion,
)


class TestNormalization:
    def test_zscore(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = fit_normalization(scores, method="zscore")
        normed = normalize_scores(scores, stats)
        assert abs(normed.mean()) < 1e-6
        assert abs(normed.std() - 1.0) < 0.1

    def test_robust(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = fit_normalization(scores, method="robust")
        normed = normalize_scores(scores, stats)
        assert normed[2] == 0.0  # Median centered

    def test_none(self):
        scores = np.array([1.0, 2.0, 3.0])
        stats = fit_normalization(scores, method="none")
        stats.method = "none"
        normed = normalize_scores(scores, stats)
        assert np.allclose(normed, scores)


class TestFusion:
    def test_uniform(self):
        scores = {
            "model_a": np.array([1.0, 2.0, 3.0]),
            "model_b": np.array([3.0, 2.0, 1.0]),
        }
        fused = uniform_fusion(scores)
        assert np.allclose(fused, [2.0, 2.0, 2.0])

    def test_optimize_weights_complementary(self):
        """Complementary models should get meaningful weights."""
        rng = np.random.default_rng(42)
        n = 200
        labels = np.array([1] * 100 + [0] * 100)

        # Model A: good at detecting bonafide
        scores_a = np.where(labels == 1, rng.normal(2, 0.5, n), rng.normal(-1, 1, n))
        # Model B: different error pattern
        scores_b = np.where(labels == 1, rng.normal(1.5, 0.8, n), rng.normal(-2, 0.5, n))

        weights = optimize_weights({"a": scores_a, "b": scores_b}, labels)
        assert weights["a"] > 0
        assert weights["b"] > 0
        assert abs(weights["a"] + weights["b"] - 1.0) < 1e-4

    def test_logistic_fusion(self):
        rng = np.random.default_rng(42)
        n = 200
        labels = np.array([1] * 100 + [0] * 100)
        scores = {
            "m1": np.where(labels == 1, rng.normal(2, 1, n), rng.normal(-1, 1, n)),
            "m2": np.where(labels == 1, rng.normal(1.5, 1, n), rng.normal(-1.5, 1, n)),
        }
        fused, coeffs = logistic_fusion(scores, labels)
        assert fused.shape == (n,)
        assert len(coeffs) == 2

    def test_correlation_matrix(self):
        scores = {
            "a": np.array([1.0, 2.0, 3.0, 4.0]),
            "b": np.array([1.0, 2.0, 3.0, 4.0]),
        }
        corr = compute_correlation_matrix(scores)
        assert abs(corr["a"]["b"] - 1.0) < 1e-6
