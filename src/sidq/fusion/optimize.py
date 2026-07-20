"""Fusion weight optimization."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from sidq.evaluation.eer import compute_eer
from sidq.fusion.normalization import fit_normalization, normalize_scores


def uniform_fusion(model_scores: dict[str, np.ndarray]) -> np.ndarray:
    """Simple uniform average of all model scores."""
    arrays = list(model_scores.values())
    return np.mean(arrays, axis=0)


def weighted_fusion(
    model_scores: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted average fusion."""
    result = np.zeros_like(next(iter(model_scores.values())))
    total_weight = sum(weights.values())
    for name, scores in model_scores.items():
        w = weights.get(name, 0.0)
        result += scores * (w / total_weight)
    return result


def optimize_weights(
    model_scores: dict[str, np.ndarray],
    labels: np.ndarray,
    normalize: str = "zscore",
) -> dict[str, float]:
    """Optimize fusion weights to minimize EER on validation data.

    Uses constrained optimization with non-negative weights summing to 1.
    """
    model_names = list(model_scores.keys())
    n_models = len(model_names)

    # Normalize each model's scores
    norm_scores: dict[str, np.ndarray] = {}
    for name, scores in model_scores.items():
        stats = fit_normalization(scores, method=normalize)
        norm_scores[name] = normalize_scores(scores, stats)

    score_matrix = np.column_stack([norm_scores[name] for name in model_names])

    def objective(w):
        fused = score_matrix @ w
        result = compute_eer(fused, labels)
        return result.eer

    # Start with uniform weights
    x0 = np.ones(n_models) / n_models

    # Constraints: weights sum to 1, each >= 0
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    optimal_weights = result.x
    return {name: float(w) for name, w in zip(model_names, optimal_weights, strict=True)}


def logistic_fusion(
    model_scores: dict[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Logistic regression fusion.

    Returns fused scores and learned coefficients.
    """
    model_names = list(model_scores.keys())
    score_matrix = np.column_stack([model_scores[name] for name in model_names])

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(score_matrix, labels)

    # Fused score = probability of bonafide class
    fused = clf.predict_proba(score_matrix)[:, 1]

    coefficients = {name: float(c) for name, c in zip(model_names, clf.coef_[0], strict=True)}
    return fused, coefficients


def compute_correlation_matrix(model_scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Compute pairwise score correlation between models."""
    names = list(model_scores.keys())
    result: dict[str, dict[str, float]] = {}
    for name_i in names:
        result[name_i] = {}
        for name_j in names:
            corr = np.corrcoef(model_scores[name_i], model_scores[name_j])[0, 1]
            result[name_i][name_j] = float(corr)
    return result
