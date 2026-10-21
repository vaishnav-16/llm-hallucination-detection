"""Decision thresholds and calibration, always fitted on a validation slice and never on the test slice."""

from __future__ import annotations

import numpy as np

from src.eval.metrics import threshold_metrics


def choose_threshold(y_val: np.ndarray, p_val: np.ndarray, criterion: str = "f1") -> float:
    """Threshold on validation scores that maximises ``criterion`` (f1 or balanced_accuracy)."""
    p_val = np.asarray(p_val, float)
    cands = np.unique(np.quantile(p_val, np.linspace(0, 1, 201)))
    best_t, best = 0.5, -1.0
    for t in cands:
        v = threshold_metrics(y_val, p_val, float(t))[criterion]
        if v > best + 1e-12:
            best_t, best = float(t), v
    return best_t


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(y_val: np.ndarray, p_val: np.ndarray, iters: int = 500, lr: float = 0.1) -> tuple[float, float]:
    """Platt scaling on the logit of p: p' = sigmoid(a * logit(p) + b). Returns (a, b)."""
    x, y = _logit(p_val), np.asarray(y_val, float)
    a, b = 1.0, 0.0
    for _ in range(iters):  # plain gradient descent on NLL; 2 parameters, convex
        z = 1 / (1 + np.exp(-(a * x + b)))
        ga, gb = np.mean((z - y) * x), np.mean(z - y)
        a, b = a - lr * ga, b - lr * gb
    return float(a), float(b)


def apply_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))
