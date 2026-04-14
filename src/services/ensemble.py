"""Ensemble methods — combine multiple model predictions."""

import logging
from typing import Dict, List

import numpy as np

from src.services.metrics import compute_metrics

logger = logging.getLogger(__name__)


def ensemble_average(scores_list: List[np.ndarray]) -> np.ndarray:
    """Simple average of hallucination probability scores."""
    return np.mean(scores_list, axis=0)


def ensemble_weighted(scores_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Weighted average of hallucination probability scores."""
    weights = np.array(weights)
    weights = weights / weights.sum()
    return sum(s * w for s, w in zip(scores_list, weights))


def ensemble_agreement(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> np.ndarray:
    """Agreement ensemble: when models agree use their output, when disagree use higher confidence."""
    return np.where(
        preds_a == preds_b,
        preds_a,
        np.where(np.abs(probs_a - 0.5) > np.abs(probs_b - 0.5), preds_a, preds_b)
    )


def evaluate_all_models(
    y_true: np.ndarray,
    models: Dict[str, dict],
) -> Dict[str, dict]:
    """Evaluate multiple models and return metrics for each.

    models: dict of {name: {"preds": array, "probs": array}}
    """
    all_metrics = {}
    for name, data in models.items():
        m = compute_metrics(y_true, data["preds"], data.get("scores"), label=name)
        all_metrics[name] = m
        logger.info(f"{name:<25} Acc={m['accuracy']:.4f} F1={m['f1_hallucinated']:.4f}")

    best = max(all_metrics, key=lambda k: all_metrics[k]["f1_hallucinated"])
    logger.info(f"Best by F1: {best} ({all_metrics[best]['f1_hallucinated']:.4f})")
    return all_metrics
