"""Metrics computation — single source of truth for all classification metrics."""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    label: str = "Model",
) -> dict:
    """Compute all classification metrics for hallucination detection.

    Args:
        y_true: Ground truth labels (0=faithful, 1=hallucinated).
        y_pred: Predicted labels.
        scores: Raw consistency scores (higher = more faithful). Used for ROC AUC.
        label: Model name for display.

    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "label": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_hallucinated": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_hallucinated": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_hallucinated": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "total_samples": int(len(y_true)),
        "num_hallucinated": int(np.sum(y_true == 1)),
        "num_faithful": int(np.sum(y_true == 0)),
        "predicted_hallucinated": int(np.sum(y_pred == 1)),
        "predicted_faithful": int(np.sum(y_pred == 0)),
    }

    if scores is not None:
        try:
            halluc_scores = 1.0 - scores
            metrics["roc_auc"] = float(roc_auc_score(y_true, halluc_scores))
        except Exception as e:
            logger.warning(f"ROC AUC computation failed: {e}")
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    label: str = "Model",
) -> dict:
    """Compute metrics at a specific threshold.

    Scores are consistency scores: score > threshold means faithful (label=0).
    """
    y_pred = np.where(scores > threshold, 0, 1)
    return compute_metrics(y_true, y_pred, scores, label=label)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return sklearn classification report as a string."""
    return classification_report(y_true, y_pred, target_names=["Faithful", "Hallucinated"])
