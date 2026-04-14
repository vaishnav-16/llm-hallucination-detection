"""Threshold optimization — sweep and cross-validated threshold selection."""

import logging
from collections import Counter
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

SEED = 42


def sweep_thresholds(
    scores: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray = None,
) -> List[dict]:
    """Sweep thresholds and compute metrics at each point.

    Scores are consistency scores: score <= threshold means hallucinated (label=1).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    results = []
    for t in thresholds:
        preds = (scores <= t).astype(int)
        prec = precision_score(true_labels, preds, pos_label=1, zero_division=0)
        rec = recall_score(true_labels, preds, pos_label=1, zero_division=0)
        f1 = f1_score(true_labels, preds, pos_label=1, zero_division=0)
        acc = accuracy_score(true_labels, preds)
        tp = ((preds == 1) & (true_labels == 1)).sum()
        tn = ((preds == 0) & (true_labels == 0)).sum()
        fp = ((preds == 1) & (true_labels == 0)).sum()
        fn = ((preds == 0) & (true_labels == 1)).sum()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        youden_j = sensitivity + specificity - 1
        results.append({
            "threshold": round(float(t), 3),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "youden_j": float(youden_j),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        })
    return results


def cv_optimize_threshold(
    scores: np.ndarray,
    true_labels: np.ndarray,
    n_folds: int = 5,
    thresholds: np.ndarray = None,
) -> dict:
    """Cross-validated threshold optimization.

    Returns dict with cv_threshold, per-fold results, and aggregate stats.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []
    fold_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(scores, true_labels)):
        train_scores = scores[train_idx]
        train_labels = true_labels[train_idx]
        val_scores = scores[val_idx]
        val_labels = true_labels[val_idx]

        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (train_scores <= t).astype(int)
            f1 = f1_score(train_labels, preds, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        val_preds = (val_scores <= best_t).astype(int)
        fold_results.append({
            "fold": fold_idx + 1,
            "optimal_threshold": best_t,
            "train_f1": float(best_f1),
            "val_f1": float(f1_score(val_labels, val_preds, pos_label=1, zero_division=0)),
            "val_accuracy": float(accuracy_score(val_labels, val_preds)),
            "val_precision": float(precision_score(val_labels, val_preds, pos_label=1, zero_division=0)),
            "val_recall": float(recall_score(val_labels, val_preds, pos_label=1, zero_division=0)),
            "val_n": len(val_idx),
        })
        fold_thresholds.append(best_t)
        logger.info(f"  Fold {fold_idx+1}: threshold={best_t:.2f}, train_F1={best_f1:.4f}, "
                     f"val_F1={fold_results[-1]['val_f1']:.4f}")

    cv_f1s = [r["val_f1"] for r in fold_results]
    thresh_counts = Counter(fold_thresholds)
    cv_threshold = thresh_counts.most_common(1)[0][0]

    # Evaluate on full dataset
    cv_preds = (scores <= cv_threshold).astype(int)

    return {
        "cv_optimal_threshold": cv_threshold,
        "cv_mean_f1": float(np.mean(cv_f1s)),
        "cv_std_f1": float(np.std(cv_f1s)),
        "cv_mean_threshold": float(np.mean(fold_thresholds)),
        "cv_majority_threshold": cv_threshold,
        "cv_full_dataset_f1": float(f1_score(true_labels, cv_preds, pos_label=1, zero_division=0)),
        "per_fold_results": fold_results,
    }
