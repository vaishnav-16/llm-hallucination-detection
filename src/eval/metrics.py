"""Classification metrics on ``p_halluc`` with pair-clustered bootstrap CIs, McNemar tests and ECE.

Rows sharing a ``pair_id`` (the two answers to one question, or all responses to one RAGTruth source)
are not independent, so confidence intervals resample whole pairs (cluster bootstrap).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from math import comb


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties get average ranks)."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = np.asarray(p)[order]
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(sp, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=ranks[order])
        ranks[order] = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision for the hallucinated class."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    return float((precision * ys).sum() / ys.sum())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(y)),
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
    }


def all_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    m = threshold_metrics(y, p, threshold)
    m.update({"auroc": auroc(y, p), "auprc": auprc(y, p), "ece": ece(y, p)})
    return m


METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray, float], float]] = {
    "auroc": lambda y, p, t: auroc(y, p),
    "auprc": lambda y, p, t: auprc(y, p),
    "f1": lambda y, p, t: threshold_metrics(y, p, t)["f1"],
    "balanced_accuracy": lambda y, p, t: threshold_metrics(y, p, t)["balanced_accuracy"],
    "accuracy": lambda y, p, t: threshold_metrics(y, p, t)["accuracy"],
}


def cluster_bootstrap(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, threshold: float, metrics: list[str],
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling whole groups (pairs) with replacement."""
    y, p = np.asarray(y), np.asarray(p)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    samples = {m: np.empty(n_boot) for m in metrics}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        for m in metrics:
            samples[m][b] = METRIC_FNS[m](y[idx], p[idx], threshold)
    return {m: (float(np.nanpercentile(s, 100 * alpha / 2)), float(np.nanpercentile(s, 100 * (1 - alpha / 2)))) for m, s in samples.items()}


def paired_bootstrap_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray, metric: str = "auroc",
    t_a: float = 0.5, t_b: float = 0.5, n_boot: int = 1000, seed: int = 0,
) -> dict[str, float]:
    """CI and two-sided bootstrap p-value for metric(a) - metric(b) on the same rows."""
    fn = METRIC_FNS[metric]
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        diffs[b] = fn(y[idx], p_a[idx], t_a) - fn(y[idx], p_b[idx], t_b)
    point = fn(y, p_a, t_a) - fn(y, p_b, t_b)
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_lo": float(np.percentile(diffs, 2.5)), "ci_hi": float(np.percentile(diffs, 97.5)), "p_value": float(min(1.0, p_two))}


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired binary predictions."""
    y = np.asarray(y).astype(int)
    ca, cb = np.asarray(pred_a) == y, np.asarray(pred_b) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}
