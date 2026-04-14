"""Static figure generation — all matplotlib/seaborn plot functions."""

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score

logger = logging.getLogger(__name__)

# Consistent color palette
COLORS = {
    "faithful": "steelblue",
    "hallucinated": "tomato",
    "model_a": "steelblue",
    "model_b": "darkorange",
    "model_c": "seagreen",
    "model_d": "purple",
    "answer": "seagreen",
    "caveat": "gold",
    "abstain": "tomato",
}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Faithful", "Hallucinated"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_score_distribution(
    df: pd.DataFrame,
    score_col: str,
    output_path: str,
    title: str = "Score Distribution by True Label",
    threshold: float = 0.5,
) -> None:
    """Plot overlapping score histograms by true label."""
    fig, ax = plt.subplots(figsize=(8, 5))

    faithful = df[df["label"] == 0][score_col].values
    halluc = df[df["label"] == 1][score_col].values

    ax.hist(faithful, bins=40, alpha=0.6, color=COLORS["faithful"], label="Faithful (label=0)", density=True)
    ax.hist(halluc, bins=40, alpha=0.6, color=COLORS["hallucinated"], label="Hallucinated (label=1)", density=True)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

    ax.set_xlabel("Consistency Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score distribution to {output_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: str,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot precision-recall curve for hallucination detection."""
    halluc_probs = 1.0 - scores
    precision, recall, thresholds = precision_recall_curve(y_true, halluc_probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="darkorange", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.15, color="darkorange")

    idx = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter(recall[idx], precision[idx], color="red", s=80, zorder=5,
               label=f"Threshold=0.5\nP={precision[idx]:.2f}, R={recall[idx]:.2f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved precision-recall curve to {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    models: Dict[str, np.ndarray],
    output_path: str,
    title: str = "ROC Curves",
) -> None:
    """Plot ROC curves for multiple models.

    models: dict of {name: hallucination_probabilities}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [COLORS["model_a"], COLORS["model_b"], COLORS["model_c"], COLORS["model_d"]]

    for (name, probs), color in zip(models.items(), colors):
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves to {output_path}")


def plot_model_comparison_bar(
    metrics: Dict[str, dict],
    output_path: str,
    title: str = "Model Comparison",
) -> None:
    """Plot bar chart comparing model metrics."""
    model_names = list(metrics.keys())
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision_hallucinated", "Precision"),
        ("recall_hallucinated", "Recall"),
        ("f1_hallucinated", "F1"),
    ]

    for ax, (key, label) in zip(axes, metric_keys):
        # Try both key formats
        vals = []
        for m in model_names:
            v = metrics[m].get(key, metrics[m].get(key.replace("_hallucinated", ""), 0))
            vals.append(v)
        bars = ax.bar(range(len(model_names)), vals, color=colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace("(", "\n(") for n in model_names], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(label, fontsize=12, fontweight="bold")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison to {output_path}")


def plot_threshold_sweep(
    sweep_results: List[dict],
    cv_threshold: float,
    cv_mean_f1: float,
    cv_std_f1: float,
    output_path: str,
) -> None:
    """Plot threshold optimization figure with metrics and Youden's J."""
    df = pd.DataFrame(sweep_results)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    ax = axes[0]
    ax.plot(df["threshold"], df["f1"], label="F1", color="blue", linewidth=2)
    ax.plot(df["threshold"], df["precision"], label="Precision", color="green", linewidth=1.5, linestyle="--")
    ax.plot(df["threshold"], df["recall"], label="Recall", color="red", linewidth=1.5, linestyle="--")
    ax.plot(df["threshold"], df["accuracy"], label="Accuracy", color="purple", linewidth=1.5, linestyle=":")
    ax.axvline(cv_threshold, color="blue", alpha=0.6, linestyle=":", linewidth=2,
               label=f"CV-optimal = {cv_threshold:.2f}")
    ax.axvline(0.5, color="gray", alpha=0.5, linestyle="--", label="Baseline = 0.5")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Metrics vs Threshold (5-fold CV: F1={cv_mean_f1:.4f} +/- {cv_std_f1:.4f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0.05, 0.95])

    ax = axes[1]
    ax.plot(df["threshold"], df["youden_j"], label="Youden's J", color="darkorange", linewidth=2)
    ax.axvline(cv_threshold, color="darkorange", alpha=0.6, linestyle=":", linewidth=2,
               label=f"CV threshold = {cv_threshold:.2f}")
    ax.axvline(0.5, color="gray", alpha=0.5, linestyle="--", label="Baseline = 0.5")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Youden's J Statistic", fontsize=12)
    ax.set_title("Youden's J vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0.05, 0.95])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved threshold optimization to {output_path}")


def plot_decision_policy(
    results: dict,
    output_path: str,
) -> None:
    """Plot stacked bar chart for 3-tier decision policy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    tol_labels = list(results.keys())
    pct_answer = [results[t]["pct_answer"] for t in tol_labels]
    pct_caveat = [results[t]["pct_caveat"] for t in tol_labels]
    pct_abstain = [results[t]["pct_abstain"] for t in tol_labels]
    x = np.arange(len(tol_labels))

    bars1 = ax.bar(x, pct_answer, label="ANSWER", color=COLORS["answer"], alpha=0.85)
    bars2 = ax.bar(x, pct_caveat, bottom=pct_answer, label="CAVEAT", color=COLORS["caveat"], alpha=0.85)
    bars3 = ax.bar(x, pct_abstain, bottom=np.array(pct_answer)+np.array(pct_caveat),
                   label="ABSTAIN", color=COLORS["abstain"], alpha=0.85)

    for bar, val in zip(bars1, pct_answer):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.0%}", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.split("(")[0].strip() for t in tol_labels], fontsize=10)
    ax.set_ylabel("Fraction of Examples", fontsize=12)
    ax.set_title("3-Tier Decision Policy: Answer / Caveat / Abstain", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved decision policy to {output_path}")


def plot_error_overlap(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: str,
) -> None:
    """Plot model agreement heatmap."""
    agree_matrix = np.zeros((2, 2), dtype=int)
    for h, f in zip(preds_a, preds_b):
        agree_matrix[h][f] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(agree_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"{label_b}: Faithful", f"{label_b}: Hallucinated"],
                yticklabels=[f"{label_a}: Faithful", f"{label_a}: Hallucinated"])
    ax.set_title(f"{label_a} vs {label_b}\nPrediction Agreement", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved error overlap to {output_path}")


def plot_reliability_diagram(
    y_true: np.ndarray,
    probs: np.ndarray,
    output_path: str,
    title: str = "Calibration",
    n_bins: int = 10,
) -> None:
    """Plot reliability diagram (calibration curve)."""
    bins = np.linspace(0, 1, n_bins + 1)
    mean_pred, mean_actual = [], []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        mean_pred.append(probs[mask].mean())
        mean_actual.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(mean_pred, mean_actual, color="darkorange", linewidth=2, label="Model calibration")
    ax.scatter(mean_pred, mean_actual, s=80, zorder=5, color="darkorange")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of hallucinated", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram to {output_path}")


def plot_coverage_precision_tradeoff(
    y_true: np.ndarray,
    probs: np.ndarray,
    policy_results: dict,
    output_path: str,
) -> None:
    """Plot coverage vs precision trade-off curve."""
    thresholds = np.arange(0.05, 0.96, 0.02)
    coverages, precisions = [], []
    for t in thresholds:
        served = probs <= t
        if served.sum() == 0:
            continue
        coverages.append(served.mean())
        precisions.append(1.0 - y_true[served].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, precisions, color="steelblue", linewidth=2)
    ax.fill_between(coverages, precisions, alpha=0.15, color="steelblue")

    for tol_name, m in policy_results.items():
        ax.scatter([m["coverage"]], [m["precision_of_answers"]], s=100, zorder=5,
                   label=f"{tol_name.split('(')[0].strip()}: cov={m['coverage']:.0%}")

    ax.set_xlabel("Coverage (fraction served)", fontsize=12)
    ax.set_ylabel("Precision (1 - halluc rate)", fontsize=12)
    ax.set_title("Coverage-Precision Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved coverage-precision tradeoff to {output_path}")
