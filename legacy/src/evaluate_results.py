"""
evaluate_results.py — Compute classification metrics and generate visualizations.

Reads results/hhem_predictions.csv, computes accuracy/precision/recall/F1,
confusion matrix, score distribution, and precision-recall curve.
Saves metrics to results/metrics.json and figures to figures/.
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Compute all classification metrics."""
    metrics = {
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
    try:
        # For ROC AUC: HHEM scores consistency, so hallucination prob = 1 - score
        halluc_scores = 1.0 - scores
        metrics["roc_auc"] = float(roc_auc_score(y_true, halluc_scores))
    except Exception as e:
        logger.warning(f"ROC AUC computation failed: {e}")
        metrics["roc_auc"] = None
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Faithful", "Hallucinated"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — HHEM on HaluEval QA", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_score_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plot overlapping score histograms by true label."""
    fig, ax = plt.subplots(figsize=(8, 5))

    faithful_scores = df[df["label"] == 0]["hhem_score"].values
    halluc_scores = df[df["label"] == 1]["hhem_score"].values

    ax.hist(faithful_scores, bins=40, alpha=0.6, color="steelblue", label="Faithful (label=0)", density=True)
    ax.hist(halluc_scores, bins=40, alpha=0.6, color="tomato", label="Hallucinated (label=1)", density=True)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.5")

    ax.set_xlabel("HHEM Consistency Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("HHEM Score Distribution by True Label", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved score distribution to {output_path}")


def plot_precision_recall_curve(y_true: np.ndarray, scores: np.ndarray, output_path: str) -> None:
    """Plot and save precision-recall curve."""
    # Scores are consistency scores; invert for hallucination detection
    halluc_probs = 1.0 - scores
    precision, recall, thresholds = precision_recall_curve(y_true, halluc_probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="darkorange", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.15, color="darkorange")

    # Mark the operating point at threshold=0.5 (i.e., halluc_prob=0.5)
    idx = np.argmin(np.abs(thresholds - 0.5))
    ax.scatter(recall[idx], precision[idx], color="red", s=80, zorder=5, label=f"Threshold=0.5\nP={precision[idx]:.2f}, R={recall[idx]:.2f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Hallucination Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved precision-recall curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HHEM predictions and generate figures.")
    parser.add_argument("--input", type=str, default="results/hhem_predictions.csv")
    parser.add_argument("--metrics_out", type=str, default="results/metrics.json")
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows.")

    y_true = df["label"].values
    y_pred = df["predicted_label"].values
    scores = df["hhem_score"].values

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, scores)

    logger.info("=== METRICS ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=["Faithful", "Hallucinated"])
    logger.info("\n" + report)
    metrics["classification_report"] = report

    # Save metrics
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {args.metrics_out}")

    # Generate figures
    logger.info("Generating figures...")
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.figures_dir, "confusion_matrix.png"))
    plot_score_distribution(df, os.path.join(args.figures_dir, "score_distribution.png"))
    plot_precision_recall_curve(y_true, scores, os.path.join(args.figures_dir, "precision_recall_curve.png"))

    logger.info("evaluate_results.py complete.")


if __name__ == "__main__":
    main()
