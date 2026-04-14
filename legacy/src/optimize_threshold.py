"""
optimize_threshold.py — Cross-validated threshold optimization for HHEM.

Uses 5-fold stratified CV to find optimal threshold without overfitting.
For each fold: optimize threshold on 4 training folds, evaluate on held-out fold.
Reports mean F1 +/- std across folds.
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42


def sweep_thresholds(scores: np.ndarray, true_labels: np.ndarray, thresholds: np.ndarray):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/hhem_predictions.csv")
    parser.add_argument("--output_metrics", default="results/optimal_threshold_metrics.json")
    parser.add_argument("--output_preds", default="results/hhem_predictions_optimized.csv")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    scores = df["hhem_score"].values
    true_labels = df["label"].values
    logger.info(f"Loaded {len(df)} predictions. Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    thresholds = np.arange(0.05, 0.96, 0.01)

    # ===== NAIVE (single-split) optimization for reference =====
    logger.info(f"Sweeping {len(thresholds)} thresholds (naive, full dataset)...")
    naive_results = sweep_thresholds(scores, true_labels, thresholds)
    df_naive = pd.DataFrame(naive_results)
    naive_best_f1_row = df_naive.loc[df_naive["f1"].idxmax()]
    naive_best_youden_row = df_naive.loc[df_naive["youden_j"].idxmax()]
    naive_best_acc_row = df_naive.loc[df_naive["accuracy"].idxmax()]

    logger.info(f"Naive best F1: threshold={naive_best_f1_row['threshold']:.2f}, F1={naive_best_f1_row['f1']:.4f}")

    # ===== CROSS-VALIDATED threshold optimization =====
    logger.info(f"\n5-fold cross-validated threshold optimization...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    fold_results = []
    fold_optimal_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(scores, true_labels)):
        train_scores = scores[train_idx]
        train_labels = true_labels[train_idx]
        val_scores = scores[val_idx]
        val_labels = true_labels[val_idx]

        # Find optimal threshold on TRAINING folds
        best_t = 0.5
        best_train_f1 = 0.0
        for t in thresholds:
            preds = (train_scores <= t).astype(int)
            f1 = f1_score(train_labels, preds, pos_label=1, zero_division=0)
            if f1 > best_train_f1:
                best_train_f1 = f1
                best_t = float(t)

        # Evaluate on HELD-OUT fold with threshold selected on training folds
        val_preds = (val_scores <= best_t).astype(int)
        val_f1 = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, pos_label=1, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, pos_label=1, zero_division=0)

        fold_result = {
            "fold": fold_idx + 1,
            "optimal_threshold": best_t,
            "train_f1": float(best_train_f1),
            "val_f1": float(val_f1),
            "val_accuracy": float(val_acc),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "val_n": len(val_idx),
        }
        fold_results.append(fold_result)
        fold_optimal_thresholds.append(best_t)

        logger.info(f"  Fold {fold_idx+1}: threshold={best_t:.2f}, train_F1={best_train_f1:.4f}, val_F1={val_f1:.4f}")

    # Aggregate CV results
    cv_f1s = [r["val_f1"] for r in fold_results]
    cv_mean_f1 = float(np.mean(cv_f1s))
    cv_std_f1 = float(np.std(cv_f1s))

    # Majority threshold: most frequently selected across folds
    from collections import Counter
    thresh_counts = Counter(fold_optimal_thresholds)
    cv_majority_threshold = thresh_counts.most_common(1)[0][0]
    cv_mean_threshold = float(np.mean(fold_optimal_thresholds))

    logger.info(f"\nCV Results ({args.n_folds}-fold):")
    logger.info(f"  Mean F1: {cv_mean_f1:.4f} +/- {cv_std_f1:.4f}")
    logger.info(f"  Threshold votes: {dict(thresh_counts)}")
    logger.info(f"  Majority threshold: {cv_majority_threshold:.2f}")
    logger.info(f"  Mean threshold: {cv_mean_threshold:.2f}")

    # Use majority threshold as the CV-optimized threshold
    cv_threshold = cv_majority_threshold

    # Evaluate CV threshold on full dataset for reference
    cv_full_preds = (scores <= cv_threshold).astype(int)
    cv_full_f1 = f1_score(true_labels, cv_full_preds, pos_label=1, zero_division=0)
    cv_full_acc = accuracy_score(true_labels, cv_full_preds)
    cv_full_prec = precision_score(true_labels, cv_full_preds, pos_label=1, zero_division=0)
    cv_full_rec = recall_score(true_labels, cv_full_preds, pos_label=1, zero_division=0)

    logger.info(f"  CV threshold ({cv_threshold:.2f}) on full data: F1={cv_full_f1:.4f}, Acc={cv_full_acc:.4f}")

    # ===== FIGURE =====
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    ax = axes[0]
    ax.plot(df_naive["threshold"], df_naive["f1"], label="F1", color="blue", linewidth=2)
    ax.plot(df_naive["threshold"], df_naive["precision"], label="Precision", color="green", linewidth=1.5, linestyle="--")
    ax.plot(df_naive["threshold"], df_naive["recall"], label="Recall", color="red", linewidth=1.5, linestyle="--")
    ax.plot(df_naive["threshold"], df_naive["accuracy"], label="Accuracy", color="purple", linewidth=1.5, linestyle=":")
    ax.axvline(cv_threshold, color="blue", alpha=0.6, linestyle=":", linewidth=2,
               label=f"CV-optimal threshold = {cv_threshold:.2f}")
    ax.axvline(0.5, color="gray", alpha=0.5, linestyle="--", label="Baseline = 0.5")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Metrics vs Decision Threshold (HHEM)\n"
                 f"5-fold CV: F1 = {cv_mean_f1:.4f} +/- {cv_std_f1:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0.05, 0.95])

    ax = axes[1]
    ax.plot(df_naive["threshold"], df_naive["youden_j"], label="Youden's J", color="darkorange", linewidth=2)
    ax.axvline(cv_threshold, color="darkorange", alpha=0.6, linestyle=":", linewidth=2,
               label=f"CV threshold = {cv_threshold:.2f}")
    ax.axvline(0.5, color="gray", alpha=0.5, linestyle="--", label="Baseline = 0.5")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Youden's J Statistic", fontsize=12)
    ax.set_title("Youden's J Statistic vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0.05, 0.95])

    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "threshold_optimization.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold_optimization.png")

    # ===== SAVE METRICS =====
    output = {
        "cv_optimal_threshold": cv_threshold,
        "cv_mean_f1": cv_mean_f1,
        "cv_std_f1": cv_std_f1,
        "cv_mean_threshold": cv_mean_threshold,
        "cv_majority_threshold": cv_majority_threshold,
        "cv_full_dataset_f1": cv_full_f1,
        "per_fold_results": fold_results,
        "naive_optimal": {
            "best_f1": naive_best_f1_row.to_dict(),
            "best_youden": naive_best_youden_row.to_dict(),
            "best_accuracy": naive_best_acc_row.to_dict(),
        },
        # Keep backward-compatible keys
        "best_f1": {
            "threshold": cv_threshold,
            "accuracy": cv_full_acc,
            "precision": cv_full_prec,
            "recall": cv_full_rec,
            "f1": cv_full_f1,
        },
        "baseline_threshold": 0.5,
        "all_results": naive_results,
    }
    with open(args.output_metrics, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {args.output_metrics}")

    # Re-predict with CV-optimized threshold
    df["predicted_label_optimized"] = (df["hhem_score"] <= cv_threshold).astype(int)
    df["optimal_threshold"] = cv_threshold
    df.to_csv(args.output_preds, index=False)
    logger.info(f"Saved optimized predictions to {args.output_preds}")

    logger.info(f"\nNaive threshold: {naive_best_f1_row['threshold']:.2f} (F1={naive_best_f1_row['f1']:.4f})")
    logger.info(f"CV threshold:    {cv_threshold:.2f} (mean F1={cv_mean_f1:.4f} +/- {cv_std_f1:.4f})")
    logger.info("optimize_threshold.py complete.")


if __name__ == "__main__":
    main()
