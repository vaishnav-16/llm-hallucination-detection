"""
ensemble_and_analysis.py — Ensemble HHEM + fine-tuned DeBERTa, comprehensive analysis.

Generates model comparison figures, ROC curves, error overlap, feature importance.
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
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_prob=None, label="Model"):
    m = {
        "label": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["roc_auc"] = None
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hhem_preds", default="results/hhem_predictions.csv")
    parser.add_argument("--hhem_opt_preds", default="results/hhem_predictions_optimized.csv")
    parser.add_argument("--ft_preds", default="results/finetuned_predictions.csv")
    parser.add_argument("--enriched", default="data/halueval_enriched.csv")
    parser.add_argument("--output_ensemble", default="results/ensemble_metrics.json")
    parser.add_argument("--output_errors", default="results/comprehensive_error_analysis.csv")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # Load all predictions
    hhem_df = pd.read_csv(args.hhem_preds)
    y_true = hhem_df["label"].values
    hhem_scores = hhem_df["hhem_score"].values
    hhem_probs = 1.0 - hhem_scores  # hallucination probability
    hhem_preds = hhem_df["predicted_label"].values

    # Optimized HHEM
    hhem_opt_preds = hhem_preds
    if os.path.exists(args.hhem_opt_preds):
        hhem_opt_df = pd.read_csv(args.hhem_opt_preds)
        if "predicted_label_optimized" in hhem_opt_df.columns:
            hhem_opt_preds = hhem_opt_df["predicted_label_optimized"].values

    # Fine-tuned
    ft_probs = hhem_probs  # fallback
    ft_preds = hhem_preds
    ft_available = False
    if os.path.exists(args.ft_preds):
        try:
            ft_df = pd.read_csv(args.ft_preds)
            if "predicted_prob" in ft_df.columns:
                ft_probs = ft_df["predicted_prob"].values
                ft_preds = ft_df["predicted_label"].values
                ft_available = True
        except Exception:
            pass

    logger.info(f"Fine-tuned model available: {ft_available}")

    # ---- ENSEMBLE ----
    logger.info("\n=== ENSEMBLE ANALYSIS ===")

    # Simple average
    avg_probs = (hhem_probs + ft_probs) / 2
    avg_preds = (avg_probs >= 0.5).astype(int)

    # Weighted average (0.3 HHEM, 0.7 DeBERTa)
    weighted_probs = 0.3 * hhem_probs + 0.7 * ft_probs
    weighted_preds = (weighted_probs >= 0.5).astype(int)

    # Agreement ensemble: when models agree use their output; when disagree use higher confidence
    agree_preds = np.where(
        hhem_preds == ft_preds,
        hhem_preds,
        np.where(np.abs(hhem_probs - 0.5) > np.abs(ft_probs - 0.5), hhem_preds, ft_preds)
    )

    all_models = {
        "HHEM (τ=0.50)": {"preds": hhem_preds, "probs": hhem_probs},
        "HHEM (τ=opt)": {"preds": hhem_opt_preds, "probs": hhem_probs},
        "Fine-tuned DeBERTa": {"preds": ft_preds, "probs": ft_probs},
        "Ensemble (avg)": {"preds": avg_preds, "probs": avg_probs},
        "Ensemble (0.3+0.7)": {"preds": weighted_preds, "probs": weighted_probs},
        "Ensemble (agree)": {"preds": agree_preds, "probs": avg_probs},
    }

    all_metrics = {}
    logger.info(f"\n{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ROC-AUC':>8}")
    logger.info("-" * 60)
    for name, data in all_models.items():
        m = compute_metrics(y_true, data["preds"], data["probs"], label=name)
        all_metrics[name] = m
        auc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"
        logger.info(f"{name:<25} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
                    f"{m['recall']:>6.4f} {m['f1']:>6.4f} {auc_str:>8}")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    logger.info(f"\nBest model by F1: {best_name} (F1={all_metrics[best_name]['f1']:.4f})")

    with open(args.output_ensemble, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ---- FIGURES ----

    # 1. Model comparison bar chart (4 subplots: acc, prec, rec, f1)
    display_models = ["HHEM (τ=0.50)", "HHEM (τ=opt)", "Fine-tuned DeBERTa", "Ensemble (avg)", "Ensemble (0.3+0.7)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(display_models)))
    for ax, metric, title in zip(axes, ["accuracy", "precision", "recall", "f1"],
                                  ["Accuracy", "Precision", "Recall", "F1"]):
        vals = [all_metrics[m][metric] for m in display_models]
        bars = ax.bar(range(len(display_models)), vals, color=colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)
        ax.set_xticks(range(len(display_models)))
        ax.set_xticklabels([m.replace("(", "\n(") for m in display_models], fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=12, fontweight="bold")
    plt.suptitle("Model Comparison: HaluEval QA (in-domain)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "model_comparison_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved model_comparison_bar.png")

    # 2. ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))
    roc_models = {
        "HHEM": hhem_probs,
        "Fine-tuned DeBERTa": ft_probs,
        "Ensemble (avg)": avg_probs,
        "Ensemble (0.3+0.7)": weighted_probs,
    }
    colors_roc = ["steelblue", "darkorange", "seagreen", "purple"]
    for (name, probs_), color in zip(roc_models.items(), colors_roc):
        try:
            fpr, tpr, _ = roc_curve(y_true, probs_)
            auc = roc_auc_score(y_true, probs_)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved roc_curves.png")

    # 3. Error overlap heatmap (HHEM vs DeBERTa agreement)
    agree_matrix = np.zeros((2, 2), dtype=int)
    for h, f in zip(hhem_preds, ft_preds):
        agree_matrix[h][f] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(agree_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["DeBERTa: Faithful", "DeBERTa: Hallucinated"],
                yticklabels=["HHEM: Faithful", "HHEM: Hallucinated"])
    ax.set_title("HHEM vs Fine-tuned DeBERTa\nPrediction Agreement", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "error_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved error_overlap.png")

    # 4. Feature importance: correlation of features with model errors
    enriched_available = False
    if os.path.exists(args.enriched):
        try:
            enriched_df = pd.read_csv(args.enriched)
            enriched_df = enriched_df.iloc[:len(hhem_df)].copy()
            enriched_df["hhem_error"] = (hhem_preds != y_true).astype(int)
            enriched_df["ft_error"] = (ft_preds != y_true).astype(int)
            enriched_df["any_error"] = ((hhem_preds != y_true) | (ft_preds != y_true)).astype(int)

            feature_cols = [c for c in [
                "context_length", "response_length", "question_length",
                "response_context_overlap", "response_context_substring_ratio",
                "lexical_diversity", "num_entities_response", "avg_word_length_response",
                "question_response_overlap"
            ] if c in enriched_df.columns]

            if feature_cols:
                corrs_hhem = enriched_df[feature_cols + ["hhem_error"]].corr()["hhem_error"].drop("hhem_error")
                corrs_ft = enriched_df[feature_cols + ["ft_error"]].corr()["ft_error"].drop("ft_error")

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                for ax, corrs, title in [(axes[0], corrs_hhem, "HHEM Error Correlations"),
                                          (axes[1], corrs_ft, "DeBERTa Error Correlations")]:
                    sorted_corrs = corrs.sort_values()
                    colors_bar = ["tomato" if v > 0 else "steelblue" for v in sorted_corrs]
                    ax.barh(range(len(sorted_corrs)), sorted_corrs.values, color=colors_bar, alpha=0.8)
                    ax.set_yticks(range(len(sorted_corrs)))
                    ax.set_yticklabels(sorted_corrs.index, fontsize=9)
                    ax.set_title(title, fontsize=12, fontweight="bold")
                    ax.axvline(0, color="black", linewidth=0.5)
                    ax.set_xlabel("Correlation with Error", fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(args.figures_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
                plt.close()
                logger.info("Saved feature_importance.png")
                enriched_available = True
        except Exception as e:
            logger.warning(f"Could not load enriched dataset: {e}")

    if not enriched_available:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Feature importance figure\n(requires enriched dataset)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        plt.savefig(os.path.join(args.figures_dir, "feature_importance.png"), dpi=150)
        plt.close()

    # 5. Score scatter: HHEM vs DeBERTa prob, colored by true label
    fig, ax = plt.subplots(figsize=(8, 7))
    colors_scatter = ["steelblue" if l == 0 else "tomato" for l in y_true]
    ax.scatter(hhem_probs, ft_probs, c=colors_scatter, alpha=0.4, s=20)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="HHEM threshold")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="DeBERTa threshold")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="Faithful (true=0)"),
        Patch(facecolor="tomato", alpha=0.7, label="Hallucinated (true=1)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel("HHEM Hallucination Probability (1-score)", fontsize=12)
    ax.set_ylabel("DeBERTa Hallucination Probability", fontsize=12)
    ax.set_title("HHEM vs DeBERTa Predictions\n(colored by true label)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "score_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved score_scatter.png")

    # ---- COMPREHENSIVE ERROR ANALYSIS ----
    logger.info("\n=== COMPREHENSIVE ERROR ANALYSIS ===")

    # Collect ALL errors from both models, categorised by error_type
    error_rows = []

    # HHEM errors
    hhem_fp_idx = np.where((hhem_preds == 1) & (y_true == 0))[0]
    hhem_fn_idx = np.where((hhem_preds == 0) & (y_true == 1))[0]
    logger.info(f"HHEM errors: {len(hhem_fp_idx)} FP, {len(hhem_fn_idx)} FN")

    # FT errors
    ft_fp_idx = np.where((ft_preds == 1) & (y_true == 0))[0]
    ft_fn_idx = np.where((ft_preds == 0) & (y_true == 1))[0]
    logger.info(f"FT errors: {len(ft_fp_idx)} FP, {len(ft_fn_idx)} FN")

    # Disagreements
    hhem_halluc_ft_faithful = (hhem_preds == 1) & (ft_preds == 0)
    hhem_faithful_ft_halluc = (hhem_preds == 0) & (ft_preds == 1)
    logger.info(f"Disagreements: HHEM=hal/FT=fai: {hhem_halluc_ft_faithful.sum()}, HHEM=fai/FT=hal: {hhem_faithful_ft_halluc.sum()}")

    def make_row(idx, error_type, model):
        return {
            "id": hhem_df.iloc[idx].get("id", idx),
            "question": str(hhem_df.iloc[idx]["question"])[:150],
            "context": str(hhem_df.iloc[idx]["context"])[:250],
            "response": str(hhem_df.iloc[idx]["response"])[:250],
            "true_label": int(y_true[idx]),
            "hhem_pred": int(hhem_preds[idx]),
            "ft_pred": int(ft_preds[idx]),
            "hhem_prob": float(hhem_probs[idx]),
            "ft_prob": float(ft_probs[idx]),
            "error_type": error_type,
            "model": model,
            "disagreement_type": (
                "agree" if hhem_preds[idx] == ft_preds[idx]
                else ("HHEM=hal,FT=fai" if hhem_preds[idx] == 1 else "HHEM=fai,FT=hal")
            ),
        }

    # Collect: 15 HHEM FP, 15 HHEM FN, 10 FT FP, 10 FT FN, 20 disagreements
    for idx in hhem_fp_idx[:15]:
        error_rows.append(make_row(idx, "false_positive", "HHEM"))
    for idx in hhem_fn_idx[:15]:
        error_rows.append(make_row(idx, "false_negative", "HHEM"))
    for idx in ft_fp_idx[:10]:
        error_rows.append(make_row(idx, "false_positive", "fine-tuned"))
    for idx in ft_fn_idx[:10]:
        error_rows.append(make_row(idx, "false_negative", "fine-tuned"))

    disagree_idx = np.where(hhem_halluc_ft_faithful | hhem_faithful_ft_halluc)[0]
    for idx in disagree_idx[:20]:
        error_rows.append(make_row(idx, "disagreement", "cross-model"))

    errors_df = pd.DataFrame(error_rows)
    # Deduplicate by id+model (some may appear in both error and disagreement lists)
    errors_df = errors_df.drop_duplicates(subset=["id", "model", "error_type"])
    errors_df.to_csv(args.output_errors, index=False)
    logger.info(f"Saved {len(errors_df)} error examples to {args.output_errors}")
    logger.info(f"Error type distribution: {errors_df['error_type'].value_counts().to_dict()}")
    logger.info("ensemble_and_analysis.py complete.")


if __name__ == "__main__":
    main()
