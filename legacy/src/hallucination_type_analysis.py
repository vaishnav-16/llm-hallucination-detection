"""
hallucination_type_analysis.py — Categorize errors by hallucination type.
"""

import json
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def extract_numbers(text):
    """Extract numbers from text."""
    return set(re.findall(r'\b\d+(?:\.\d+)?%?\b', str(text)))


def extract_entities(text):
    """Extract capitalized multi-word phrases as rough entity proxy."""
    return set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', str(text)))


def word_set(text):
    return set(re.sub(r'[^a-z\s]', '', str(text).lower()).split())


def classify_error(row, is_false_positive=False):
    """Classify a misclassified example by hallucination type."""
    if is_false_positive:
        return "FAITHFUL_MISCLASSIFIED"

    context = str(row.get("context", ""))[:2000]
    response = str(row.get("response", ""))

    # Check for numerical errors
    resp_nums = extract_numbers(response)
    ctx_nums = extract_numbers(context)
    if resp_nums and not resp_nums.issubset(ctx_nums):
        novel_nums = resp_nums - ctx_nums
        if len(novel_nums) > 0:
            return "NUMERICAL"

    # Check for entity confusion
    resp_ents = extract_entities(response)
    ctx_ents = extract_entities(context)
    if resp_ents and not resp_ents.issubset(ctx_ents):
        novel_ents = resp_ents - ctx_ents
        if len(novel_ents) >= 2:
            return "ENTITY_CONFUSION"

    # Check word overlap
    resp_words = word_set(response)
    ctx_words = word_set(context)
    if resp_words and ctx_words:
        overlap = len(resp_words & ctx_words) / len(resp_words) if resp_words else 0
        if overlap < 0.15:
            return "FACTUAL"
        elif overlap < 0.4:
            return "UNSUPPORTED_INFERENCE"

    # Check for contradiction patterns
    contradiction_markers = ["not", "never", "no", "none", "unlike", "instead", "however"]
    resp_lower = response.lower()
    for marker in contradiction_markers:
        if marker in resp_lower and marker not in context.lower():
            return "CONTRADICTION"

    return "UNSUPPORTED_INFERENCE"


def analyze_dataset(predictions_path, dataset_name, label_col="label",
                    pred_col="predicted_label", prob_col="predicted_prob"):
    """Analyze errors for a dataset."""
    df = pd.read_csv(predictions_path)
    if pred_col not in df.columns and prob_col in df.columns:
        df[pred_col] = (df[prob_col] >= 0.5).astype(int)

    if pred_col not in df.columns:
        logger.warning(f"No prediction column found in {predictions_path}")
        return pd.DataFrame()

    # Find errors
    errors = []
    for idx, row in df.iterrows():
        true_label = int(row[label_col])
        pred_label = int(row[pred_col])
        if true_label != pred_label:
            is_fp = (pred_label == 1 and true_label == 0)
            error_type = classify_error(row, is_false_positive=is_fp)
            errors.append({
                "dataset": dataset_name,
                "id": row.get("id", idx),
                "question": str(row.get("question", ""))[:200],
                "response": str(row.get("response", ""))[:200],
                "context_preview": str(row.get("context", ""))[:300],
                "true_label": true_label,
                "predicted_label": pred_label,
                "error_direction": "false_positive" if is_fp else "false_negative",
                "hallucination_type": error_type,
                "predicted_prob": float(row.get(prob_col, 0.5)) if prob_col in row.index else 0.5,
            })

    return pd.DataFrame(errors)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_errors = []

    # 1. PHANTOM-trained model on PHANTOM test data
    if os.path.exists("results/phantom_trained_predictions.csv"):
        logger.info("Analyzing PHANTOM-trained model errors...")
        phantom_errors = analyze_dataset(
            "results/phantom_trained_predictions.csv", "PHANTOM",
            pred_col="predicted_label", prob_col="predicted_prob"
        )
        if len(phantom_errors) > 0:
            all_errors.append(phantom_errors)
            logger.info(f"  PHANTOM errors: {len(phantom_errors)}")
            logger.info(f"  Types: {phantom_errors['hallucination_type'].value_counts().to_dict()}")

    # 2. HaluEval-trained model on HaluEval
    if os.path.exists("results/finetuned_predictions.csv"):
        logger.info("Analyzing HaluEval-trained model errors...")
        halueval_errors = analyze_dataset(
            "results/finetuned_predictions.csv", "HaluEval",
            pred_col="predicted_label", prob_col="predicted_prob"
        )
        if len(halueval_errors) > 0:
            all_errors.append(halueval_errors)
            logger.info(f"  HaluEval errors: {len(halueval_errors)}")
            logger.info(f"  Types: {halueval_errors['hallucination_type'].value_counts().to_dict()}")

    # 3. HHEM baseline errors on HaluEval
    if os.path.exists("results/hhem_predictions.csv"):
        logger.info("Analyzing HHEM baseline errors...")
        hhem_df = pd.read_csv("results/hhem_predictions.csv")
        hhem_errors_list = []
        for idx, row in hhem_df.iterrows():
            true_label = int(row["label"])
            pred_label = int(row["predicted_label"])
            if true_label != pred_label:
                is_fp = (pred_label == 1 and true_label == 0)
                error_type = classify_error(row, is_false_positive=is_fp)
                hhem_errors_list.append({
                    "dataset": "HaluEval_HHEM",
                    "id": row.get("id", idx),
                    "question": str(row.get("question", ""))[:200],
                    "response": str(row.get("response", ""))[:200],
                    "context_preview": str(row.get("context", ""))[:300],
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "error_direction": "false_positive" if is_fp else "false_negative",
                    "hallucination_type": error_type,
                    "predicted_prob": float(row.get("hhem_score", 0.5)),
                })
        hhem_errors = pd.DataFrame(hhem_errors_list)
        if len(hhem_errors) > 0:
            all_errors.append(hhem_errors)
            logger.info(f"  HHEM errors: {len(hhem_errors)}")
            logger.info(f"  Types: {hhem_errors['hallucination_type'].value_counts().to_dict()}")

    if not all_errors:
        logger.error("No error data found!")
        return

    combined = pd.concat(all_errors, ignore_index=True)
    combined.to_csv("results/hallucination_type_analysis.csv", index=False)
    logger.info(f"\nSaved {len(combined)} error examples to results/hallucination_type_analysis.csv")

    # === FIGURE 1: Hallucination type distribution ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    type_order = ["NUMERICAL", "FACTUAL", "ENTITY_CONFUSION", "UNSUPPORTED_INFERENCE",
                  "CONTRADICTION", "FAITHFUL_MISCLASSIFIED"]
    colors = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6", "#95a5a6"]

    for ax_idx, (dataset_filter, title) in enumerate([
        (combined["dataset"].isin(["HaluEval", "HaluEval_HHEM"]), "HaluEval Errors"),
        (combined["dataset"] == "PHANTOM", "PHANTOM Errors"),
    ]):
        subset = combined[dataset_filter]
        if len(subset) == 0:
            ax_idx_ax = axes[ax_idx]
            ax_idx_ax.text(0.5, 0.5, "No errors", ha="center", va="center")
            ax_idx_ax.set_title(title)
            continue

        counts = subset["hallucination_type"].value_counts()
        types_present = [t for t in type_order if t in counts.index]
        vals = [counts.get(t, 0) for t in types_present]
        cols = [colors[type_order.index(t)] for t in types_present]

        ax = axes[ax_idx]
        bars = ax.barh(types_present, vals, color=cols)
        ax.set_xlabel("Number of Errors")
        ax.set_title(title, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)

    plt.suptitle("Error Distribution by Hallucination Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/hallucination_type_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved hallucination_type_distribution.png")

    # === FIGURE 2: Error type by model ===
    fig, ax = plt.subplots(figsize=(12, 6))

    model_groups = combined.groupby("dataset")["hallucination_type"].value_counts().unstack(fill_value=0)
    # Reorder columns
    cols_present = [t for t in type_order if t in model_groups.columns]
    model_groups = model_groups[cols_present]

    x = np.arange(len(model_groups.index))
    width = 0.12
    for i, col in enumerate(cols_present):
        offset = (i - len(cols_present)/2) * width
        ax.bar(x + offset, model_groups[col], width, label=col,
               color=colors[type_order.index(col)])

    ax.set_xlabel("Model / Dataset")
    ax.set_ylabel("Error Count")
    ax.set_title("Error Types by Model", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_groups.index, rotation=15, ha="right")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/error_type_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved error_type_by_model.png")

    # Summary
    logger.info("\n=== HALLUCINATION TYPE SUMMARY ===")
    for dataset in combined["dataset"].unique():
        subset = combined[combined["dataset"] == dataset]
        fn_only = subset[subset["error_direction"] == "false_negative"]
        logger.info(f"\n{dataset} ({len(subset)} errors, {len(fn_only)} false negatives):")
        if len(fn_only) > 0:
            dist = fn_only["hallucination_type"].value_counts()
            for t, c in dist.items():
                logger.info(f"  {t}: {c} ({c/len(fn_only)*100:.1f}%)")

    # Which types are hardest?
    fn_all = combined[combined["error_direction"] == "false_negative"]
    if len(fn_all) > 0:
        logger.info(f"\nOverall hardest hallucination types (false negatives):")
        for t, c in fn_all["hallucination_type"].value_counts().items():
            logger.info(f"  {t}: {c}")

    logger.info("\nhallucination_type_analysis.py complete.")


if __name__ == "__main__":
    main()
