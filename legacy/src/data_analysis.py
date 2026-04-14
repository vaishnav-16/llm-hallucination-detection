"""
data_analysis.py — Deep analysis and feature engineering on HaluEval QA dataset.

Computes statistics, extracts linguistic features, generates figures.
Saves enriched dataset to data/halueval_enriched.csv.
"""

import argparse
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42


def approx_tokens(text: str) -> int:
    """Rough token count: ~4 chars/token."""
    return max(1, len(str(text)) // 4)


def word_set(text: str) -> set:
    return set(re.sub(r'[^a-z\s]', '', str(text).lower()).split())


def jaccard(a: str, b: str) -> float:
    sa, sb = word_set(a), word_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def substring_ratio(response: str, context: str) -> float:
    """Fraction of response words that appear in context."""
    rw = word_set(response)
    cw = word_set(context)
    if not rw:
        return 0.0
    return len(rw & cw) / len(rw)


def lexical_diversity(text: str) -> float:
    words = str(text).lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def count_entities(text: str) -> int:
    """Rough NER proxy: count capitalized multi-word phrases."""
    tokens = str(text).split()
    count = 0
    i = 0
    while i < len(tokens):
        if tokens[i][0:1].isupper() and not tokens[i].isupper():
            j = i + 1
            while j < len(tokens) and tokens[j][0:1].isupper():
                j += 1
            if j > i:
                count += 1
            i = j
        else:
            i += 1
    return count


def avg_word_length(text: str) -> float:
    words = str(text).split()
    if not words:
        return 0.0
    return np.mean([len(w.strip('.,!?;:')) for w in words])


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting features...")
    df = df.copy()
    df["context_length"] = df["context"].astype(str).apply(len)
    df["response_length"] = df["response"].astype(str).apply(len)
    df["question_length"] = df["question"].astype(str).apply(len)
    df["context_tokens"] = df["context"].astype(str).apply(approx_tokens)
    df["response_tokens"] = df["response"].astype(str).apply(approx_tokens)

    logger.info("  Computing Jaccard overlap (response ↔ context)...")
    df["response_context_overlap"] = [jaccard(r, c) for r, c in zip(df["response"], df["context"])]

    logger.info("  Computing substring ratio...")
    df["response_context_substring_ratio"] = [substring_ratio(r, c) for r, c in zip(df["response"], df["context"])]

    logger.info("  Computing lexical diversity...")
    df["lexical_diversity"] = df["response"].apply(lexical_diversity)

    logger.info("  Counting entities...")
    df["num_entities_response"] = df["response"].apply(count_entities)

    logger.info("  Computing avg word length...")
    df["avg_word_length_response"] = df["response"].apply(avg_word_length)

    logger.info("  Computing question-response overlap...")
    df["question_response_overlap"] = [jaccard(q, r) for q, r in zip(df["question"], df["response"])]

    logger.info("Feature extraction complete.")
    return df


def print_stats(df: pd.DataFrame):
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    logger.info(f"Total examples: {len(df)}")
    for label, count in df["label"].value_counts().items():
        logger.info(f"  Label {label}: {count} ({count/len(df)*100:.1f}%)")

    for col, desc in [("context_length", "Context"), ("response_length", "Response"), ("question_length", "Question")]:
        vals = df[col]
        logger.info(f"\n{desc} length (chars):")
        logger.info(f"  mean={vals.mean():.1f}, median={vals.median():.1f}, std={vals.std():.1f}, min={vals.min()}, max={vals.max()}")
        logger.info(f"  Approx tokens: mean={vals.mean()/4:.1f}, max={vals.max()/4:.1f}")

    logger.info("\nFeature means by label:")
    feature_cols = ["context_length", "response_length", "response_context_overlap",
                    "response_context_substring_ratio", "lexical_diversity",
                    "num_entities_response", "avg_word_length_response", "question_response_overlap"]
    summary = df.groupby("label")[feature_cols].mean()
    logger.info("\n" + summary.to_string())

    logger.info("\nCorrelation with hallucination label:")
    corrs = df[feature_cols + ["label"]].corr()["label"].drop("label").sort_values(key=abs, ascending=False)
    for feat, corr in corrs.items():
        logger.info(f"  {feat}: {corr:.4f}")

    logger.info("\nKey Findings:")
    if df[df["label"]==1]["response_length"].mean() > df[df["label"]==0]["response_length"].mean():
        logger.info("  → Hallucinated responses tend to be LONGER than faithful ones.")
    else:
        logger.info("  → Faithful responses tend to be LONGER than hallucinated ones.")

    diff = df[df["label"]==0]["response_context_overlap"].mean() - df[df["label"]==1]["response_context_overlap"].mean()
    logger.info(f"  → Faithful responses have {diff:.4f} higher word overlap with context than hallucinated.")

    top_feat = corrs.index[0]
    logger.info(f"  → Strongest predictor: '{top_feat}' (r={corrs[top_feat]:.4f})")


def generate_figures(df: pd.DataFrame, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    label_colors = {0: "steelblue", 1: "tomato"}
    label_names = {0: "Faithful", 1: "Hallucinated"}

    # Context length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl in [0, 1]:
        vals = df[df["label"] == lbl]["context_length"]
        ax.hist(vals, bins=40, alpha=0.6, color=label_colors[lbl], label=label_names[lbl], density=True)
    ax.set_xlabel("Context Length (chars)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Context Length Distribution by Label", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "data_context_length_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Response length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl in [0, 1]:
        vals = df[df["label"] == lbl]["response_length"]
        ax.hist(vals, bins=40, alpha=0.6, color=label_colors[lbl], label=label_names[lbl], density=True)
    ax.set_xlabel("Response Length (chars)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Response Length Distribution by Label", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "data_response_length_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Feature correlation heatmap
    feature_cols = ["context_length", "response_length", "question_length",
                    "response_context_overlap", "response_context_substring_ratio",
                    "lexical_diversity", "num_entities_response",
                    "avg_word_length_response", "question_response_overlap", "label"]
    corr_mat = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix (incl. label)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "data_feature_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Box plot: response_context_overlap by label
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (feat, title) in enumerate([
        ("response_context_overlap", "Response-Context Word Overlap (Jaccard)"),
        ("response_context_substring_ratio", "Response Words in Context (Ratio)"),
    ]):
        data = [df[df["label"] == lbl][feat].values for lbl in [0, 1]]
        bp = axes[i].boxplot(data, labels=["Faithful", "Hallucinated"], patch_artist=True,
                             medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("tomato")
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].set_ylabel(feat, fontsize=10)
    plt.suptitle("Overlap Features by Label", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "data_overlap_vs_label.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Figures saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/halueval_qa_normalized.csv")
    parser.add_argument("--output", default="data/halueval_enriched.csv")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows.")

    df = extract_features(df)
    print_stats(df)
    generate_figures(df, args.figures_dir)

    df.to_csv(args.output, index=False)
    logger.info(f"Saved enriched dataset to {args.output}")
    logger.info("data_analysis.py complete.")


if __name__ == "__main__":
    main()
