"""
load_phantom.py — Load and analyze PHANTOM financial hallucination dataset.

Tries multiple loading strategies for seyled/Phantom_Hallucination_Detection.
Normalizes to same format as HaluEval. Generates comparison figures.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42


def try_load_phantom():
    """Try multiple loading strategies for PHANTOM."""
    from datasets import load_dataset

    strategies = [
        {"data_files": "PhantomDataset/Phantom_10K_seed.csv"},
        {"data_files": "Phantom_10K_seed.csv"},
        {"data_files": "PhantomDataset/Phantom_def14A_seed.csv"},
        {"data_files": "Phantom_def14A_seed.csv"},
        {},  # default
    ]

    for kwargs in strategies:
        try:
            desc = str(kwargs) if kwargs else "default"
            logger.info(f"Trying PHANTOM load with {desc}...")
            ds = load_dataset("seyled/Phantom_Hallucination_Detection", **kwargs)
            split = list(ds.keys())[0]
            logger.info(f"  Loaded! Split={split}, rows={len(ds[split])}")
            logger.info(f"  Columns: {ds[split].column_names}")
            return ds[split]
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    # Try to list available files
    try:
        from huggingface_hub import list_repo_files
        files = list(list_repo_files("seyled/Phantom_Hallucination_Detection", repo_type="dataset"))
        logger.info(f"Available files in repo: {files}")
        csv_files = [f for f in files if f.endswith(".csv")]
        for csv_file in csv_files[:3]:
            try:
                ds = load_dataset("seyled/Phantom_Hallucination_Detection", data_files=csv_file)
                split = list(ds.keys())[0]
                logger.info(f"  Loaded {csv_file}! rows={len(ds[split])}")
                return ds[split]
            except Exception as e2:
                logger.warning(f"  {csv_file} failed: {e2}")
    except Exception as e:
        logger.warning(f"Could not list repo files: {e}")

    return None


def inspect_and_normalize(ds_split, num_samples: int = 500) -> pd.DataFrame:
    """Inspect PHANTOM schema and normalize to standard format."""
    import random
    random.seed(SEED)

    cols = ds_split.column_names
    logger.info(f"\nPHANTOM Schema:")
    logger.info(f"  Columns: {cols}")
    logger.info(f"  Total rows: {len(ds_split)}")

    # Print 5 sample rows
    logger.info("\nSample rows:")
    for i in range(min(5, len(ds_split))):
        row = dict(ds_split[i])
        for k, v in row.items():
            val_str = str(v)[:150]
            logger.info(f"  [{i}] {k}: {val_str}")
        logger.info("")

    # Detect columns
    def find_col(candidates):
        for c in candidates:
            if c in cols:
                return c
        # Try partial match
        for c in cols:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        return None

    q_col = find_col(["question", "query", "prompt", "input"])
    ctx_col = find_col(["context", "document", "passage", "text", "content", "excerpt"])
    ans_col = find_col(["answer", "response", "correct_answer", "right_answer", "output"])
    hal_col = find_col(["hallucinated", "hallucination", "wrong_answer", "incorrect_answer", "false_answer"])
    label_col = find_col(["label", "is_hallucination", "hallucinated", "target"])

    logger.info(f"\nDetected columns: question={q_col}, context={ctx_col}, answer={ans_col}, hallucinated={hal_col}, label={label_col}")

    # Sample
    n = min(num_samples, len(ds_split))
    indices = list(range(len(ds_split)))
    random.shuffle(indices)
    indices = indices[:n]

    rows = []
    for i, idx in enumerate(indices):
        item = dict(ds_split[idx])

        question = str(item.get(q_col, "")) if q_col else ""
        context = str(item.get(ctx_col, "")) if ctx_col else ""

        # Handle paired format (correct + hallucinated) or single-example format
        if ans_col and hal_col:
            correct_ans = str(item.get(ans_col, ""))
            hal_ans = str(item.get(hal_col, ""))
            rows.append({
                "id": f"phantom_{idx}_faithful",
                "question": question,
                "context": context,
                "response": correct_ans,
                "label": 0,
            })
            rows.append({
                "id": f"phantom_{idx}_hallucinated",
                "question": question,
                "context": context,
                "response": hal_ans,
                "label": 1,
            })
        elif label_col and ans_col:
            # Single-response format with label
            response = str(item.get(ans_col, ""))
            label_val = item.get(label_col, 0)
            try:
                label = int(float(str(label_val)))
            except Exception:
                label = 1 if str(label_val).lower() in ["true", "yes", "hallucinated", "1"] else 0
            rows.append({
                "id": f"phantom_{idx}",
                "question": question,
                "context": context,
                "response": response,
                "label": label,
            })
        else:
            # Try to use all string columns
            all_vals = list(item.values())
            str_vals = [str(v) for v in all_vals if isinstance(v, str) and len(str(v)) > 10]
            if len(str_vals) >= 2:
                rows.append({
                    "id": f"phantom_{idx}",
                    "question": str_vals[0][:200] if q_col is None else question,
                    "context": str_vals[1][:1000] if ctx_col is None else context,
                    "response": str_vals[2] if len(str_vals) > 2 else str_vals[1],
                    "label": 0,
                })

    df = pd.DataFrame(rows)
    logger.info(f"\nNormalized PHANTOM: {len(df)} rows")
    if len(df) > 0:
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add same features as HaluEval for comparison."""
    def word_set(text):
        return set(re.sub(r'[^a-z\s]', '', str(text).lower()).split())

    def jaccard(a, b):
        sa, sb = word_set(a), word_set(b)
        return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0

    def substr_ratio(r, c):
        rw, cw = word_set(r), word_set(c)
        return len(rw & cw) / len(rw) if rw else 0.0

    def lex_div(text):
        words = str(text).lower().split()
        return len(set(words)) / len(words) if words else 0.0

    df = df.copy()
    df["context_length"] = df["context"].astype(str).apply(len)
    df["response_length"] = df["response"].astype(str).apply(len)
    df["question_length"] = df["question"].astype(str).apply(len)
    df["response_context_overlap"] = [jaccard(r, c) for r, c in zip(df["response"], df["context"])]
    df["response_context_substring_ratio"] = [substr_ratio(r, c) for r, c in zip(df["response"], df["context"])]
    df["lexical_diversity"] = df["response"].apply(lex_div)
    return df


def compare_datasets_figure(halueval_path: str, phantom_df: pd.DataFrame, figures_dir: str):
    """Generate side-by-side comparison figure."""
    try:
        halueval_df = pd.read_csv(halueval_path)
    except Exception:
        logger.warning("Could not load HaluEval for comparison figure.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    metrics = [
        ("context_length", "Context Length (chars)"),
        ("response_length", "Response Length (chars)"),
        ("response_context_overlap", "Response-Context Overlap (Jaccard)"),
    ]

    for col, (feat, title) in enumerate(metrics):
        for row, (df, dname) in enumerate([(halueval_df, "HaluEval QA"), (phantom_df, "PHANTOM")]):
            if feat not in df.columns:
                continue
            ax = axes[row][col]
            for lbl, color, name in [(0, "steelblue", "Faithful"), (1, "tomato", "Hallucinated")]:
                subset = df[df["label"] == lbl][feat].dropna()
                if len(subset) > 0:
                    ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, density=True)
            ax.set_title(f"{dname}: {title}", fontsize=9, fontweight="bold")
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)

    plt.suptitle("HaluEval vs PHANTOM: Dataset Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "phantom_data_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phantom_data_analysis.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output", default="data/phantom_normalized.csv")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    ds = try_load_phantom()

    if ds is None:
        logger.error("ALL PHANTOM loading attempts failed. Creating synthetic PHANTOM-like dataset.")
        # Create synthetic financial QA dataset as fallback
        rng = np.random.RandomState(SEED)
        questions = [
            "What was the company's revenue in Q3?",
            "Who is the current CEO of the company?",
            "What are the primary risk factors mentioned?",
            "What was the net income for the fiscal year?",
            "How many employees does the company have?",
        ] * 100
        contexts = [
            "The company reported Q3 revenue of $4.2 billion, up 12% year-over-year driven by strong product sales.",
            "Jane Smith has served as CEO since 2019, previously serving as CFO at Goldman Sachs.",
            "Key risks include market volatility, regulatory changes, and supply chain disruptions.",
            "Net income for fiscal year 2023 was $892 million, representing a 8% increase from prior year.",
            "As of December 2023, the company employed approximately 45,000 full-time employees globally.",
        ] * 100
        correct_answers = [
            "$4.2 billion", "Jane Smith", "Market volatility, regulatory changes, supply chain disruptions",
            "$892 million", "45,000"
        ] * 100
        hallucinated_answers = [
            "$3.8 billion", "John Smith", "Only market volatility",
            "$750 million", "55,000"
        ] * 100

        rows = []
        for i in range(min(args.num_samples, 500)):
            rows.append({"id": f"phantom_synth_{i}_faithful", "question": questions[i],
                         "context": contexts[i % 5], "response": correct_answers[i % 5], "label": 0})
            rows.append({"id": f"phantom_synth_{i}_hallucinated", "question": questions[i],
                         "context": contexts[i % 5], "response": hallucinated_answers[i % 5], "label": 1})
        df = pd.DataFrame(rows)
        df["phantom_synthetic"] = True
        logger.warning(f"Created synthetic PHANTOM dataset: {len(df)} rows")
    else:
        df = inspect_and_normalize(ds, args.num_samples)
        if len(df) == 0:
            logger.error("Normalization produced empty dataframe. Aborting PHANTOM phase.")
            return

    df = add_features(df)

    logger.info(f"\nPHANTOM dataset summary:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Label dist: {df['label'].value_counts().to_dict()}")
    logger.info(f"  Avg context len: {df['context_length'].mean():.1f} chars")
    logger.info(f"  Avg response len: {df['response_length'].mean():.1f} chars")

    df.to_csv(args.output, index=False)
    logger.info(f"Saved to {args.output}")

    compare_datasets_figure("data/halueval_enriched.csv", df, args.figures_dir)

    logger.info("load_phantom.py complete.")


if __name__ == "__main__":
    main()
