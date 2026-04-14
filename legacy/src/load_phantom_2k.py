"""
load_phantom_2k.py — Load PHANTOM 2K-token context variant and normalize.
"""

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/phantom_2k_normalized.csv")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    # Step 1: List files in repo
    from huggingface_hub import list_repo_files
    files = list_repo_files("seyled/Phantom_Hallucination_Detection", repo_type="dataset")
    files_2k = [f for f in files if "2000" in f or "2k" in f.lower()]
    logger.info(f"All files with '2000' or '2k': {files_2k}")
    logger.info(f"All CSV files: {[f for f in files if f.endswith('.csv')]}")

    # Step 2: Try loading 2K variant
    from datasets import load_dataset
    candidates = [
        "PhantomDataset/Phantom_10k_2000tokens_middle.csv",
        "PhantomDataset/Phantom_10k_2000tokens_beginning.csv",
        "PhantomDataset/Phantom_10k_2000tokens_end.csv",
        "PhantomDataset/Phantom_10K_2000tokens_middle.csv",
        "PhantomDataset/Phantom_10K_2000tokens_beginning.csv",
        "PhantomDataset/Phantom_10K_2000tokens_end.csv",
    ]
    # Add any files found with "2000" in name
    for f in files_2k:
        if f.endswith(".csv") and f not in candidates:
            candidates.append(f)

    ds = None
    loaded_file = None
    for cand in candidates:
        try:
            logger.info(f"Trying: {cand}")
            ds = load_dataset("seyled/Phantom_Hallucination_Detection", data_files=cand)
            split = list(ds.keys())[0]
            ds = ds[split]
            loaded_file = cand
            logger.info(f"  SUCCESS: {len(ds)} rows, columns={ds.column_names}")
            break
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    if ds is None:
        logger.error("Could not load any 2K variant. Falling back to seed variant.")
        try:
            ds = load_dataset("seyled/Phantom_Hallucination_Detection",
                              data_files="PhantomDataset/Phantom_10K_seed.csv")
            split = list(ds.keys())[0]
            ds = ds[split]
            loaded_file = "Phantom_10K_seed.csv (fallback)"
            logger.info(f"Loaded seed fallback: {len(ds)} rows")
        except Exception as e:
            logger.error(f"Seed also failed: {e}")
            return

    # Step 3: Print schema
    cols = ds.column_names
    logger.info(f"\nSchema: {cols}")
    logger.info(f"Dtypes: { {c: str(ds.features[c]) for c in cols} }")
    for i in range(min(3, len(ds))):
        row = dict(ds[i])
        logger.info(f"\nSample {i}:")
        for k, v in row.items():
            logger.info(f"  {k}: {str(v)[:200]}")

    # Step 4: Normalize
    import random
    random.seed(SEED)

    def find_col(candidates_list):
        for c in candidates_list:
            if c in cols:
                return c
        for c in cols:
            for cand in candidates_list:
                if cand.lower() in c.lower():
                    return c
        return None

    q_col = find_col(["question", "query", "prompt", "input"])
    ctx_col = find_col(["context", "document", "passage", "text", "content", "excerpt"])
    ans_col = find_col(["answer", "response", "correct_answer", "right_answer", "output"])
    hal_col = find_col(["hallucinated", "hallucination", "wrong_answer", "incorrect_answer", "false_answer",
                         "hallucinated_answer"])
    label_col = find_col(["label", "is_hallucination", "hallucinated", "target"])

    logger.info(f"\nDetected: question={q_col}, context={ctx_col}, answer={ans_col}, hallucinated={hal_col}, label={label_col}")

    rows = []
    for idx in range(len(ds)):
        item = dict(ds[idx])
        question = str(item.get(q_col, "")) if q_col else ""
        context = str(item.get(ctx_col, "")) if ctx_col else ""

        if ans_col and hal_col:
            correct_ans = str(item.get(ans_col, ""))
            hal_ans = str(item.get(hal_col, ""))
            if correct_ans.strip() and correct_ans.strip().lower() != "nan":
                rows.append({
                    "id": f"phantom2k_{idx}_faithful",
                    "question": question,
                    "context": context,
                    "response": correct_ans,
                    "label": 0,
                })
            if hal_ans.strip() and hal_ans.strip().lower() != "nan":
                rows.append({
                    "id": f"phantom2k_{idx}_hallucinated",
                    "question": question,
                    "context": context,
                    "response": hal_ans,
                    "label": 1,
                })
        elif label_col and ans_col:
            response = str(item.get(ans_col, ""))
            label_val = item.get(label_col, 0)
            try:
                lv = str(label_val).strip().lower()
                if lv in ["hallucination", "hallucinated", "true", "yes", "1"]:
                    label = 1
                elif lv in ["not hallucination", "not_hallucination", "faithful", "false", "no", "0"]:
                    label = 0
                else:
                    label = int(float(lv))
            except Exception:
                label = 1 if "hallucin" in str(label_val).lower() and "not" not in str(label_val).lower() else 0
            rows.append({
                "id": f"phantom2k_{idx}",
                "question": question,
                "context": context,
                "response": response,
                "label": label,
            })

    df = pd.DataFrame(rows)
    logger.info(f"\nNormalized: {len(df)} rows")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Step 5: Context length stats
    df["context_length"] = df["context"].astype(str).apply(len)
    df["response_length"] = df["response"].astype(str).apply(len)
    logger.info(f"\nContext length stats (chars):")
    logger.info(f"  Mean:   {df['context_length'].mean():.0f}")
    logger.info(f"  Median: {df['context_length'].median():.0f}")
    logger.info(f"  Min:    {df['context_length'].min()}")
    logger.info(f"  Max:    {df['context_length'].max()}")

    # Step 6: Save
    df.to_csv(args.output, index=False)
    logger.info(f"\nSaved to {args.output}")

    # Step 7: Compare with seed
    try:
        seed_df = pd.read_csv("data/phantom_normalized.csv")
        logger.info(f"\n=== PHANTOM SEED vs 2K COMPARISON ===")
        logger.info(f"  Seed:  {len(seed_df)} rows, context mean={seed_df['context_length'].mean():.0f} chars, labels={seed_df['label'].value_counts().to_dict()}")
        logger.info(f"  2K:    {len(df)} rows, context mean={df['context_length'].mean():.0f} chars, labels={df['label'].value_counts().to_dict()}")
    except Exception:
        pass

    logger.info(f"\nLoaded from: {loaded_file}")
    logger.info("load_phantom_2k.py complete.")


if __name__ == "__main__":
    main()
