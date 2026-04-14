"""
error_analysis.py — Analyze false positives and false negatives from HHEM predictions.

Reads results/hhem_predictions.csv, identifies errors, prints examples,
and saves to results/error_analysis.csv.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def truncate(text: str, n: int = 200) -> str:
    """Truncate string to n characters."""
    text = str(text)
    return text[:n] + "..." if len(text) > n else text


def analyze_errors(df: pd.DataFrame, n_show: int = 10) -> pd.DataFrame:
    """Find and analyze false positives and false negatives."""
    # False positives: predicted hallucinated (1) but actually faithful (0)
    fp = df[(df["predicted_label"] == 1) & (df["label"] == 0)].copy()
    # False negatives: predicted faithful (0) but actually hallucinated (1)
    fn = df[(df["predicted_label"] == 0) & (df["label"] == 1)].copy()

    logger.info(f"Total false positives (predicted hallucinated, actually faithful): {len(fp)}")
    logger.info(f"Total false negatives (predicted faithful, actually hallucinated): {len(fn)}")
    logger.info(f"Total errors: {len(fp) + len(fn)} / {len(df)} ({(len(fp)+len(fn))/len(df)*100:.1f}%)")

    # Score statistics for errors
    logger.info(f"\nFalse Positive score stats: mean={fp['hhem_score'].mean():.4f}, std={fp['hhem_score'].std():.4f}, min={fp['hhem_score'].min():.4f}")
    logger.info(f"False Negative score stats: mean={fn['hhem_score'].mean():.4f}, std={fn['hhem_score'].std():.4f}, max={fn['hhem_score'].max():.4f}")

    # Context length analysis
    df["context_len"] = df["context"].astype(str).str.len()
    df["response_len"] = df["response"].astype(str).str.len()

    fp_ctx = df[(df["predicted_label"] == 1) & (df["label"] == 0)]["context_len"].mean()
    fn_ctx = df[(df["predicted_label"] == 0) & (df["label"] == 1)]["context_len"].mean()
    correct_ctx = df[df["predicted_label"] == df["label"]]["context_len"].mean()

    logger.info(f"\nContext length analysis:")
    logger.info(f"  Correctly classified: avg context len = {correct_ctx:.0f} chars")
    logger.info(f"  False positives: avg context len = {fp_ctx:.0f} chars")
    logger.info(f"  False negatives: avg context len = {fn_ctx:.0f} chars")

    # Print false positive examples
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {min(n_show, len(fp))} FALSE POSITIVES (predicted=hallucinated, actual=faithful)")
    logger.info(f"{'='*60}")
    for i, (_, row) in enumerate(fp.head(n_show).iterrows()):
        logger.info(f"\nFP #{i+1} | ID: {row['id']} | Score: {row['hhem_score']:.4f}")
        logger.info(f"  Question:  {truncate(row['question'], 100)}")
        logger.info(f"  Context:   {truncate(row['context'], 200)}")
        logger.info(f"  Response:  {truncate(row['response'], 200)}")
        logger.info(f"  True=0 (faithful), Predicted=1 (hallucinated)")

    # Print false negative examples
    logger.info(f"\n{'='*60}")
    logger.info(f"TOP {min(n_show, len(fn))} FALSE NEGATIVES (predicted=faithful, actual=hallucinated)")
    logger.info(f"{'='*60}")
    for i, (_, row) in enumerate(fn.head(n_show).iterrows()):
        logger.info(f"\nFN #{i+1} | ID: {row['id']} | Score: {row['hhem_score']:.4f}")
        logger.info(f"  Question:  {truncate(row['question'], 100)}")
        logger.info(f"  Context:   {truncate(row['context'], 200)}")
        logger.info(f"  Response:  {truncate(row['response'], 200)}")
        logger.info(f"  True=1 (hallucinated), Predicted=0 (faithful)")

    # Print pattern observations
    logger.info(f"\n{'='*60}")
    logger.info("PATTERN OBSERVATIONS")
    logger.info(f"{'='*60}")
    logger.info(f"  1. False positives (score range): {fp['hhem_score'].min():.3f} – {fp['hhem_score'].max():.3f}")
    logger.info(f"     FPs cluster near threshold: {((fp['hhem_score'] > 0.3) & (fp['hhem_score'] <= 0.5)).sum()} FPs have score in [0.3, 0.5]")
    logger.info(f"  2. False negatives (score range): {fn['hhem_score'].min():.3f} – {fn['hhem_score'].max():.3f}")
    logger.info(f"     FNs cluster near threshold: {((fn['hhem_score'] > 0.5) & (fn['hhem_score'] < 0.7)).sum()} FNs have score in [0.5, 0.7]")
    logger.info(f"  3. Faithful responses avg score: {df[df['label']==0]['hhem_score'].mean():.4f}")
    logger.info(f"     Hallucinated responses avg score: {df[df['label']==1]['hhem_score'].mean():.4f}")

    # Build combined error dataframe
    fp_out = fp.head(n_show).copy()
    fp_out["error_type"] = "false_positive"
    fn_out = fn.head(n_show).copy()
    fn_out["error_type"] = "false_negative"
    errors_df = pd.concat([fp_out, fn_out], ignore_index=True)
    errors_df = errors_df[["id", "question", "context", "response", "label", "predicted_label", "hhem_score", "error_type"]]

    return errors_df


def main():
    parser = argparse.ArgumentParser(description="Error analysis of HHEM predictions.")
    parser.add_argument("--input", type=str, default="results/hhem_predictions.csv")
    parser.add_argument("--output", type=str, default="results/error_analysis.csv")
    parser.add_argument("--n_show", type=int, default=10)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    logger.info(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows.")

    errors_df = analyze_errors(df, args.n_show)

    errors_df.to_csv(args.output, index=False)
    logger.info(f"\nSaved error analysis to {args.output}")
    logger.info("error_analysis.py complete.")


if __name__ == "__main__":
    main()
