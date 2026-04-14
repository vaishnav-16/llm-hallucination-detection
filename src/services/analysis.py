"""Error analysis — classify errors, extract examples, and categorize hallucination types."""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def truncate(text: str, n: int = 200) -> str:
    """Truncate string to n characters."""
    text = str(text)
    return text[:n] + "..." if len(text) > n else text


def get_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    error_type: str = "all",
) -> pd.DataFrame:
    """Extract error examples from predictions.

    Args:
        error_type: "all", "fp" (false positives), or "fn" (false negatives).
    """
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    if error_type == "fp":
        mask = fp_mask
    elif error_type == "fn":
        mask = fn_mask
    else:
        mask = fp_mask | fn_mask

    errors = df[mask].copy()
    errors["error_type"] = np.where(fp_mask[mask], "false_positive", "false_negative")
    return errors


def classify_errors(df: pd.DataFrame, n_show: int = 10) -> pd.DataFrame:
    """Analyze false positives and false negatives with statistics."""
    fp = df[(df["predicted_label"] == 1) & (df["label"] == 0)]
    fn = df[(df["predicted_label"] == 0) & (df["label"] == 1)]

    logger.info(f"False positives: {len(fp)}, False negatives: {len(fn)}")
    logger.info(f"Total errors: {len(fp) + len(fn)} / {len(df)} ({(len(fp)+len(fn))/len(df)*100:.1f}%)")

    score_col = "hhem_score" if "hhem_score" in df.columns else "predicted_prob"
    if score_col in df.columns:
        if len(fp) > 0:
            logger.info(f"FP score stats: mean={fp[score_col].mean():.4f}, std={fp[score_col].std():.4f}")
        if len(fn) > 0:
            logger.info(f"FN score stats: mean={fn[score_col].mean():.4f}, std={fn[score_col].std():.4f}")

    fp_out = fp.head(n_show).copy()
    fp_out["error_type"] = "false_positive"
    fn_out = fn.head(n_show).copy()
    fn_out["error_type"] = "false_negative"

    errors_df = pd.concat([fp_out, fn_out], ignore_index=True)
    cols = [c for c in ["id", "question", "context", "response", "label",
                         "predicted_label", score_col, "error_type"] if c in errors_df.columns]
    return errors_df[cols]


# --- Hallucination type classification ---

def _extract_numbers(text: str) -> set:
    return set(re.findall(r'\b\d+(?:\.\d+)?%?\b', str(text)))


def _extract_entities(text: str) -> set:
    return set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', str(text)))


def _word_set(text: str) -> set:
    return set(re.sub(r'[^a-z\s]', '', str(text).lower()).split())


def categorize_hallucination(row: dict, is_false_positive: bool = False) -> str:
    """Classify an error by hallucination type."""
    if is_false_positive:
        return "FAITHFUL_MISCLASSIFIED"

    context = str(row.get("context", ""))[:2000]
    response = str(row.get("response", ""))

    resp_nums = _extract_numbers(response)
    ctx_nums = _extract_numbers(context)
    if resp_nums and not resp_nums.issubset(ctx_nums):
        if len(resp_nums - ctx_nums) > 0:
            return "NUMERICAL"

    resp_ents = _extract_entities(response)
    ctx_ents = _extract_entities(context)
    if resp_ents and not resp_ents.issubset(ctx_ents):
        if len(resp_ents - ctx_ents) >= 2:
            return "ENTITY_CONFUSION"

    resp_words = _word_set(response)
    ctx_words = _word_set(context)
    if resp_words and ctx_words:
        overlap = len(resp_words & ctx_words) / len(resp_words)
        if overlap < 0.15:
            return "FACTUAL"
        elif overlap < 0.4:
            return "UNSUPPORTED_INFERENCE"

    contradiction_markers = ["not", "never", "no", "none", "unlike", "instead", "however"]
    resp_lower = response.lower()
    for marker in contradiction_markers:
        if marker in resp_lower and marker not in context.lower():
            return "CONTRADICTION"

    return "UNSUPPORTED_INFERENCE"


def analyze_dataset_errors(
    predictions_path: str,
    dataset_name: str,
    label_col: str = "label",
    pred_col: str = "predicted_label",
    prob_col: str = "predicted_prob",
) -> pd.DataFrame:
    """Analyze errors for an entire dataset with hallucination type classification."""
    df = pd.read_csv(predictions_path)
    if pred_col not in df.columns and prob_col in df.columns:
        df[pred_col] = (df[prob_col] >= 0.5).astype(int)

    if pred_col not in df.columns:
        logger.warning(f"No prediction column found in {predictions_path}")
        return pd.DataFrame()

    errors = []
    for idx, row in df.iterrows():
        true_label = int(row[label_col])
        pred_label = int(row[pred_col])
        if true_label != pred_label:
            is_fp = (pred_label == 1 and true_label == 0)
            errors.append({
                "dataset": dataset_name,
                "id": row.get("id", idx),
                "question": str(row.get("question", ""))[:200],
                "response": str(row.get("response", ""))[:200],
                "context_preview": str(row.get("context", ""))[:300],
                "true_label": true_label,
                "predicted_label": pred_label,
                "error_direction": "false_positive" if is_fp else "false_negative",
                "hallucination_type": categorize_hallucination(dict(row), is_false_positive=is_fp),
                "predicted_prob": float(row.get(prob_col, 0.5)) if prob_col in row.index else 0.5,
            })

    return pd.DataFrame(errors)
