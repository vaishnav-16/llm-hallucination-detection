"""Shortcut baselines and the shortcut audit.

- ``answer_length``  logistic regression on answer-only surface features (chars, words, trailing period, digits).
- ``answer_tfidf``   TF-IDF (1-2 grams) logistic regression on the answer text only.
- ``lexical_support`` 1 - fraction of the answer's content words that occur in the context (uses the context, no training).

Answer-only baselines never see the context, so their scores cannot depend on context length. On a
length curve they are a flat line; a detector whose curve is just as flat is reading the answer, not the context.

``run_shortcut_audit`` trains each answer-only baseline on the training split of each benchmark and
scores the corresponding test slices, writing predictions (same format as the runner) and a summary table.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, SEED
from src.data.store import write_table
from src.verify.slices import META_COLUMNS, get_slice

logger = logging.getLogger(__name__)

_STOP = set("the a an of to in and or for on at by with is are was were be been it its this that as from which who what when where how".split())
_TOK = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def answer_features(responses: pd.Series) -> np.ndarray:
    r = responses.astype(str)
    return np.column_stack(
        [
            r.str.len(),
            r.str.split().str.len(),
            r.str.strip().str.endswith(".").astype(int),
            r.str.count(r"\d"),
        ]
    ).astype(float)


def fit_answer_length(train: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(answer_features(train["response"]), train["label"])
    return lambda df: model.predict_proba(answer_features(df["response"]))[:, 1]


def fit_answer_tfidf(train: pd.DataFrame):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train["response"].astype(str))
    clf = LogisticRegression(max_iter=4000, C=4.0, random_state=SEED).fit(X, train["label"])
    return lambda df: clf.predict_proba(vec.transform(df["response"].astype(str)))[:, 1]


def lexical_support(df: pd.DataFrame) -> np.ndarray:
    """1 - share of the answer's content tokens that appear anywhere in the context."""
    out = np.empty(len(df))
    for i, (ctx, resp) in enumerate(zip(df["context"].astype(str), df["response"].astype(str))):
        words = [w for w in _TOK.findall(resp.lower()) if w not in _STOP]
        if not words:
            out[i] = 0.5
            continue
        vocab = set(_TOK.findall(ctx.lower()))
        out[i] = 1.0 - sum(w in vocab for w in words) / len(words)
    return out


def _concat(slice_name: str) -> pd.DataFrame:
    return pd.concat([p.loader().assign(part=p.part) for p in get_slice(slice_name)], ignore_index=True)


def _write_preds(df: pd.DataFrame, scores: np.ndarray, detector: str, slice_name: str, elapsed: float) -> pd.DataFrame:
    pred = df[META_COLUMNS].copy()
    pred["part"] = df["part"].values
    pred["detector"] = detector
    pred["strategy"] = "none" if detector.startswith("answer_") else "full_context"
    pred["p_halluc"] = scores.astype(float)
    pred["n_chunks"] = 0 if detector.startswith("answer_") else 1
    pred["latency_ms"] = 1000.0 * elapsed / max(1, len(df))
    run_id = f"{detector}__{pred['strategy'].iloc[0]}__{slice_name}"
    pred["run_id"] = run_id
    write_table(pred, ARTIFACTS_DIR / "predictions" / f"{run_id}.parquet")
    return pred


AUDIT = [
    # (train slice, [evaluation slices])
    ("halueval_train", ["halueval_val", "halueval_test"]),
    ("ragtruth_train", ["ragtruth_val", "ragtruth_test"]),
    ("phantom_train", ["phantom_val", "phantom_grid_full"]),
]


def run_shortcut_audit(include_normalized: bool = True) -> pd.DataFrame:
    """Train answer-only baselines per benchmark, score its slices, write predictions and a summary."""
    from src.data.prepare import NORM_DIR

    audit = list(AUDIT)
    if include_normalized and (NORM_DIR / "halueval_normalized.parquet").exists():
        audit.append(("halueval_norm_train", ["halueval_norm_val", "halueval_norm_test"]))
    rows = []
    for train_name, eval_names in audit:
        train = _concat(train_name)
        logger.info("Shortcut audit: training answer-only baselines on %s (%d rows)", train_name, len(train))
        fitted = {"answer_length": fit_answer_length(train), "answer_tfidf": fit_answer_tfidf(train)}
        for eval_name in eval_names:
            df = _concat(eval_name)
            for det, fn in list(fitted.items()) + [("lexical_support", lexical_support)]:
                t0 = time.time()
                scores = fn(df)
                pred = _write_preds(df, np.asarray(scores), det, eval_name, time.time() - t0)
                acc = float(((pred["p_halluc"] >= 0.5).astype(int) == pred["label"]).mean())
                auc = float(roc_auc_score(pred["label"], pred["p_halluc"]))
                rows.append({"train": train_name, "eval": eval_name, "detector": det, "n": len(pred), "acc@0.5": acc, "auroc": auc})
                logger.info("  %-16s %-18s acc@0.5=%.3f auroc=%.3f (n=%d)", det, eval_name, acc, auc, len(pred))
    summary = pd.DataFrame(rows)
    write_table(summary, ARTIFACTS_DIR / "metrics" / "shortcut_baselines.csv")
    return summary
