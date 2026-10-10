"""Canonical example schema shared by every dataset loader, plus strict validation.

Every loader returns a DataFrame with exactly these columns. ``label`` is 1 for a
hallucinated response and 0 for a faithful one. ``pair_id`` groups rows that must
never be split across train/val/test: the correct and hallucinated answers of one
question (HaluEval, PHANTOM, including every long-context variant of that question),
or all model responses to one source context (RAGTruth).
"""

from __future__ import annotations

import hashlib

import pandas as pd

COLUMNS: list[str] = [
    "example_id",     # unique row id
    "pair_id",        # split-grouping id
    "dataset",        # halueval | phantom | ragtruth
    "variant",        # halueval: original | normalized ; others: original
    "filing",         # phantom: 10k | def14a | 8k | 497k ; others: ""
    "target_tokens",  # phantom nominal context length (500 for seed) ; others: 0
    "position",       # phantom: seed | beginning | middle | end ; others: ""
    "task_type",      # ragtruth: QA | Summary | Data2txt ; halueval/phantom: QA
    "question",
    "context",
    "response",
    "label",          # 1 = hallucinated, 0 = faithful
    "source_file",
]

STRING_COLUMNS = [c for c in COLUMNS if c not in ("target_tokens", "label")]


class SchemaError(ValueError):
    """Raised when a dataset fails validation. Never caught silently."""


def stable_id(*parts: object, length: int = 16) -> str:
    """Deterministic short hash of the given parts (stable across runs and machines)."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:length]


def validate(df: pd.DataFrame, name: str, require_both_classes: bool = True) -> pd.DataFrame:
    """Check columns, types, nulls, label values, id uniqueness and class balance.

    Returns the frame with canonical column order and dtypes. Raises SchemaError on any problem.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")
    df = df[COLUMNS].copy()
    if df.empty:
        raise SchemaError(f"{name}: no rows")
    for col in ("question", "context", "response", "example_id", "pair_id"):
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        # Context may legitimately be empty only for context-ablation copies, which are never validated here.
        if bad.any():
            raise SchemaError(f"{name}: {int(bad.sum())} empty values in column {col!r}")
    if not set(df["label"].unique()) <= {0, 1}:
        raise SchemaError(f"{name}: labels must be 0/1, got {sorted(df['label'].unique())}")
    if require_both_classes and df["label"].nunique() != 2:
        raise SchemaError(f"{name}: only one class present ({df['label'].value_counts().to_dict()})")
    dup = df["example_id"].duplicated()
    if dup.any():
        raise SchemaError(f"{name}: {int(dup.sum())} duplicate example_id values")
    for col in STRING_COLUMNS:
        df[col] = df[col].astype(str)
    df["label"] = df["label"].astype("int8")
    df["target_tokens"] = df["target_tokens"].astype("int32")
    return df.reset_index(drop=True)
