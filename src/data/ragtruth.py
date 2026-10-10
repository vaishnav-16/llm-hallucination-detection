"""RAGTruth loader (``wandb/RAGTruth-processed``) at a pinned revision.

RAGTruth contains real responses from 6 LLMs to the same source context, annotated at span level.
A response is labelled hallucinated (1) if it has any ``evident_conflict`` or ``baseless_info`` span.
Rows with quality other than "good" (incorrect refusals, truncated outputs) are dropped.
``pair_id`` groups all responses to one source (task type + query + context), so they share a split.
"""

from __future__ import annotations

import ast
import logging

import pandas as pd

from src.data.hub import download_dataset_file
from src.data.schema import stable_id, validate

logger = logging.getLogger(__name__)

FILES = {"train": "data/train-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}


def _label(processed: object) -> int:
    d = processed if isinstance(processed, dict) else ast.literal_eval(str(processed))
    return int(int(d.get("evident_conflict", 0)) + int(d.get("baseless_info", 0)) > 0)


def load_ragtruth(official_split: str) -> pd.DataFrame:
    """Load RAGTruth's official 'train' or 'test' split as canonical rows."""
    raw = pd.read_parquet(download_dataset_file("ragtruth", FILES[official_split]))
    n0 = len(raw)
    raw = raw[raw["quality"] == "good"].copy()
    logger.info("RAGTruth %s: kept %d/%d rows with quality == 'good'", official_split, len(raw), n0)
    df = pd.DataFrame(
        {
            "pair_id": [stable_id("ragtruth", t, q, c) for t, q, c in zip(raw["task_type"], raw["query"], raw["context"])],
            "example_id": [stable_id("ragtruth", official_split, i) for i in raw["id"]],
            "question": raw["query"].astype(str),
            "context": raw["context"].astype(str),
            "response": raw["output"].astype(str),
            "label": [_label(p) for p in raw["hallucination_labels_processed"]],
            "task_type": raw["task_type"].astype(str),
        }
    )
    df["dataset"] = "ragtruth"
    df["variant"] = "original"
    df["filing"] = ""
    df["target_tokens"] = 0
    df["position"] = ""
    df["source_file"] = f"ragtruth_{official_split}"
    empty = df["response"].str.strip() == ""
    if empty.any():
        logger.warning("RAGTruth %s: dropping %d empty responses", official_split, int(empty.sum()))
        df = df[~empty]
    return validate(df, f"ragtruth[{official_split}]")
