"""Leak-free split assignment, built once and committed to ``artifacts/splits.json``.

Rules (all grouped by ``pair_id``):
- PHANTOM 8-K and 497K seed pairs: train 85% / val 15%.
- PHANTOM 10-K and DEF 14A: test only, at every length and position. Their long variants reuse the seed
  questions and answers, so none of their rows (not even seeds) may ever be trained on.
- HaluEval: train 70% / val 15% / test 15%.
- RAGTruth: the official test split is test; the official train split is divided train 85% / val 15%.

``grid_sub`` is a fixed sample of PHANTOM test pairs that appear in every length x position cell of
their filing. Every detector scores exactly these pairs, so comparisons across detectors and across
lengths are paired.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, SEED
from src.data import phantom
from src.data.store import read_json, write_json

logger = logging.getLogger(__name__)

SPLITS_PATH = ARTIFACTS_DIR / "splits.json"
GRID_SUB_PAIRS = 200


def _partition(pair_ids: Iterable[str], fractions: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
    ids = sorted(set(pair_ids))
    order = rng.permutation(len(ids))
    out: dict[str, str] = {}
    start = 0
    names = list(fractions)
    for i, name in enumerate(names):
        n = len(ids) - start if i == len(names) - 1 else int(round(fractions[name] * len(ids)))
        for j in order[start : start + n]:
            out[ids[j]] = name
        start += n
    return out


def build_splits(
    halueval: pd.DataFrame,
    ragtruth_train: pd.DataFrame,
    ragtruth_test: pd.DataFrame,
    phantom_seeds: dict[str, pd.DataFrame],
    phantom_cell_pairs: dict[str, list[set[str]]],
    grid_sub_pairs: int = GRID_SUB_PAIRS,
    seed: int = SEED,
) -> dict:
    """Return the splits document. ``phantom_cell_pairs[filing]`` lists the pair-id set of every cell."""
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[str, str]] = {}

    splits["halueval"] = _partition(halueval["pair_id"], {"train": 0.70, "val": 0.15, "test": 0.15}, rng)

    rt_test = set(ragtruth_test["pair_id"])
    rt_train = set(ragtruth_train["pair_id"]) - rt_test
    overlap = set(ragtruth_train["pair_id"]) & rt_test
    if overlap:
        logger.warning("RAGTruth: %d sources appear in both official splits; assigned to test", len(overlap))
    rag = _partition(rt_train, {"train": 0.85, "val": 0.15}, rng)
    rag.update({p: "test" for p in rt_test})
    splits["ragtruth"] = rag

    ph: dict[str, str] = {}
    train_pairs = set().union(*(set(phantom_seeds[f]["pair_id"]) for f in phantom.TRAIN_FILINGS))
    ph.update(_partition(train_pairs, {"train": 0.85, "val": 0.15}, rng))
    for f in phantom.TEST_FILINGS:
        test_pairs = set(phantom_seeds[f]["pair_id"]).union(*phantom_cell_pairs[f])  # includes long_only pairs
        for p in sorted(test_pairs):
            if p in ph:
                raise ValueError("PHANTOM pair id shared between train and test filings")
            ph[p] = "test"
    splits["phantom"] = ph

    grid_sub: dict[str, list[str]] = {}
    for f in phantom.TEST_FILINGS:
        common = set(phantom_seeds[f]["pair_id"]).intersection(*phantom_cell_pairs[f])
        chosen = sorted(common)
        if len(chosen) > grid_sub_pairs:
            chosen = sorted(rng.choice(chosen, size=grid_sub_pairs, replace=False).tolist())
        grid_sub[f] = chosen
        logger.info("grid_sub[%s]: %d pairs (of %d present in every cell)", f, len(chosen), len(common))

    counts = {
        ds: pd.Series(list(m.values())).value_counts().to_dict() for ds, m in splits.items()
    }
    return {
        "version": 1,
        "seed": seed,
        "created": _dt.date.today().isoformat(),
        "rules": __doc__.split("Rules (all grouped by ``pair_id``):")[1].split("``grid_sub``")[0].strip(),
        "pair_counts": counts,
        "grid_sub": grid_sub,
        "splits": splits,
    }


def save_splits(doc: dict) -> None:
    write_json(doc, SPLITS_PATH)
    logger.info("Saved splits to %s", SPLITS_PATH)


def load_splits() -> dict:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} missing; run `python -m src.cli prepare` first")
    return read_json(SPLITS_PATH)


def assign(df: pd.DataFrame, doc: dict | None = None) -> pd.Series:
    """Split name for every row of a canonical frame. Rows whose pair is not in the splits get 'unassigned'."""
    doc = doc or load_splits()
    out = pd.Series("unassigned", index=df.index, dtype=object)
    for ds, mapping in doc["splits"].items():
        m = df["dataset"] == ds
        out[m] = df.loc[m, "pair_id"].map(mapping).fillna("unassigned")
    return out


def in_grid_sub(df: pd.DataFrame, doc: dict | None = None) -> pd.Series:
    doc = doc or load_splits()
    keep = set().union(*(set(v) for v in doc["grid_sub"].values()))
    return (df["dataset"] == "phantom") & df["pair_id"].isin(keep)
