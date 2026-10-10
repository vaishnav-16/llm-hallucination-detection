"""Named evaluation slices. A slice is a list of *parts*, and a part is the unit of resumable work.

For PHANTOM each part is one cell (filing x length x position), so an interrupted 30K-token run
restarts at the cell it was in rather than from scratch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.data import phantom, splits
from src.data.prepare import load_normalized

HARD_FAITHFUL_CHARS = 224  # shortest hallucinated answer in the PHANTOM 10-K seed; see data card


@dataclass(frozen=True)
class Part:
    slice_name: str
    part: str
    loader: Callable[[], pd.DataFrame]


def _with_meta(df: pd.DataFrame, doc: dict) -> pd.DataFrame:
    df = df.copy()
    df["split"] = splits.assign(df, doc)
    faithful_chars = df[df["label"] == 0].groupby("pair_id")["response"].apply(lambda s: s.str.len().max())
    df["faithful_chars"] = df["pair_id"].map(faithful_chars).fillna(0).astype(int)
    return df


def _phantom_cells(filings: tuple[str, ...], include_seed: bool = True, include_long: bool = True) -> list[phantom.Cell]:
    cells = []
    for c in phantom.all_cells(filings):
        if (c.position == "seed" and include_seed) or (c.position != "seed" and include_long):
            cells.append(c)
    return cells


def get_slice(name: str, max_pairs: int | None = None) -> list[Part]:
    """Parts for a named slice. ``max_pairs`` keeps the first N pairs per part (smoke tests)."""
    doc = splits.load_splits()

    def limit(df: pd.DataFrame) -> pd.DataFrame:
        if max_pairs is None:
            return df
        keep = sorted(df["pair_id"].unique())[:max_pairs]
        return df[df["pair_id"].isin(keep)]

    def table(key: str, split: str | None = None, sub: bool = False, source: str | None = None) -> Callable[[], pd.DataFrame]:
        def load() -> pd.DataFrame:
            df = _with_meta(load_normalized(key), doc)
            if source is not None:
                df = df[df["source_file"] == source]
            if split is not None:
                df = df[df["split"] == split]
            if sub:
                df = df[splits.in_grid_sub(df, doc)]
            return limit(df).reset_index(drop=True)
        return load

    if name.startswith("cell-"):  # one PHANTOM test cell, e.g. cell-phantom_10k_30000_middle (smoke tests)
        key = name[len("cell-"):]
        return [Part(name, key, table(key, "test"))]
    m = re.fullmatch(r"phantom_grid_(sub|full)_min(\d+)", name)  # e.g. phantom_grid_sub_min20000: long cells only
    if m:
        cells = [c for c in _phantom_cells(phantom.TEST_FILINGS, include_seed=False) if c.target_tokens >= int(m.group(2))]
        return [Part(name, c.key, table(c.key, "test", sub=(m.group(1) == "sub"))) for c in cells]
    if name in ("phantom_grid_sub", "phantom_grid_full", "phantom_seed_test"):
        cells = _phantom_cells(phantom.TEST_FILINGS, include_long=(name != "phantom_seed_test"))
        return [Part(name, c.key, table(c.key, "test", sub=(name == "phantom_grid_sub"))) for c in cells]
    if name in ("phantom_val", "phantom_train"):
        split = name.split("_")[1]
        return [Part(name, c.key, table(c.key, split)) for c in _phantom_cells(phantom.TRAIN_FILINGS, include_long=False)]
    for ds in ("halueval", "ragtruth"):
        for split in ("train", "val", "test"):
            if name == f"{ds}_{split}":
                return [Part(name, name, table(ds, split))]
    if name.startswith("halueval_norm_"):
        split = name.rsplit("_", 1)[1]
        return [Part(name, name, table("halueval_normalized", split))]
    raise KeyError(f"unknown slice {name!r}")


SLICE_NAMES = [
    "phantom_grid_sub", "phantom_grid_full", "phantom_seed_test", "phantom_val", "phantom_train",
    "halueval_train", "halueval_val", "halueval_test", "halueval_norm_train", "halueval_norm_val", "halueval_norm_test",
    "ragtruth_train", "ragtruth_val", "ragtruth_test",
]

META_COLUMNS = ["example_id", "pair_id", "dataset", "variant", "filing", "target_tokens", "position", "task_type", "split", "label", "faithful_chars"]
