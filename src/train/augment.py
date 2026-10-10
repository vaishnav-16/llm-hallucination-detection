"""Distractor augmentation: embed a training context among unrelated training passages.

This imitates PHANTOM's long-context construction (the evidence passage sits at the beginning, middle
or end of a longer document) using only training-split data, so no test filing is ever touched.
Lengths are approximated with characters / 4 to avoid re-tokenizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEED

CHARS_PER_TOKEN = 4


def add_distractors(
    df: pd.DataFrame,
    fraction: float = 0.5,
    target_tokens: tuple[int, ...] = (2000, 3500),
    seed: int = SEED,
) -> pd.DataFrame:
    """Return ``df`` plus augmented copies of a ``fraction`` of its rows with padded contexts."""
    rng = np.random.default_rng(seed)
    pool_ids = df["pair_id"].to_numpy()
    pool_ctx = df["context"].to_numpy()
    pick = rng.random(len(df)) < fraction
    rows = []
    for i in np.flatnonzero(pick):
        r = df.iloc[i]
        target_chars = int(rng.choice(target_tokens)) * CHARS_PER_TOKEN
        filler: list[str] = []
        size = len(r["context"])
        while size < target_chars:
            j = int(rng.integers(len(df)))
            if pool_ids[j] == r["pair_id"]:
                continue
            filler.append(pool_ctx[j])
            size += len(pool_ctx[j]) + 2
        pos = rng.choice(["beginning", "middle", "end"])
        if pos == "beginning":
            parts = [r["context"], *filler]
        elif pos == "end":
            parts = [*filler, r["context"]]
        else:
            k = len(filler) // 2
            parts = [*filler[:k], r["context"], *filler[k:]]
        new = r.copy()
        new["context"] = "\n\n".join(parts)
        new["example_id"] = f"{r['example_id']}_aug"
        new["position"] = f"aug_{pos}"
        rows.append(new)
    if not rows:
        return df
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
