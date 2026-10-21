"""`prepare` stage: download, normalize and validate every dataset, then build splits and the data card.

Outputs:
- ``data/normalized/*.parquet``          full-text canonical tables (git-ignored)
- ``artifacts/splits.json``              leak-free pair-level splits (committed; never silently rebuilt)
- ``artifacts/data_card.{md,json}``      statistics and tokenizer measurements (committed)
"""

from __future__ import annotations

import datetime as _dt
import logging

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR
from src.data import phantom, splits
from src.data.card import cell_stats, render_markdown, token_lengths
from src.data.halueval import load_halueval
from src.data.ragtruth import load_ragtruth
from src.data.store import read_table, write_json, write_table

logger = logging.getLogger(__name__)

NORM_DIR = DATA_DIR / "normalized"


def phantom_path(cell: phantom.Cell):
    return NORM_DIR / "phantom" / f"{cell.key}.parquet"


def prepare(rebuild_splits: bool = False, measure_tokens: bool = True, filings: tuple[str, ...] = tuple(phantom.FILINGS)) -> dict:
    """Run the whole prepare stage. Returns the data-card dict."""
    slices: dict[str, dict] = {}

    logger.info("Loading HaluEval QA...")
    hal = load_halueval("original")
    write_table(hal, NORM_DIR / "halueval.parquet")
    slices["halueval"] = cell_stats(hal)
    from src.data.normalize_answers import REWRITES_PATH

    if REWRITES_PATH.exists():  # rewrites are committed; rebuild the git-ignored table on every machine
        hal_norm = load_halueval("normalized")
        write_table(hal_norm, NORM_DIR / "halueval_normalized.parquet")
        slices["halueval_normalized"] = cell_stats(hal_norm)

    logger.info("Loading RAGTruth...")
    rt_train, rt_test = load_ragtruth("train"), load_ragtruth("test")
    write_table(pd.concat([rt_train, rt_test], ignore_index=True), NORM_DIR / "ragtruth.parquet")
    for name, d in (("ragtruth_train", rt_train), ("ragtruth_test", rt_test)):
        slices[name] = cell_stats(d)

    logger.info("Loading PHANTOM (%d cells)...", len(phantom.all_cells(filings)))
    seeds: dict[str, pd.DataFrame] = {}
    cell_pairs: dict[str, list[set[str]]] = {f: [] for f in filings}
    long_samples: dict[str, list[str]] = {}
    for cell in phantom.all_cells(filings):
        if cell.position == "seed":
            df = phantom.load_seed(cell.filing)
            seeds[cell.filing] = df
        else:
            df = phantom.load_long(cell, seeds[cell.filing])
            cell_pairs[cell.filing].append(set(df["pair_id"]))
            if cell.position == "middle":
                long_samples[cell.key] = df["context"].sample(min(20, len(df)), random_state=0).tolist()
        write_table(df, phantom_path(cell))
        slices[cell.key] = cell_stats(df)
        if "join_counts" in df.attrs:
            slices[cell.key]["join"] = df.attrs["join_counts"]

    if splits.SPLITS_PATH.exists() and not rebuild_splits:
        doc = splits.load_splits()
        logger.info("Keeping existing %s (pass --rebuild-splits to regenerate)", splits.SPLITS_PATH)
    else:
        doc = splits.build_splits(hal, rt_train, rt_test, seeds, cell_pairs)
        splits.save_splits(doc)
    _check_coverage(doc, hal, rt_train, rt_test, seeds)

    card = {"created": _dt.date.today().isoformat(), "slices": slices, "split_counts": doc["pair_counts"], "tokens": {}}
    if measure_tokens:
        logger.info("Measuring context lengths under each tokenizer...")
        for f, d in seeds.items():
            card["tokens"][f"phantom_{f}_seed"] = token_lengths(d.drop_duplicates("pair_id")["context"].tolist())
        for key, texts in long_samples.items():
            card["tokens"][key] = token_lengths(texts)
        card["tokens"]["halueval"] = token_lengths(hal.drop_duplicates("pair_id")["context"].tolist())
        card["tokens"]["ragtruth_test"] = token_lengths(rt_test.drop_duplicates("pair_id")["context"].tolist())
    write_json(card, ARTIFACTS_DIR / "data_card.json")
    (ARTIFACTS_DIR / "data_card.md").write_text(render_markdown(card), encoding="utf-8")
    logger.info("Wrote data card to %s", ARTIFACTS_DIR / "data_card.md")
    return card


def _check_coverage(doc: dict, hal: pd.DataFrame, rt_train: pd.DataFrame, rt_test: pd.DataFrame, seeds: dict) -> None:
    """Every pair in the freshly loaded data must have a split; otherwise the committed splits are stale."""
    frames = [hal, rt_train, rt_test, *seeds.values()]
    missing = 0
    for d in frames:
        missing += int((splits.assign(d, doc) == "unassigned").sum())
    if missing:
        raise RuntimeError(f"{missing} rows have no split in {splits.SPLITS_PATH}; data changed? use --rebuild-splits")


def load_normalized(name: str) -> pd.DataFrame:
    """Load a normalized table written by prepare ('halueval', 'ragtruth', or a PHANTOM cell key)."""
    if name.startswith("phantom_"):
        path = NORM_DIR / "phantom" / f"{name}.parquet"
    else:
        path = NORM_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m src.cli prepare`")
    return read_table(path)
