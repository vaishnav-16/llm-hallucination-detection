"""File IO for datasets, predictions and metrics.

Two roots:
- ``data/``      regenerated datasets with full text (git-ignored, large).
- ``artifacts/`` splits, id+score predictions, metrics and figures (committed, small).

Parquet files are written with zstd compression.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import ARTIFACTS_DIR, DATA_DIR

logger = logging.getLogger(__name__)


def write_table(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame as zstd-compressed parquet (or CSV if the suffix is .csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False, compression="zstd")
    logger.info("Wrote %s rows to %s", len(df), path)
    return path


def read_table(path: Path | str) -> pd.DataFrame:
    """Read a parquet or CSV table written by write_table."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def write_json(obj: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)
    return path


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_path(name: str) -> Path:
    """Path of a normalized dataset table under data/."""
    return DATA_DIR / "normalized" / f"{name}.parquet"


def predictions_dir() -> Path:
    return ARTIFACTS_DIR / "predictions"


# --- Helpers kept for the Streamlit dashboard (path-based, return None when missing) ---

def read_predictions(path: Path | str) -> Optional[pd.DataFrame]:
    path = Path(path)
    return read_table(path) if path.exists() else None


def read_metrics(path: Path | str) -> Optional[dict]:
    path = Path(path)
    return read_json(path) if path.exists() else None
