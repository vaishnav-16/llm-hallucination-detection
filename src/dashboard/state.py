"""Cached access to the artifacts for the Streamlit pages.

The pure loaders live in ``src.dashboard.data``; this module only adds ``st.cache_data`` so that switching
pages or widgets does not re-read CSV/parquet files. File modification times are part of each cache key,
so re-running ``evaluate`` (which rewrites the artifacts) is picked up without restarting Streamlit.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard import data


@st.cache_data(show_spinner=False)
def _metrics_long(root: str, mtime: float) -> pd.DataFrame:
    return data.load_metrics_long(root)


@st.cache_data(show_spinner=False)
def _comparisons(root: str, mtime: float) -> pd.DataFrame:
    return data.load_comparisons(root)


@st.cache_data(show_spinner=False)
def _seed_summary(root: str, mtime: float) -> pd.DataFrame:
    return data.load_seed_summary(root)


@st.cache_data(show_spinner=False)
def _runs(root: str, mtime: float) -> pd.DataFrame:
    return data.list_runs(root)


@st.cache_data(show_spinner=False)
def _manifests(root: str, mtime: float) -> dict[str, dict[str, Any]]:
    return data.load_manifests(root)


@st.cache_data(show_spinner="Loading predictions...", max_entries=8)
def _predictions(root: str, run_id: str, mtime: float) -> pd.DataFrame:
    return data.load_predictions(run_id, root)


@st.cache_data(show_spinner="Joining text from data/...", max_entries=16)
def _text(rows: pd.DataFrame, slice_name: str) -> pd.DataFrame | None:
    return data.attach_text(rows, slice_name)


def _root() -> str:
    return str(data.artifacts_root())


def metrics_long() -> pd.DataFrame:
    return _metrics_long(_root(), data.file_mtime(data.metrics_dir() / "metrics_long.csv"))


def comparisons() -> pd.DataFrame:
    return _comparisons(_root(), data.file_mtime(data.metrics_dir() / "comparisons.csv"))


def seed_summary() -> pd.DataFrame:
    return _seed_summary(_root(), data.file_mtime(data.metrics_dir() / "seed_summary.csv"))


def runs() -> pd.DataFrame:
    return _runs(_root(), data.file_mtime(data.predictions_dir()))


def manifests() -> dict[str, dict[str, Any]]:
    return _manifests(_root(), data.file_mtime(data.predictions_dir() / "manifest"))


def predictions(run_id: str) -> pd.DataFrame:
    return _predictions(_root(), run_id, data.file_mtime(data.predictions_dir() / f"{run_id}.parquet"))


def text_for(rows: pd.DataFrame, slice_name: str) -> pd.DataFrame | None:
    return _text(rows, slice_name)
