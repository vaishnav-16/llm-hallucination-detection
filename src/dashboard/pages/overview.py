"""Overview page: what the project measures and the headline AUROC table.

The headline table is rebuilt from metrics_long (not headline.csv) so it always reflects the current
artifacts, and answer-only baselines are listed first and shaded: on several benchmarks they are the
bar every context-reading detector has to clear.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.dashboard import data, state
from src.dashboard.config import RESEARCH_QUESTION

logger = logging.getLogger(__name__)


def _shade_baselines(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def style(row: pd.Series) -> list[str]:
        css = "background-color: rgba(127,140,141,0.25); font-style: italic" if row["type"] == "answer-only baseline" else ""
        return [css] * len(row)
    return df.style.apply(style, axis=1)


def render() -> None:
    st.header("FastHalluCheck: hallucination detectors on QA benchmarks")
    st.markdown(RESEARCH_QUESTION)

    long = state.metrics_long()
    runs = state.runs()
    if long.empty:
        st.warning("artifacts/metrics/metrics_long.csv is missing or empty; run the evaluate stage first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction runs", len(runs))
    c2.metric("Detectors", long["detector"].nunique())
    c3.metric("Detector x strategy", long[["detector", "strategy"]].drop_duplicates().shape[0])
    c4.metric("Test slices evaluated", long["slice"].nunique())

    st.subheader("Headline AUROC [95% CI]")
    st.caption("Columns are test slices (`all` rows) plus the length-matched hard subset where it exists. "
               "Shaded rows are answer-only baselines: they score the answer without the context.")
    table = data.headline_table(long, "auroc")
    st.dataframe(_shade_baselines(table), hide_index=True, width="stretch")

    figs = sorted(data.figures_dir().glob("*.png"))
    if figs:
        with st.expander(f"Report figures ({len(figs)})"):
            for f in figs:
                st.image(str(f), caption=f.name)
