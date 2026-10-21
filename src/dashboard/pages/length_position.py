"""Length & position page (PHANTOM): does detection degrade as the context grows or the evidence moves?

The pooled curve uses the ``len=<N>`` groups (optionally ``len=<N>|hard``, the length-matched subset where
answer length is not a giveaway). A single filing can be shown via the ``cell=`` groups, faceted by evidence
position because cells are not poolable without a new bootstrap. Answer-only baselines are drawn as grey
dashed curves when they exist for the slice (or for phantom_grid_full as a stated fallback).
"""

from __future__ import annotations

import streamlit as st

from src.dashboard import data, state
from src.dashboard.components import line_with_ci
from src.dashboard.config import metric_name

POOLED = "all filings (pooled)"


def _baseline_caption(source: str | None, slice_name: str) -> None:
    if source is None:
        st.caption("No answer-only baseline is available for this view.")
    elif source != slice_name:
        st.caption(f"Baseline curves come from **{source}** (baselines were not run on {slice_name}); different rows, "
                   "so indicative only.")


def render() -> None:
    st.header("Length & position (PHANTOM)")
    long = state.metrics_long()
    phantom = long[(long["dataset"] == "phantom") & long["group"].str.startswith("len=")]
    if phantom.empty:
        st.info("No PHANTOM length groups in metrics_long.")
        return

    c1, c2, c3 = st.columns(3)
    slices = sorted(phantom["slice"].unique())
    slice_name = c1.selectbox("Slice", slices, index=slices.index("phantom_grid_sub") if "phantom_grid_sub" in slices else 0)
    on_slice = phantom[phantom["slice"] == slice_name]
    metrics = sorted(on_slice["metric"].unique())
    metric = c2.selectbox("Metric", metrics, index=metrics.index("auroc") if "auroc" in metrics else 0, format_func=metric_name, key="len_metric")
    filings = data.filings_with_cells(long, slice_name)
    filing = c3.selectbox("Filing", [POOLED, *filings], help="A single filing uses the per-cell groups, one panel per evidence position.")

    # Keys carry the option set: page-local state that still resets when the slice changes the options.
    all_dets = sorted(on_slice["detector"].unique())
    default_dets = [d for d in all_dets if not data.is_baseline(d)] or all_dets
    detectors = st.multiselect("Detectors", all_dets, default=default_dets, key=f"len_dets|{','.join(all_dets)}")
    strat_options = sorted(on_slice.loc[on_slice["detector"].isin(detectors), "strategy"].unique())
    strategies = st.multiselect("Strategies", strat_options, default=strat_options, key=f"len_strats|{','.join(strat_options)}")
    hard = st.toggle("Hard subset only (len=N|hard: faithful answer at least as long as the hallucinated ones)",
                     value=False, disabled=filing != POOLED)

    st.subheader(f"{metric_name(metric)} vs target context length")
    if filing == POOLED:
        curve = data.length_curve(long, slice_name, metric, detectors, strategies, hard=hard)
        base, source = data.baseline_curve(long, slice_name, metric, "length", hard=hard)
        if curve.empty:
            st.info("No rows for this selection.")
        else:
            st.plotly_chart(line_with_ci(curve, "target_tokens", metric, base, log_x=True, x_title="target tokens (log)"),
                            width="stretch")
            _baseline_caption(source, slice_name)
    else:
        cells = data.cell_curve(long, slice_name, metric, filing, detectors, strategies)
        if cells.empty:
            st.info("No cell rows for this filing.")
        else:
            st.caption("Per-cell values (no pooling). The 500-token point is the seed cell (whole document, no evidence "
                       "position) and is repeated in each panel. The hard toggle does not apply: there are no cell|hard groups.")
            cols = st.columns(cells["position"].nunique())
            for col, (pos, g) in zip(cols, cells.groupby("position", sort=False)):
                with col:
                    st.markdown(f"**evidence: {pos}**")
                    st.plotly_chart(line_with_ci(g, "target_tokens", metric, log_x=True, x_title="target tokens (log)"),
                                    width="stretch", key=f"cell_{pos}")

    st.subheader(f"{metric_name(metric)} by evidence position")
    pos = data.position_frame(long, slice_name, metric, detectors, strategies)
    if pos.empty:
        st.info("No position groups for this slice.")
    else:
        base, source = data.baseline_curve(long, slice_name, metric, "position")
        st.plotly_chart(line_with_ci(pos, "position", metric, base, x_title="evidence position"), width="stretch")
        _baseline_caption(source, slice_name)
        st.caption("Position groups pool all filings and lengths >= 2000 tokens; the 500-token seed cells have no position.")
