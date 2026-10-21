"""Predictions page: inspect one run's scores, its confusion matrix and its worst errors.

The reported threshold is the one ``evaluate`` fitted on the matching validation slice (read from
metrics_long), never one tuned on the test labels shown here. The slider exists only to explore how the
trade-off moves; its value is never what the report quotes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import data, state
from src.dashboard.components import confusion_heatmap, score_histogram
from src.dashboard.config import SLIDER_NOTE

SUMMARY_KEYS = [("balanced_accuracy", "Balanced acc."), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
                ("accuracy", "Accuracy")]


def _metric_row(summary: dict[str, float], ref: dict[str, float] | None = None) -> None:
    cols = st.columns(len(SUMMARY_KEYS))
    for col, (key, name) in zip(cols, SUMMARY_KEYS):
        delta = None if ref is None else f"{summary[key] - ref[key]:+.3f}"
        col.metric(name, f"{summary[key]:.3f}", delta=delta)


def _error_table(title: str, rows: pd.DataFrame, slice_name: str, with_text: bool) -> None:
    st.markdown(f"**{title}** ({len(rows)} shown)")
    if rows.empty:
        st.caption("None at this threshold.")
        return
    shown = rows
    if with_text:
        joined = state.text_for(rows, slice_name)
        if joined is None:
            st.caption("Question/response text is unavailable: it lives in the git-ignored data/ directory, "
                       "which is missing or incomplete on this machine. Showing metadata only.")
        else:
            front = ["question", "response", "p_halluc", "context"]
            shown = joined[front + [c for c in joined.columns if c not in front]]
    st.dataframe(shown, hide_index=True, width="stretch")


def render() -> None:
    st.header("Predictions")
    runs = state.runs()
    if runs.empty:
        st.warning("No prediction files under artifacts/predictions/.")
        return
    long = state.metrics_long()

    c1, c2, c3 = st.columns(3)
    detector = c1.selectbox("Detector", sorted(runs["detector"].unique()))
    r_det = runs[runs["detector"] == detector]
    strategy = c2.selectbox("Strategy", sorted(r_det["strategy"].unique()))
    r_str = r_det[r_det["strategy"] == strategy]
    slices = sorted(r_str["slice"].unique(), key=lambda s: (not s.endswith("_test") and "_grid_" not in s, s))
    slice_name = c3.selectbox("Slice", slices)
    run_id = r_str.loc[r_str["slice"] == slice_name, "run_id"].iloc[0]

    meta = state.manifests().get(run_id)
    with st.expander(f"Run metadata: {run_id}", expanded=False):
        if meta is None:
            st.caption("No manifest record for this run (answer-only baselines are written without one).")
        else:
            st.code(json.dumps(meta, indent=2), language="json")

    preds = state.predictions(run_id)
    y, p = preds["label"].to_numpy(), preds["p_halluc"].to_numpy()
    st.caption(f"{len(preds):,} rows, {preds['pair_id'].nunique():,} pairs, "
               f"{int((y == 1).sum()):,} hallucinated / {int((y == 0).sum()):,} faithful.")

    thr, source = data.val_threshold(long, detector, strategy, slice_name)
    if thr is None:
        thr = data.DEFAULT_THRESHOLD
        st.caption(f"No stored validation threshold for this run (only test slices are evaluated); using {thr:.2f}.")
    elif source == "default":
        st.caption(f"evaluate had no validation predictions for this run and used its default threshold {thr:.3f}.")
    elif source == slice_name:
        st.caption(f"Threshold {thr:.3f} is the one evaluate fitted on this validation slice (balanced accuracy).")
    else:
        st.caption(f"Reported threshold {thr:.3f}, fitted on **{source}** by balanced accuracy.")

    lo, hi = float(min(0.0, np.nanmin(p))), float(max(1.0, np.nanmax(p)))
    explore = st.slider(f"Threshold ({SLIDER_NOTE})", min_value=lo, max_value=hi, value=float(thr), step=0.001,
                        format="%.3f", key=f"thr_{run_id}")

    reported = data.threshold_summary(y, p, thr)
    st.markdown(f"**At the reported threshold ({thr:.3f})**")
    _metric_row(reported)
    differs = not np.isclose(explore, thr)
    if differs:
        st.markdown(f"**At the exploration threshold ({explore:.3f})** — {SLIDER_NOTE}")
        _metric_row(data.threshold_summary(y, p, explore), ref=reported)

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(score_histogram(preds, thr, explore if differs else None), width="stretch")
    with right:
        shown_thr = explore if differs else thr
        title = f"Confusion at {'exploration' if differs else 'reported'} threshold {shown_thr:.3f}"
        st.plotly_chart(confusion_heatmap(data.confusion(y, p, shown_thr), title), width="stretch")

    st.subheader("Errors")
    n = st.number_input("Rows per table", min_value=5, max_value=100, value=10, step=5)
    with_text = st.checkbox("Join question/response text from local data/ (not in the artifacts)", value=False)
    fp, fn = data.top_errors(preds, explore, int(n))
    _error_table("False positives: faithful rows with the highest p_halluc", fp, slice_name, with_text)
    _error_table("False negatives: hallucinated rows with the lowest p_halluc", fn, slice_name, with_text)
