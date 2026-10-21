"""Results page: one metric for many detectors on one slice and group, against answer-only baselines.

Bars show the stored metrics_long values (threshold-dependent metrics use the validation-derived threshold
``evaluate`` chose); nothing is recomputed here. Baseline reference lines come from the same slice when
available, otherwise from phantom_grid_full (different rows), which the caption states.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard import data, state
from src.dashboard.components import bar_with_ci
from src.dashboard.config import metric_name

THRESHOLD_METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}


def render() -> None:
    st.header("Results by slice")
    long = state.metrics_long()
    if long.empty:
        st.warning("artifacts/metrics/metrics_long.csv is missing or empty.")
        return

    c1, c2, c3, c4 = st.columns(4)
    datasets = data.ordered_datasets(long["slice"].unique())
    dataset = c1.selectbox("Dataset", datasets)
    slices = sorted(long.loc[long["dataset"] == dataset, "slice"].unique())
    slice_name = c2.selectbox("Slice", slices)
    on_slice = long[long["slice"] == slice_name]
    groups = sorted(on_slice["group"].unique(), key=data.group_sort_key)
    group = c3.selectbox("Group", groups, index=groups.index("all") if "all" in groups else 0)
    metrics = sorted(on_slice["metric"].unique())
    metric = c4.selectbox("Metric", metrics, index=metrics.index("auroc") if "auroc" in metrics else 0,
                          format_func=metric_name, key="res_metric")

    rows = on_slice[(on_slice["group"] == group) & (on_slice["metric"] == metric)]
    # Keys carry the option set: page-local state that still resets when the slice changes the options.
    all_dets = sorted(rows["detector"].unique())
    default_dets = [d for d in all_dets if not data.is_baseline(d)] or all_dets
    detectors = st.multiselect("Detectors", all_dets, default=default_dets, key=f"res_dets|{','.join(all_dets)}",
                               help="Answer-only baselines are drawn as reference lines; select them to also show bars.")
    strat_options = sorted(rows.loc[rows["detector"].isin(detectors), "strategy"].unique())
    strategies = st.multiselect("Strategies", strat_options, default=strat_options, key=f"res_strats|{','.join(strat_options)}")

    df = data.results_frame(long, slice_name, group, metric, detectors, strategies)
    if df.empty:
        st.info("No rows for this selection.")
        return

    base, source = data.baseline_reference(long, slice_name, group, metric)
    st.plotly_chart(bar_with_ci(df, metric, base if source else None), width="stretch")

    notes = []
    if source is None:
        notes.append("No answer-only baseline is available for this slice and group.")
    elif source != slice_name:
        notes.append(f"Baseline lines come from **{source}** (same group `{group}`), because the baselines were not "
                     f"run on {slice_name}: these are different rows, so the comparison is indicative, not paired.")
    if metric in THRESHOLD_METRICS:
        srcs = ", ".join(sorted(df["threshold_source"].unique()))
        notes.append(f"{metric_name(metric)} is computed at the validation-derived threshold (source: {srcs}).")
    if not df["ci_lo"].notna().any():
        notes.append("No bootstrap CI is stored for this metric.")
    for n in notes:
        st.caption(n)

    table = df[["detector", "strategy", "n", "n_pairs", "value", "ci_lo", "ci_hi", "threshold", "threshold_source"]]
    st.dataframe(table.style.format({"value": "{:.3f}", "ci_lo": "{:.3f}", "ci_hi": "{:.3f}", "threshold": "{:.3f}"},
                                    na_rep="-"), hide_index=True, width="stretch")
    if source:
        st.caption(f"Answer-only baselines ({source})")
        st.dataframe(base[["detector", "strategy", "n", "value", "ci_lo", "ci_hi"]], hide_index=True, width="stretch")
