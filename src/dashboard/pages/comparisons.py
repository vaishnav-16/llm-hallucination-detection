"""Comparisons & seeds page: paired differences from comparisons.csv and seed variability from seed_summary.csv.

Paired comparisons (same rows, bootstrap over pairs) are the right way to say one configuration beats
another; independent CIs on the Results page overlap more than paired differences do.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import data, state

ALL = "(all)"


def _pick(col, label: str, values: pd.Series, default: str | None = None) -> str:
    options = [ALL, *sorted(values.dropna().astype(str).unique(), key=data.group_sort_key)]
    return col.selectbox(label, options, index=options.index(default) if default in options else 0)


def render() -> None:
    st.header("Paired comparisons")
    comp = state.comparisons()
    if comp.empty:
        st.info("artifacts/metrics/comparisons.csv is missing or empty.")
    else:
        st.caption("`diff` = metric(a) - metric(b) on the same rows; CI and p-value from a paired, pair-clustered bootstrap. "
                   "`a` is a strategy of `detector`; `b` is the reference (detector:strategy).")
        c1, c2, c3, c4 = st.columns(4)
        det = _pick(c1, "Detector", comp["detector"])
        sl = _pick(c2, "Slice", comp["slice"])
        grp = _pick(c3, "Group", comp["group"], default="all")
        met = _pick(c4, "Metric", comp["metric"], default="auroc")
        df = comp
        for col, val in (("detector", det), ("slice", sl), ("group", grp), ("metric", met)):
            if val != ALL:
                df = df[df[col].astype(str) == val]
        df = df.assign(significant=df["p_value"] < 0.05)
        st.dataframe(df.style.format({"diff": "{:+.3f}", "ci_lo": "{:+.3f}", "ci_hi": "{:+.3f}", "p_value": "{:.3g}"}, na_rep="-"),
                     hide_index=True, width="stretch")
        st.caption(f"{len(df)} of {len(comp)} comparisons shown.")

    st.header("Seed variability")
    seeds = state.seed_summary()
    if seeds.empty:
        st.info("seed_summary.csv is empty: no detector has results for more than one training seed yet "
                "(runs named <base>_s<seed>, e.g. modernbert_ft_s42/s43/s44).")
    else:
        st.dataframe(seeds.style.format({"mean": "{:.3f}", "sd": "{:.3f}"}, na_rep="-"), hide_index=True, width="stretch")
