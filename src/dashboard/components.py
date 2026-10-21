"""Plotly figures and small widgets shared by the dashboard pages.

Figures take already-filtered frames from ``src.dashboard.data`` so the chart code holds no selection
logic. Answer-only baselines are always drawn the same way (grey, dashed or dotted) so a reader can tell
at a glance whether a detector beats a model that never looked at the context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.config import BASELINE_DASH, COLORS, metric_name


def _has_ci(df: pd.DataFrame) -> bool:
    return {"ci_lo", "ci_hi"} <= set(df.columns) and df["ci_lo"].notna().any() and df["ci_hi"].notna().any()


def _baseline_name(row: pd.Series) -> str:
    return f"{row['detector']} (answer-only)"


def bar_with_ci(df: pd.DataFrame, metric: str, baselines: pd.DataFrame | None = None) -> go.Figure:
    """Bars per detector x strategy with 95% CI error bars and horizontal baseline reference lines."""
    fig = go.Figure()
    error_y = None
    if _has_ci(df):
        error_y = dict(type="data", symmetric=False, array=(df["ci_hi"] - df["value"]).to_numpy(),
                       arrayminus=(df["value"] - df["ci_lo"]).to_numpy())
    fig.add_trace(go.Bar(
        x=df["label"], y=df["value"], error_y=error_y, marker_color=COLORS["faithful"],
        customdata=np.stack([df["detector"], df["strategy"], df["n"]], axis=-1) if len(df) else None,
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>value %{y:.3f}<br>n=%{customdata[2]}<extra></extra>",
        name="detectors", showlegend=False,
    ))
    if baselines is not None:
        for _, row in baselines.iterrows():
            fig.add_hline(
                y=row["value"], line_dash=BASELINE_DASH.get(row["detector"], "dash"), line_color=COLORS["baseline"],
                annotation_text=f"{_baseline_name(row)}: {row['value']:.3f}", annotation_position="top left",
            )
    fig.update_layout(yaxis_title=metric_name(metric), xaxis_title=None, height=480, margin=dict(t=30, b=10),
                      xaxis_tickangle=-30)
    if metric in ("auroc", "balanced_accuracy"):
        fig.add_hline(y=0.5, line_color="#BBBBBB", line_width=1, annotation_text="chance", annotation_position="bottom right")
    return fig


def _band(fig: go.Figure, x: pd.Series, lo: pd.Series, hi: pd.Series, color: str, group: str) -> None:
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=color, opacity=0.15,
        line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=group,
    ))


_PALETTE = ["#4A90D9", "#E67E22", "#27AE60", "#8E44AD", "#C0392B", "#16A085", "#D35400", "#2C3E50",
            "#F1C40F", "#7F8C8D", "#E84393", "#00B894"]


def line_with_ci(df: pd.DataFrame, x: str, metric: str, baselines: pd.DataFrame | None = None,
                 log_x: bool = False, x_title: str | None = None) -> go.Figure:
    """One line per ``label`` with a shaded CI band; baselines as grey dashed lines."""
    fig = go.Figure()
    for i, (label, g) in enumerate(df.groupby("label", sort=True)):
        color = _PALETTE[i % len(_PALETTE)]
        g = g.sort_values(x)
        xs = g[x].astype(str) if not log_x else g[x]
        if _has_ci(g):
            _band(fig, xs, g["ci_lo"], g["ci_hi"], color, label)
        fig.add_trace(go.Scatter(x=xs, y=g["value"], mode="lines+markers", name=label, line=dict(color=color),
                                 legendgroup=label, hovertemplate=f"{label}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    if baselines is not None and len(baselines):
        for det, g in baselines.groupby("detector"):
            g = g.sort_values(x)
            xs = g[x].astype(str) if not log_x else g[x]
            fig.add_trace(go.Scatter(x=xs, y=g["value"], mode="lines", name=f"{det} (answer-only)",
                                     line=dict(color=COLORS["baseline"], dash=BASELINE_DASH.get(det, "dash"))))
    fig.update_layout(yaxis_title=metric_name(metric), xaxis_title=x_title or x, height=460, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.18))
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def score_histogram(preds: pd.DataFrame, val_thr: float, explore_thr: float | None = None) -> go.Figure:
    """Overlaid histograms of p_halluc for faithful vs hallucinated rows, with threshold markers."""
    fig = go.Figure()
    for label, name, color in ((0, "faithful (label 0)", COLORS["faithful"]), (1, "hallucinated (label 1)", COLORS["hallucinated"])):
        fig.add_trace(go.Histogram(x=preds.loc[preds["label"] == label, "p_halluc"], name=name, marker_color=color,
                                   opacity=0.6, nbinsx=50))
    fig.add_vline(x=val_thr, line_color=COLORS["val_threshold"], line_dash="dash",
                  annotation_text=f"reported threshold {val_thr:.3f}", annotation_position="top")
    if explore_thr is not None and not np.isclose(explore_thr, val_thr):
        fig.add_vline(x=explore_thr, line_color=COLORS["explore_threshold"], line_dash="dot",
                      annotation_text=f"exploration {explore_thr:.3f}", annotation_position="bottom")
    fig.update_layout(barmode="overlay", xaxis_title="p_halluc (higher = more likely hallucinated)", yaxis_title="rows",
                      height=380, margin=dict(t=30, b=10), legend=dict(orientation="h", yanchor="top", y=-0.2))
    return fig


def confusion_heatmap(cm: np.ndarray, title: str) -> go.Figure:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] as an annotated heatmap."""
    labels = ["faithful", "hallucinated"]
    fig = go.Figure(go.Heatmap(z=cm, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels],
                               colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}"))
    fig.update_layout(title=title, height=330, margin=dict(t=50, b=10), yaxis_autorange="reversed")
    return fig
