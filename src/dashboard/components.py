"""Reusable UI components for the Streamlit dashboard."""

from typing import Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from src.dashboard.config import COLORS


def metric_card(label: str, value, delta: Optional[str] = None, delta_color: str = "normal"):
    """Display a metric card using st.metric."""
    if isinstance(value, float):
        value = f"{value:.4f}"
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def metric_cards_row(metrics: dict, baseline: Optional[dict] = None):
    """Display a row of 4 key metric cards."""
    cols = st.columns(4)
    keys = [
        ("accuracy", "Accuracy"),
        ("precision_hallucinated", "Precision"),
        ("recall_hallucinated", "Recall"),
        ("f1_hallucinated", "F1 Score"),
    ]

    for col, (key, label) in zip(cols, keys):
        val = metrics.get(key, metrics.get(key.replace("_hallucinated", ""), 0))
        delta = None
        if baseline and key in baseline:
            base_val = baseline.get(key, baseline.get(key.replace("_hallucinated", ""), 0))
            if base_val > 0:
                diff = val - base_val
                delta = f"{diff:+.4f}"
        with col:
            metric_card(label, val, delta)


def plotly_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"):
    """Interactive Plotly confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Faithful", "Hallucinated"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale="Blues",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        height=400,
    )
    return fig


def plotly_score_distribution(
    scores_faithful: np.ndarray,
    scores_hallucinated: np.ndarray,
    threshold: float = 0.5,
    title: str = "Score Distribution",
):
    """Interactive Plotly score distribution histogram."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_faithful, nbinsx=40, name="Faithful",
        marker_color=COLORS["faithful"], opacity=0.6,
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=scores_hallucinated, nbinsx=40, name="Hallucinated",
        marker_color=COLORS["hallucinated"], opacity=0.6,
        histnorm="probability density",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="white",
                  annotation_text=f"Threshold={threshold:.2f}")
    fig.update_layout(
        title=title,
        xaxis_title="Consistency Score",
        yaxis_title="Density",
        barmode="overlay",
        height=400,
    )
    return fig


def plotly_roc_curves(y_true: np.ndarray, models: dict, title: str = "ROC Curves"):
    """Interactive Plotly ROC curves for multiple models."""
    from sklearn.metrics import roc_curve, roc_auc_score
    colors = [COLORS["model_hhem"], COLORS["model_deberta"],
              COLORS["model_ensemble"], COLORS["model_phantom"]]

    fig = go.Figure()
    for (name, probs), color in zip(models.items(), colors):
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                line=dict(color=color, width=2),
            ))
        except Exception:
            pass

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
    )
    return fig


def plotly_model_comparison_bar(metrics: dict, title: str = "Model Comparison"):
    """Interactive Plotly bar chart comparing models."""
    names = list(metrics.keys())
    metric_keys = ["accuracy", "precision_hallucinated", "recall_hallucinated", "f1_hallucinated"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig = go.Figure()
    for key, label in zip(metric_keys, labels):
        vals = []
        for name in names:
            m = metrics[name]
            vals.append(m.get(key, m.get(key.replace("_hallucinated", ""), 0)))
        fig.add_trace(go.Bar(name=label, x=names, y=vals, text=[f"{v:.3f}" for v in vals],
                             textposition="auto"))

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_range=[0, 1.1],
        height=500,
    )
    return fig


def data_table(df: pd.DataFrame, max_rows: int = 100):
    """Display a paginated data table."""
    total = len(df)
    if total > max_rows:
        page = st.number_input("Page", min_value=1, max_value=(total // max_rows) + 1, value=1)
        start = (page - 1) * max_rows
        end = min(start + max_rows, total)
        st.caption(f"Showing rows {start+1}-{end} of {total}")
        st.dataframe(df.iloc[start:end], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def no_data_message(message: str = "No data available. Run the pipeline first."):
    """Display a styled warning when data is missing."""
    st.warning(message)
    st.markdown("""
    **To generate results, run:**
    ```bash
    python run_all.py
    ```
    Or use the CLI:
    ```bash
    python -m src.cli.commands run
    ```
    """)
