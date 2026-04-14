"""Threshold Tuner page — interactive threshold adjustment with live metric updates."""

import streamlit as st
import numpy as np

from src.dashboard.state import SessionState
from src.dashboard.components import (
    metric_cards_row, no_data_message,
    plotly_confusion_matrix, plotly_score_distribution,
)
from src.services.metrics import compute_metrics_at_threshold


def render():
    st.header("Threshold Tuner")
    st.markdown("Adjust the classification threshold and see metrics update in real-time.")

    hhem_df = SessionState.get_hhem_predictions()
    if hhem_df is None:
        no_data_message("HHEM predictions not found.")
        return

    y_true = hhem_df["label"].values
    scores = hhem_df["hhem_score"].values

    # Get optimal threshold for reference
    opt_metrics = SessionState.get_optimal_threshold_metrics()
    opt_threshold = opt_metrics.get("cv_optimal_threshold", 0.5) if opt_metrics else 0.5

    # Threshold slider
    col_slider, col_info = st.columns([3, 1])
    with col_slider:
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.5,
            step=0.01,
            help="Scores below this threshold are classified as hallucinated."
        )
    with col_info:
        st.markdown(f"**Current:** {threshold:.2f}")
        st.markdown(f"**Baseline:** 0.50")
        st.markdown(f"**CV-Optimal:** {opt_threshold:.2f}")

    # Quick preset buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Baseline (0.50)"):
            threshold = 0.50
    with col2:
        if st.button(f"CV-Optimal ({opt_threshold:.2f})"):
            threshold = opt_threshold
    with col3:
        if st.button("High Precision (0.30)"):
            threshold = 0.30
    with col4:
        if st.button("High Recall (0.70)"):
            threshold = 0.70

    st.divider()

    # Compute metrics at current threshold
    metrics = compute_metrics_at_threshold(y_true, scores, threshold, label="HHEM")
    baseline_metrics = compute_metrics_at_threshold(y_true, scores, 0.5, label="Baseline")

    # Metrics cards with delta from baseline
    st.subheader(f"Metrics at threshold = {threshold:.2f}")
    metric_cards_row(metrics, baseline=baseline_metrics)

    st.divider()

    # Visualizations
    col_left, col_right = st.columns(2)

    y_pred = np.where(scores > threshold, 0, 1)

    with col_left:
        fig = plotly_confusion_matrix(y_true, y_pred, f"Confusion Matrix (threshold={threshold:.2f})")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        faithful_scores = scores[y_true == 0]
        halluc_scores = scores[y_true == 1]
        fig = plotly_score_distribution(faithful_scores, halluc_scores, threshold,
                                        f"Score Distribution (threshold={threshold:.2f})")
        st.plotly_chart(fig, use_container_width=True)

    # Detailed breakdown
    st.divider()
    st.subheader("Detailed Breakdown")

    col1, col2 = st.columns(2)
    with col1:
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        st.markdown(f"""
        | | Predicted Faithful | Predicted Hallucinated |
        |---|---|---|
        | **Actually Faithful** | TN = {tn} | FP = {fp} |
        | **Actually Hallucinated** | FN = {fn} | TP = {tp} |
        """)

    with col2:
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        youden_j = sensitivity + specificity - 1

        st.markdown(f"""
        - **Sensitivity (TPR):** {sensitivity:.4f}
        - **Specificity (TNR):** {specificity:.4f}
        - **Youden's J:** {youden_j:.4f}
        - **False Positive Rate:** {fp / (fp + tn + 1e-9):.4f}
        - **False Negative Rate:** {fn / (fn + tp + 1e-9):.4f}
        """)
