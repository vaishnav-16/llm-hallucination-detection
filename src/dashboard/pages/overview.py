"""Overview page — project summary, key metrics, and model comparison."""

import streamlit as st
import numpy as np

from src.dashboard.state import SessionState
from src.dashboard.components import metric_cards_row, no_data_message, plotly_confusion_matrix, plotly_score_distribution
from src.dashboard.config import COLORS


def render():
    st.header("Project Overview")
    st.markdown("""
    **FastHalluCheck** evaluates pretrained hallucination detection models against
    benchmark QA datasets. This dashboard provides interactive exploration of results.
    """)

    # Check data availability
    hhem_df = SessionState.get_hhem_predictions()
    if hhem_df is None:
        no_data_message("HHEM predictions not found. Run the pipeline to generate results.")
        return

    metrics = SessionState.get_baseline_metrics()
    ft_metrics = SessionState.get_finetuned_metrics()
    ensemble_metrics = SessionState.get_ensemble_metrics()

    # Key Metrics Cards
    st.subheader("HHEM Baseline Performance")
    metric_cards_row(metrics)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", metrics.get("total_samples", len(hhem_df)))
    with col2:
        roc = metrics.get("roc_auc")
        st.metric("ROC AUC", f"{roc:.4f}" if roc else "N/A")
    with col3:
        opt = SessionState.get_optimal_threshold_metrics()
        if opt:
            st.metric("Optimal Threshold", f"{opt.get('cv_optimal_threshold', 0.5):.2f}")

    st.divider()

    # Two-column layout: confusion matrix + score distribution
    col_left, col_right = st.columns(2)

    y_true = hhem_df["label"].values
    y_pred = hhem_df["predicted_label"].values
    scores = hhem_df["hhem_score"].values

    with col_left:
        fig = plotly_confusion_matrix(y_true, y_pred, "HHEM Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        faithful_scores = scores[y_true == 0]
        halluc_scores = scores[y_true == 1]
        fig = plotly_score_distribution(faithful_scores, halluc_scores,
                                        title="HHEM Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Key Findings
    st.divider()
    st.subheader("Key Findings")

    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        **Error Summary**
        - False Positives: **{fp.sum()}**
        - False Negatives: **{fn.sum()}**
        - Error Rate: **{(fp.sum() + fn.sum()) / len(y_true):.1%}**
        """)
    with col2:
        st.markdown(f"""
        **Score Statistics**
        - Faithful mean: **{scores[y_true == 0].mean():.4f}**
        - Hallucinated mean: **{scores[y_true == 1].mean():.4f}**
        - Score gap: **{scores[y_true == 0].mean() - scores[y_true == 1].mean():.4f}**
        """)
    with col3:
        if ft_metrics:
            ft_f1 = ft_metrics.get("f1_hallucinated", 0)
            base_f1 = metrics.get("f1_hallucinated", 0)
            st.markdown(f"""
            **Fine-tuned DeBERTa**
            - F1: **{ft_f1:.4f}** ({ft_f1 - base_f1:+.4f} vs baseline)
            - Accuracy: **{ft_metrics.get('accuracy', 0):.4f}**
            """)
        else:
            st.markdown("*Fine-tuned model results not available.*")

    # Model comparison table
    if ensemble_metrics:
        st.divider()
        st.subheader("Model Comparison Summary")

        rows = []
        for name, m in ensemble_metrics.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "Precision": f"{m.get('precision', 0):.4f}",
                "Recall": f"{m.get('recall', 0):.4f}",
                "F1": f"{m.get('f1', 0):.4f}",
                "ROC AUC": f"{m.get('roc_auc', 'N/A')}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
