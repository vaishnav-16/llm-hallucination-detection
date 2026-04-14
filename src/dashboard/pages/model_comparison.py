"""Model Comparison page — side-by-side evaluation of all models."""

import streamlit as st
import numpy as np

from src.dashboard.state import SessionState
from src.dashboard.components import (
    no_data_message, plotly_model_comparison_bar, plotly_roc_curves,
    metric_cards_row,
)


def render():
    st.header("Model Comparison")
    st.markdown("Compare HHEM, fine-tuned DeBERTa, and ensemble models side by side.")

    hhem_df = SessionState.get_hhem_predictions()
    if hhem_df is None:
        no_data_message("HHEM predictions not found.")
        return

    ensemble_metrics = SessionState.get_ensemble_metrics()
    baseline_metrics = SessionState.get_baseline_metrics()
    ft_metrics = SessionState.get_finetuned_metrics()

    # Model selector
    available_models = ["HHEM Baseline"]
    if ft_metrics:
        available_models.append("Fine-tuned DeBERTa")
    if ensemble_metrics:
        available_models.extend([k for k in ensemble_metrics.keys()])

    selected = st.multiselect("Select models to compare", available_models,
                               default=available_models[:3])

    if not selected:
        st.info("Select at least one model to compare.")
        return

    st.divider()

    # Build comparison metrics
    comparison = {}
    if "HHEM Baseline" in selected and baseline_metrics:
        comparison["HHEM Baseline"] = baseline_metrics
    if "Fine-tuned DeBERTa" in selected and ft_metrics:
        comparison["Fine-tuned DeBERTa"] = ft_metrics
    if ensemble_metrics:
        for name in selected:
            if name in ensemble_metrics:
                comparison[name] = ensemble_metrics[name]

    if comparison:
        # Bar chart
        fig = plotly_model_comparison_bar(comparison, "Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ROC curves
        y_true = hhem_df["label"].values
        hhem_scores = hhem_df["hhem_score"].values
        hhem_probs = 1.0 - hhem_scores

        roc_models = {}
        if "HHEM Baseline" in selected:
            roc_models["HHEM"] = hhem_probs

        ft_df = SessionState.get_finetuned_predictions()
        if ft_df is not None and "Fine-tuned DeBERTa" in selected:
            if "predicted_prob" in ft_df.columns:
                roc_models["Fine-tuned DeBERTa"] = ft_df["predicted_prob"].values

        if len(roc_models) > 1:
            # Add ensemble if both available
            if "HHEM" in roc_models and "Fine-tuned DeBERTa" in roc_models:
                roc_models["Ensemble (avg)"] = (roc_models["HHEM"] + roc_models["Fine-tuned DeBERTa"]) / 2

        if roc_models:
            fig = plotly_roc_curves(y_true, roc_models, "ROC Curves")
            st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        st.divider()
        st.subheader("Detailed Metrics")

        rows = []
        for name, m in comparison.items():
            rows.append({
                "Model": name,
                "Accuracy": m.get("accuracy", m.get("accuracy", 0)),
                "Precision": m.get("precision_hallucinated", m.get("precision", 0)),
                "Recall": m.get("recall_hallucinated", m.get("recall", 0)),
                "F1": m.get("f1_hallucinated", m.get("f1", 0)),
                "ROC AUC": m.get("roc_auc", "N/A"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Decision policy section
    policy_metrics = SessionState.get_policy_metrics()
    if policy_metrics:
        st.divider()
        st.subheader("Decision Policy Results")
        st.markdown("3-tier confidence policy: ANSWER / CAVEAT / ABSTAIN")

        rows = []
        for name, m in policy_metrics.items():
            rows.append({
                "Tolerance": name,
                "Coverage": f"{m['coverage']:.1%}",
                "Answer %": f"{m['pct_answer']:.1%}",
                "Caveat %": f"{m['pct_caveat']:.1%}",
                "Abstain %": f"{m['pct_abstain']:.1%}",
                "Halluc Rate": f"{m['halluc_rate_in_answers']:.3%}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
