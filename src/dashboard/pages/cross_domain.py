"""Cross-Domain Analysis page — HaluEval vs PHANTOM comparison."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.dashboard.state import SessionState
from src.dashboard.components import no_data_message, metric_cards_row
from src.dashboard.config import COLORS, PATHS


def render():
    st.header("Cross-Domain Analysis")
    st.markdown("""
    Compare model performance across domains: **HaluEval** (general QA) vs **PHANTOM** (financial SEC filings).
    Domain shift is a critical challenge for hallucination detection models.
    """)

    # Load data
    hhem_df = SessionState.get_hhem_predictions()
    phantom_df = SessionState.get_phantom_predictions()
    baseline_metrics = SessionState.get_baseline_metrics()
    phantom_metrics = SessionState.load_metrics(PATHS["phantom_metrics"])

    if hhem_df is None:
        no_data_message("HaluEval predictions not found.")
        return

    # HaluEval metrics
    st.subheader("HaluEval (General QA)")
    if baseline_metrics:
        metric_cards_row(baseline_metrics)
    else:
        st.info("Baseline metrics not available.")

    st.divider()

    # PHANTOM metrics
    st.subheader("PHANTOM (Financial Domain)")
    if phantom_df is not None and phantom_metrics:
        metric_cards_row(phantom_metrics, baseline=baseline_metrics)

        # Domain gap analysis
        st.divider()
        st.subheader("Domain Gap Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Metric comparison bar chart
            metrics_compare = {
                "HaluEval": baseline_metrics,
                "PHANTOM": phantom_metrics,
            }

            metric_keys = ["accuracy", "precision_hallucinated", "recall_hallucinated", "f1_hallucinated"]
            labels = ["Accuracy", "Precision", "Recall", "F1"]

            fig = go.Figure()
            for dataset_name, m in metrics_compare.items():
                vals = [m.get(k, m.get(k.replace("_hallucinated", ""), 0)) for k in metric_keys]
                fig.add_trace(go.Bar(name=dataset_name, x=labels, y=vals,
                                     text=[f"{v:.3f}" for v in vals], textposition="auto"))

            fig.update_layout(
                title="Performance by Domain",
                barmode="group",
                yaxis_range=[0, 1.1],
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Performance drop table
            st.markdown("### Performance Delta")
            for key, label in zip(metric_keys, labels):
                halueval_val = baseline_metrics.get(key, baseline_metrics.get(key.replace("_hallucinated", ""), 0))
                phantom_val = phantom_metrics.get(key, phantom_metrics.get(key.replace("_hallucinated", ""), 0))
                delta = phantom_val - halueval_val
                emoji = "🔴" if delta < -0.05 else ("🟡" if delta < 0 else "🟢")
                st.markdown(f"{emoji} **{label}:** {delta:+.4f} ({halueval_val:.4f} -> {phantom_val:.4f})")

        # Dataset statistics comparison
        st.divider()
        st.subheader("Dataset Statistics")

        halueval_norm = SessionState.load_predictions(PATHS["halueval_normalized"])
        phantom_norm = SessionState.load_predictions(PATHS["phantom_normalized"])

        if halueval_norm is not None and phantom_norm is not None:
            stats = []
            for name, df in [("HaluEval", halueval_norm), ("PHANTOM", phantom_norm)]:
                ctx_len = df["context"].astype(str).apply(len)
                resp_len = df["response"].astype(str).apply(len)
                stats.append({
                    "Dataset": name,
                    "Total Rows": len(df),
                    "Avg Context Length": f"{ctx_len.mean():.0f} chars",
                    "Avg Response Length": f"{resp_len.mean():.0f} chars",
                    "Label Balance": f"{(df['label'] == 1).mean():.1%} hallucinated",
                })
            st.dataframe(stats, use_container_width=True, hide_index=True)

    else:
        st.info("""
        PHANTOM results not available. To generate them, run:
        ```bash
        python src/load_phantom.py
        python src/run_phantom_eval.py
        ```
        """)

    # Key insight
    st.divider()
    st.subheader("Key Insights")
    st.markdown("""
    - **Domain shift significantly impacts performance.** Models trained on general QA often struggle with
      domain-specific terminology and reasoning patterns in financial documents.
    - **Financial text has longer contexts** with more technical vocabulary, making hallucination detection harder.
    - **Fine-tuning on domain data helps** — the PHANTOM-trained DeBERTa model shows improved performance
      on financial text compared to the HaluEval-trained variant.
    """)
