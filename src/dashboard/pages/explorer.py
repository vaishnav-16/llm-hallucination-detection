"""Data Explorer page — browse predictions, filter, and search examples."""

import streamlit as st
import pandas as pd

from src.dashboard.state import SessionState
from src.dashboard.components import data_table, no_data_message
from src.dashboard.config import PATHS


def render():
    st.header("Data Explorer")
    st.markdown("Browse predictions and raw data interactively.")

    # Dataset selector
    dataset = st.selectbox("Dataset", ["HaluEval (HHEM)", "HaluEval (Fine-tuned)", "PHANTOM", "Raw HaluEval"])

    path_map = {
        "HaluEval (HHEM)": PATHS["hhem_predictions"],
        "HaluEval (Fine-tuned)": PATHS["finetuned_predictions"],
        "PHANTOM": PATHS["phantom_hhem_predictions"],
        "Raw HaluEval": PATHS["halueval_normalized"],
    }

    df = SessionState.load_predictions(path_map[dataset])
    if df is None:
        no_data_message(f"Data for '{dataset}' not found.")
        return

    st.caption(f"{len(df)} rows loaded")

    # Sidebar filters
    with st.sidebar:
        st.subheader("Filters")

        if "label" in df.columns:
            label_filter = st.multiselect("True Label", [0, 1], default=[0, 1],
                                           format_func=lambda x: "Faithful" if x == 0 else "Hallucinated")
            df = df[df["label"].isin(label_filter)]

        if "predicted_label" in df.columns:
            pred_filter = st.multiselect("Predicted Label", [0, 1], default=[0, 1],
                                          format_func=lambda x: "Faithful" if x == 0 else "Hallucinated")
            df = df[df["predicted_label"].isin(pred_filter)]

        # Error filter
        if "label" in df.columns and "predicted_label" in df.columns:
            error_only = st.checkbox("Show errors only")
            if error_only:
                df = df[df["label"] != df["predicted_label"]]

        # Score filter
        score_col = None
        for col in ["hhem_score", "predicted_prob"]:
            if col in df.columns:
                score_col = col
                break
        if score_col:
            score_range = st.slider("Score Range", 0.0, 1.0, (0.0, 1.0), 0.01)
            df = df[(df[score_col] >= score_range[0]) & (df[score_col] <= score_range[1])]

    # Search
    search = st.text_input("Search in questions/responses", "")
    if search:
        mask = pd.Series(False, index=df.index)
        for col in ["question", "response", "context"]:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]

    st.caption(f"Showing {len(df)} rows after filters")

    # Column selector
    all_cols = list(df.columns)
    default_cols = [c for c in ["id", "question", "response", "label", "predicted_label",
                                 "hhem_score", "predicted_prob"] if c in all_cols]
    selected_cols = st.multiselect("Columns to display", all_cols, default=default_cols)

    if selected_cols:
        data_table(df[selected_cols])
    else:
        data_table(df)

    # Example detail expander
    if len(df) > 0:
        st.divider()
        st.subheader("Example Detail")
        idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0)
        row = df.iloc[idx]

        col1, col2 = st.columns(2)
        with col1:
            if "question" in row.index:
                st.markdown(f"**Question:** {row['question']}")
            if "label" in row.index:
                label_text = "Faithful" if row["label"] == 0 else "Hallucinated"
                st.markdown(f"**True Label:** {label_text}")
            if "predicted_label" in row.index:
                pred_text = "Faithful" if row["predicted_label"] == 0 else "Hallucinated"
                correct = row.get("label") == row.get("predicted_label")
                st.markdown(f"**Predicted:** {pred_text} {'✓' if correct else '✗'}")
        with col2:
            if score_col and score_col in row.index:
                st.markdown(f"**Score:** {row[score_col]:.4f}")
            if "response" in row.index:
                st.markdown(f"**Response:** {row['response'][:500]}")

        if "context" in row.index:
            with st.expander("Full Context"):
                st.text(str(row["context"]))
