"""Error Browser page — explore false positives and false negatives."""

import streamlit as st
import pandas as pd
import numpy as np

from src.dashboard.state import SessionState
from src.dashboard.components import no_data_message, data_table


def render():
    st.header("Error Browser")
    st.markdown("Explore misclassified examples to understand model failure patterns.")

    hhem_df = SessionState.get_hhem_predictions()
    if hhem_df is None:
        no_data_message("HHEM predictions not found.")
        return

    y_true = hhem_df["label"].values
    y_pred = hhem_df["predicted_label"].values

    # Error stats
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("False Positives", int(fp_mask.sum()),
                   help="Predicted hallucinated but actually faithful")
    with col2:
        st.metric("False Negatives", int(fn_mask.sum()),
                   help="Predicted faithful but actually hallucinated")
    with col3:
        st.metric("Total Errors", int(fp_mask.sum() + fn_mask.sum()))

    st.divider()

    # Error type filter
    error_type = st.radio("Error Type", ["All Errors", "False Positives", "False Negatives"],
                           horizontal=True)

    if error_type == "False Positives":
        errors_df = hhem_df[fp_mask].copy()
        errors_df["error_type"] = "false_positive"
    elif error_type == "False Negatives":
        errors_df = hhem_df[fn_mask].copy()
        errors_df["error_type"] = "false_negative"
    else:
        fp_df = hhem_df[fp_mask].copy()
        fp_df["error_type"] = "false_positive"
        fn_df = hhem_df[fn_mask].copy()
        fn_df["error_type"] = "false_negative"
        errors_df = pd.concat([fp_df, fn_df])

    # Search within errors
    search = st.text_input("Search in errors", "",
                            placeholder="Type to filter by question, response, or context...")
    if search:
        mask = pd.Series(False, index=errors_df.index)
        for col in ["question", "response", "context"]:
            if col in errors_df.columns:
                mask = mask | errors_df[col].astype(str).str.contains(search, case=False, na=False)
        errors_df = errors_df[mask]

    # Sort options
    sort_col = "hhem_score" if "hhem_score" in errors_df.columns else None
    if sort_col:
        sort_order = st.radio("Sort by score", ["Most confident errors first", "Least confident first"],
                               horizontal=True)
        ascending = sort_order == "Least confident first"
        errors_df = errors_df.sort_values(sort_col, ascending=ascending)

    st.caption(f"Showing {len(errors_df)} errors")

    # Display columns
    display_cols = [c for c in ["id", "error_type", "question", "response", "hhem_score",
                                 "label", "predicted_label"] if c in errors_df.columns]
    data_table(errors_df[display_cols])

    # Detail view
    if len(errors_df) > 0:
        st.divider()
        st.subheader("Error Detail")
        idx = st.selectbox("Select error", range(min(len(errors_df), 50)),
                            format_func=lambda i: f"#{i+1}: {errors_df.iloc[i].get('id', i)} "
                                                    f"({errors_df.iloc[i].get('error_type', '')})")
        row = errors_df.iloc[idx]

        error_color = "🔴" if row.get("error_type") == "false_negative" else "🟡"
        st.markdown(f"### {error_color} {row.get('error_type', '').replace('_', ' ').title()}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Question:** {row.get('question', 'N/A')}")
            st.markdown(f"**True Label:** {'Faithful' if row.get('label') == 0 else 'Hallucinated'}")
            st.markdown(f"**Predicted:** {'Faithful' if row.get('predicted_label') == 0 else 'Hallucinated'}")
            if "hhem_score" in row.index:
                st.markdown(f"**HHEM Score:** {row['hhem_score']:.4f}")

        with col2:
            st.markdown(f"**Response:**")
            st.text(str(row.get("response", ""))[:500])

        if "context" in row.index:
            with st.expander("Full Context"):
                st.text(str(row["context"]))

    # Hallucination type analysis
    hal_types = SessionState.get_hallucination_types()
    if hal_types is not None and len(hal_types) > 0:
        st.divider()
        st.subheader("Error Categorization")
        type_counts = hal_types["hallucination_type"].value_counts()
        st.bar_chart(type_counts)
