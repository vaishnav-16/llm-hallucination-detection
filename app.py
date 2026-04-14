"""FastHalluCheck — Interactive Streamlit Dashboard.

Launch: streamlit run app.py
"""

import streamlit as st

from src.dashboard.config import PAGE_TITLE, PAGE_ICON, LAYOUT

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


def main():
    # Sidebar navigation
    with st.sidebar:
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        st.markdown("Hallucination Detection Evaluation")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Data Explorer",
                "Threshold Tuner",
                "Error Browser",
                "Model Comparison",
                "Cross-Domain Analysis",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # Data availability indicator
        from src.dashboard.state import SessionState
        available = SessionState.get_available_data()
        n_available = sum(available.values())
        n_total = len(available)
        st.caption(f"Data: {n_available}/{n_total} files available")

        with st.expander("Data Status"):
            for name, exists in sorted(available.items()):
                icon = "✓" if exists else "✗"
                st.text(f"{icon} {name}")

    # Route to selected page
    try:
        if page == "Overview":
            from src.dashboard.pages.overview import render
        elif page == "Data Explorer":
            from src.dashboard.pages.explorer import render
        elif page == "Threshold Tuner":
            from src.dashboard.pages.threshold_tuner import render
        elif page == "Error Browser":
            from src.dashboard.pages.error_browser import render
        elif page == "Model Comparison":
            from src.dashboard.pages.model_comparison import render
        elif page == "Cross-Domain Analysis":
            from src.dashboard.pages.cross_domain import render
        else:
            from src.dashboard.pages.overview import render

        render()

    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.markdown("""
        **Troubleshooting:**
        1. Make sure you've run the pipeline: `python run_all.py`
        2. Check that result files exist in `results/`
        3. Install dependencies: `pip install streamlit plotly`
        """)
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
