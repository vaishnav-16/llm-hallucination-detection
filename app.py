"""FastHalluCheck dashboard: browse the v2 evaluation artifacts (metrics, comparisons, predictions).

Launch with ``streamlit run app.py``. Everything shown is read from ``artifacts/`` (committed), so the
dashboard works on a fresh clone; the optional text join on the Predictions page uses the git-ignored
``data/`` and degrades to metadata-only when it is absent.
"""

from __future__ import annotations

import importlib
import logging
import traceback

import streamlit as st

from src.config import setup_logging
from src.dashboard import data
from src.dashboard.config import LAYOUT, PAGE_ICON, PAGE_TITLE, PAGES

logger = logging.getLogger(__name__)

PAGE_MODULES = {
    "Overview": "src.dashboard.pages.overview",
    "Results": "src.dashboard.pages.results",
    "Length & position (PHANTOM)": "src.dashboard.pages.length_position",
    "Comparisons & seeds": "src.dashboard.pages.comparisons",
    "Predictions": "src.dashboard.pages.predictions",
}


def main() -> None:
    setup_logging()
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT, initial_sidebar_state="expanded")

    with st.sidebar:
        st.title(f"{PAGE_ICON} {PAGE_TITLE}")
        st.caption("Hallucination-detector evaluation (v2 artifacts)")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        with st.expander("Data status"):
            st.caption(f"Reading from `{data.artifacts_root()}`")
            for name, status in data.artifact_status().items():
                st.text(f"{name}: {status}")

    try:
        importlib.import_module(PAGE_MODULES[page]).render()
    except Exception as e:  # show the error in the page instead of a blank screen
        logger.exception("page %s failed", page)
        st.error(f"Error rendering page '{page}': {e}")
        st.markdown("Check that `artifacts/metrics/metrics_long.csv` and `artifacts/predictions/` exist "
                    "(they are produced by the `evaluate` and `verify` stages).")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
