"""Session state management — centralized state for the Streamlit dashboard."""

import os
from typing import Optional

import pandas as pd
import streamlit as st

from src.data.store import read_predictions, read_metrics
from src.dashboard.config import PATHS


class SessionState:
    """Centralized session state manager.

    Handles lazy loading, caching, and derived state invalidation.
    """

    @staticmethod
    def _ensure_key(key: str, default=None):
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Threshold ---

    @staticmethod
    def get_threshold() -> float:
        SessionState._ensure_key("threshold", 0.5)
        return st.session_state["threshold"]

    @staticmethod
    def set_threshold(value: float):
        if st.session_state.get("threshold") != value:
            st.session_state["threshold"] = value
            SessionState.invalidate_metrics()

    @staticmethod
    def invalidate_metrics():
        """Clear cached derived metrics when inputs change."""
        for key in list(st.session_state.keys()):
            if key.startswith("_cached_metrics_"):
                del st.session_state[key]

    # --- Data Loading (cached) ---

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_predictions(path: str) -> Optional[pd.DataFrame]:
        """Load predictions from CSV with caching."""
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_metrics(path: str) -> dict:
        """Load metrics from JSON with caching."""
        return read_metrics(path)

    # --- Convenience accessors ---

    @staticmethod
    def get_hhem_predictions() -> Optional[pd.DataFrame]:
        return SessionState.load_predictions(PATHS["hhem_predictions"])

    @staticmethod
    def get_finetuned_predictions() -> Optional[pd.DataFrame]:
        return SessionState.load_predictions(PATHS["finetuned_predictions"])

    @staticmethod
    def get_phantom_predictions() -> Optional[pd.DataFrame]:
        return SessionState.load_predictions(PATHS["phantom_hhem_predictions"])

    @staticmethod
    def get_baseline_metrics() -> dict:
        return SessionState.load_metrics(PATHS["metrics"])

    @staticmethod
    def get_finetuned_metrics() -> dict:
        return SessionState.load_metrics(PATHS["finetuned_metrics"])

    @staticmethod
    def get_ensemble_metrics() -> dict:
        return SessionState.load_metrics(PATHS["ensemble_metrics"])

    @staticmethod
    def get_optimal_threshold_metrics() -> dict:
        return SessionState.load_metrics(PATHS["optimal_threshold_metrics"])

    @staticmethod
    def get_policy_metrics() -> dict:
        return SessionState.load_metrics(PATHS["decision_policy_metrics"])

    @staticmethod
    def get_error_analysis() -> Optional[pd.DataFrame]:
        path = PATHS["comprehensive_error_analysis"]
        if not os.path.exists(path):
            path = PATHS["error_analysis"]
        return SessionState.load_predictions(path)

    @staticmethod
    def get_hallucination_types() -> Optional[pd.DataFrame]:
        return SessionState.load_predictions(PATHS["hallucination_type_analysis"])

    @staticmethod
    def get_available_data() -> dict:
        """Check which data files are available."""
        return {name: os.path.exists(path) for name, path in PATHS.items()}
