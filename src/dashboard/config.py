"""Dashboard presentation settings: page config, colours and display names.

Kept separate from the data layer so the look of the dashboard can change without touching
how artifacts are read, and so the data helpers stay importable without Streamlit.
"""

from __future__ import annotations

PAGE_TITLE = "FastHalluCheck"
PAGE_ICON = "🔍"
LAYOUT = "wide"

PAGES = ["Overview", "Results", "Length & position (PHANTOM)", "Comparisons & seeds", "Predictions"]

RESEARCH_QUESTION = (
    "How effective are pretrained hallucination detectors at identifying unsupported answers in "
    "question-answering settings? Each detector scores (context, answer) pairs from HaluEval, RAGTruth "
    "and the long-context financial PHANTOM benchmark; every number is measured on held-out test slices, "
    "with decision thresholds chosen on the matching validation slice and 95% pair-clustered bootstrap CIs. "
    "Answer-only baselines (which never see the context) show how much of a benchmark can be solved by "
    "answer-style shortcuts alone."
)

SLIDER_NOTE = "exploration only; reported numbers use validation thresholds"

COLORS = {
    "faithful": "#4A90D9",
    "hallucinated": "#E74C3C",
    "baseline": "#7F8C8D",
    "val_threshold": "#2C3E50",
    "explore_threshold": "#F39C12",
}
BASELINE_DASH = {"answer_tfidf": "dash", "answer_length": "dot"}

METRIC_NAMES = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "ece": "ECE (lower is better)",
}


def display_name(detector: str, strategy: str | None = None) -> str:
    """Short human label for a detector (and optionally its context strategy)."""
    if strategy is None or strategy in ("", "none"):
        return detector
    return f"{detector} · {strategy}"


def metric_name(metric: str) -> str:
    return METRIC_NAMES.get(metric, metric)
