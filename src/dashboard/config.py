"""Dashboard configuration — colors, labels, page settings."""

# Page config
PAGE_TITLE = "FastHalluCheck"
PAGE_ICON = "🔍"
LAYOUT = "wide"

# Color palette
COLORS = {
    "faithful": "#4A90D9",
    "hallucinated": "#E74C3C",
    "model_hhem": "#4A90D9",
    "model_deberta": "#E67E22",
    "model_ensemble": "#27AE60",
    "model_phantom": "#8E44AD",
    "answer": "#27AE60",
    "caveat": "#F1C40F",
    "abstain": "#E74C3C",
    "background": "#0E1117",
    "card_bg": "#1E2130",
}

# Model display names
MODEL_NAMES = {
    "hhem": "HHEM (Vectara)",
    "finetuned": "Fine-tuned DeBERTa",
    "ensemble_avg": "Ensemble (Average)",
    "ensemble_weighted": "Ensemble (Weighted)",
}

# File paths
PATHS = {
    "hhem_predictions": "results/hhem_predictions.csv",
    "hhem_predictions_optimized": "results/hhem_predictions_optimized.csv",
    "finetuned_predictions": "results/finetuned_predictions.csv",
    "phantom_hhem_predictions": "results/phantom_hhem_predictions.csv",
    "phantom_finetuned_predictions": "results/phantom_finetuned_predictions.csv",
    "phantom_trained_predictions": "results/phantom_trained_predictions.csv",
    "metrics": "results/metrics.json",
    "finetuned_metrics": "results/finetuned_metrics.json",
    "phantom_metrics": "results/phantom_metrics.json",
    "optimal_threshold_metrics": "results/optimal_threshold_metrics.json",
    "ensemble_metrics": "results/ensemble_metrics.json",
    "decision_policy_metrics": "results/decision_policy_metrics.json",
    "error_analysis": "results/error_analysis.csv",
    "comprehensive_error_analysis": "results/comprehensive_error_analysis.csv",
    "hallucination_type_analysis": "results/hallucination_type_analysis.csv",
    "halueval_normalized": "data/halueval_qa_normalized.csv",
    "halueval_enriched": "data/halueval_enriched.csv",
    "phantom_normalized": "data/phantom_normalized.csv",
}
