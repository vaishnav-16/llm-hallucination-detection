"""Markdown table formatters for metrics display."""

from typing import Dict, List


def metrics_to_markdown_table(metrics: dict, title: str = "Metrics") -> str:
    """Format a single metrics dictionary as a markdown table."""
    rows = []
    rows.append(f"### {title}")
    rows.append("")
    rows.append("| Metric | Value |")
    rows.append("|--------|-------|")

    display_keys = [
        ("accuracy", "Accuracy"),
        ("precision_hallucinated", "Precision (Hallucinated)"),
        ("recall_hallucinated", "Recall (Hallucinated)"),
        ("f1_hallucinated", "F1 (Hallucinated)"),
        ("roc_auc", "ROC AUC"),
        ("total_samples", "Total Samples"),
    ]

    for key, label in display_keys:
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            if isinstance(val, float):
                rows.append(f"| {label} | {val:.4f} |")
            else:
                rows.append(f"| {label} | {val} |")

    return "\n".join(rows)


def model_comparison_table(all_metrics: Dict[str, dict]) -> str:
    """Format multiple model metrics as a comparison table."""
    rows = []
    rows.append("| Model | Accuracy | Precision | Recall | F1 | ROC AUC |")
    rows.append("|-------|----------|-----------|--------|-----|---------|")

    for name, m in all_metrics.items():
        acc = m.get("accuracy", 0)
        prec = m.get("precision_hallucinated", m.get("precision", 0))
        rec = m.get("recall_hallucinated", m.get("recall", 0))
        f1 = m.get("f1_hallucinated", m.get("f1", 0))
        auc = m.get("roc_auc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        rows.append(f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc_str} |")

    return "\n".join(rows)


def policy_table(results: dict) -> str:
    """Format decision policy results as a table."""
    rows = []
    rows.append("| Tolerance | Low Thresh | High Thresh | Coverage | Answer% | Caveat% | Abstain% | Halluc Rate |")
    rows.append("|-----------|-----------|-------------|----------|---------|---------|----------|-------------|")

    for name, m in results.items():
        rows.append(
            f"| {name} | {m['low_threshold']:.2f} | {m['high_threshold']:.2f} | "
            f"{m['coverage']:.1%} | {m['pct_answer']:.1%} | {m['pct_caveat']:.1%} | "
            f"{m['pct_abstain']:.1%} | {m['halluc_rate_in_answers']:.3%} |"
        )

    return "\n".join(rows)
