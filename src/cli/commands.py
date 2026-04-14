"""Unified CLI for FastHalluCheck pipeline.

Usage:
    python -m src.cli.commands run          # Full pipeline
    python -m src.cli.commands evaluate     # Compute metrics
    python -m src.cli.commands analyze      # Error analysis
    python -m src.cli.commands report       # Generate report
    python -m src.cli.commands compare      # Model comparison
    python -m src.cli.commands dashboard    # Launch Streamlit dashboard
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_run(args):
    """Run the full pipeline using legacy scripts."""
    steps = [
        ("1/5", "Loading dataset", "legacy/src/load_data.py"),
        ("2/5", "Running HHEM inference", "legacy/src/run_hhem.py"),
        ("3/5", "Computing metrics", "legacy/src/evaluate_results.py"),
        ("4/5", "Error analysis", "legacy/src/error_analysis.py"),
        ("5/5", "Generating report", "legacy/src/generate_report.py"),
    ]

    print("=" * 50)
    print("  FastHalluCheck — Full Pipeline")
    print("=" * 50)

    for step_num, desc, script in steps:
        print(f"\n[{step_num}] {desc}...")
        if not os.path.exists(script):
            print(f"  ERROR: {script} not found. Skipping.")
            continue
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"  WARNING: {script} exited with code {result.returncode}")

    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    print("=" * 50)


def cmd_evaluate(args):
    """Compute metrics from predictions."""
    from src.services.metrics import compute_metrics, compute_metrics_at_threshold, get_classification_report
    from src.data.store import read_predictions, save_metrics

    input_path = args.input or "results/hhem_predictions.csv"
    df = read_predictions(input_path)
    if df is None:
        logger.error(f"No predictions found at {input_path}. Run the pipeline first.")
        return

    y_true = df["label"].values
    score_col = "hhem_score" if "hhem_score" in df.columns else "predicted_prob"
    scores = df[score_col].values

    threshold = args.threshold or 0.5
    metrics = compute_metrics_at_threshold(y_true, scores, threshold)
    report = get_classification_report(y_true, np.where(scores > threshold, 0, 1))

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        if k != "label":
            print(f"  {k}: {v}")
    print(f"\n{report}")

    output = args.output or "results/metrics.json"
    save_metrics(metrics, output)


def cmd_analyze(args):
    """Run error analysis."""
    from src.services.analysis import classify_errors
    from src.data.store import read_predictions, save_errors

    input_path = args.input or "results/hhem_predictions.csv"
    df = read_predictions(input_path)
    if df is None:
        logger.error(f"No predictions found at {input_path}.")
        return

    errors = classify_errors(df, n_show=args.n_show or 10)
    output = args.output or "results/error_analysis.csv"
    save_errors(errors, output)


def cmd_report(args):
    """Generate the final report."""
    from src.display.report_generator import generate_report
    output = args.output or "report/final_report.md"
    generate_report(output)
    print(f"Report generated at {output}")


def cmd_compare(args):
    """Compare all available models."""
    from src.data.store import read_predictions, read_metrics, list_available_results

    available = list_available_results()
    print("\n=== Available Results ===")
    for name, path in sorted(available.items()):
        print(f"  {name}: {path}")

    # Simple metrics files (single model per file)
    simple_files = [
        ("HHEM Baseline", "results/metrics.json"),
        ("Fine-tuned DeBERTa", "results/finetuned_metrics.json"),
    ]
    # Nested metrics files (multiple models per file)
    nested_files = [
        ("results/ensemble_metrics.json", "Ensemble"),
        ("results/phantom_metrics.json", "PHANTOM"),
    ]

    print("\n=== Model Comparison ===")
    print(f"{'Model':<30} {'Accuracy':>8} {'F1':>8} {'Precision':>9} {'Recall':>8}")
    print("-" * 65)

    def _print_model(name, m):
        # Sanitize name for Windows console encoding
        safe_name = name.encode("ascii", errors="replace").decode("ascii")
        acc = m.get("accuracy", 0)
        f1 = m.get("f1_hallucinated", m.get("f1", 0))
        prec = m.get("precision_hallucinated", m.get("precision", 0))
        rec = m.get("recall_hallucinated", m.get("recall", 0))
        print(f"{safe_name:<30} {acc:>8.4f} {f1:>8.4f} {prec:>9.4f} {rec:>8.4f}")

    for name, path in simple_files:
        m = read_metrics(path)
        if m:
            _print_model(name, m)

    for path, prefix in nested_files:
        data = read_metrics(path)
        if data:
            for sub_name, m in data.items():
                if isinstance(m, dict) and "accuracy" in m:
                    _print_model(f"{prefix}: {sub_name}", m)


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    app_path = os.path.join(os.path.dirname(__file__), "..", "..", "app.py")
    app_path = os.path.abspath(app_path)
    if not os.path.exists(app_path):
        logger.error(f"Dashboard app not found at {app_path}")
        return
    print(f"Launching dashboard: streamlit run {app_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def main():
    parser = argparse.ArgumentParser(
        prog="fasthallucheck",
        description="FastHalluCheck — Hallucination Detection Evaluation Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    sub = subparsers.add_parser("run", help="Run full pipeline")
    sub.set_defaults(func=cmd_run)

    # evaluate
    sub = subparsers.add_parser("evaluate", help="Compute metrics from predictions")
    sub.add_argument("--input", help="Predictions CSV path")
    sub.add_argument("--output", help="Metrics JSON output path")
    sub.add_argument("--threshold", type=float, help="Classification threshold")
    sub.set_defaults(func=cmd_evaluate)

    # analyze
    sub = subparsers.add_parser("analyze", help="Run error analysis")
    sub.add_argument("--input", help="Predictions CSV path")
    sub.add_argument("--output", help="Error analysis CSV output path")
    sub.add_argument("--n_show", type=int, help="Number of examples to show")
    sub.set_defaults(func=cmd_analyze)

    # report
    sub = subparsers.add_parser("report", help="Generate final report")
    sub.add_argument("--output", help="Report output path")
    sub.set_defaults(func=cmd_report)

    # compare
    sub = subparsers.add_parser("compare", help="Compare all models")
    sub.set_defaults(func=cmd_compare)

    # dashboard
    sub = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    sub.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
