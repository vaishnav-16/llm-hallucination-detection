"""Cross-platform pipeline runner for FastHalluCheck project."""
import subprocess
import sys
import os

STEPS = [
    ("1/5", "Loading and normalizing dataset", "src/load_data.py"),
    ("2/5", "Running HHEM inference", "src/run_hhem.py"),
    ("3/5", "Computing evaluation metrics", "src/evaluate_results.py"),
    ("4/5", "Running error analysis", "src/error_analysis.py"),
    ("5/5", "Generating report", "src/generate_report.py"),
]


def main():
    print("=" * 50)
    print("  FastHalluCheck — Full Pipeline")
    print("=" * 50)

    for step_num, desc, script in STEPS:
        print(f"\n[{step_num}] {desc}...")
        if not os.path.exists(script):
            print(f"  ERROR: {script} not found. Skipping.")
            continue
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"  WARNING: {script} exited with code {result.returncode}")

    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    print("  Results: results/")
    print("  Figures: figures/")
    print("  Report:  report/report.md")
    print("=" * 50)


if __name__ == "__main__":
    main()
