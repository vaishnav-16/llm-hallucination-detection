#!/bin/bash
set -e
echo "============================================"
echo "  FastHalluCheck — Full Pipeline"
echo "============================================"
echo ""

echo "[1/5] Loading and normalizing dataset..."
python src/load_data.py
echo ""

echo "[2/5] Running HHEM inference..."
python src/run_hhem.py
echo ""

echo "[3/5] Computing evaluation metrics..."
python src/evaluate_results.py
echo ""

echo "[4/5] Running error analysis..."
python src/error_analysis.py
echo ""

echo "[5/5] Generating report..."
python src/generate_report.py
echo ""

echo "============================================"
echo "  Pipeline complete!"
echo "  Results: results/"
echo "  Figures: figures/"
echo "  Report:  report/report.md"
echo "============================================"
