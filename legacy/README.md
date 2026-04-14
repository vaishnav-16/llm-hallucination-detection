# Legacy Scripts

This folder contains the original flat-script pipeline from Phases 1 and 2 of the FastHalluCheck project. These scripts were the initial implementation before the layered architecture refactoring.

## Scripts

| Script | Purpose |
|--------|---------|
| `src/load_data.py` | Download and normalize HaluEval QA dataset |
| `src/run_hhem.py` | Run Vectara HHEM inference |
| `src/evaluate_results.py` | Compute metrics and generate figures |
| `src/error_analysis.py` | Analyze false positives/negatives |
| `src/generate_report.py` | Generate initial markdown report |
| `src/data_analysis.py` | Feature engineering and data exploration |
| `src/finetune_classifier.py` | Fine-tune DeBERTa on HaluEval |
| `src/finetune_phantom.py` | Fine-tune DeBERTa on PHANTOM |
| `src/load_phantom.py` | Load PHANTOM financial dataset |
| `src/load_phantom_2k.py` | Load PHANTOM 2K-token variant |
| `src/optimize_threshold.py` | Cross-validated threshold optimization |
| `src/hallucination_type_analysis.py` | Categorize errors by hallucination type |
| `src/ensemble_and_analysis.py` | Ensemble models + comprehensive analysis |
| `src/decision_policy.py` | 3-tier ANSWER/CAVEAT/ABSTAIN policy |
| `src/run_phantom_eval.py` | Evaluate models on PHANTOM domain |
| `src/eval_phantom_2k.py` | Evaluate all models on PHANTOM 2K variant |
| `src/generate_final_report.py` | Generate comprehensive final report |

## Root-level scripts

| Script | Purpose |
|--------|---------|
| `run_all.py` | Sequential pipeline runner (5 steps) |
| `run_all.sh` | Bash wrapper for pipeline |
| `setup.py` | Cross-platform environment setup |
| `setup.sh` | Bash setup script |

## Running Legacy Scripts

These scripts can still be run directly from the project root:

```bash
python legacy/src/load_data.py
python legacy/src/run_hhem.py
python legacy/src/evaluate_results.py
```

They read from and write to the same `data/`, `results/`, `figures/`, and `report/` directories as the new architecture.

## Why These Were Replaced

1. **Code duplication** — `compute_metrics()` was defined in 3+ files
2. **No separation of concerns** — each script mixed data I/O, business logic, and visualization
3. **No interactive exploration** — static figures only, no dashboard
4. **No unified CLI** — each script had its own argparse entry point
