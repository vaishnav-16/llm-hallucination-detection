# CLAUDE.md — FastHalluCheck Project

## Project Overview

**FastHalluCheck: Rapid Evaluation of Pretrained Hallucination Detectors on QA Benchmarks**

This is a self-contained ML evaluation project. The goal is to evaluate pretrained hallucination detection models against benchmark QA datasets and produce a clean NeurIPS-style report with quantitative metrics and qualitative error analysis.

**Research Question:** How effective are pretrained hallucination detection models at identifying unsupported answers in question-answering settings?

## Critical Rules

1. **NEVER ask the user for input.** Make all decisions autonomously. If something is ambiguous, pick the most reasonable default and document it.
2. **If a package fails to install, find an alternative.** Do not stop. For example, if `hdm2` fails, skip it and note it in the report.
3. **If a dataset schema is unexpected, inspect it and adapt.** Print columns, print 3 sample rows, then write the normalization logic.
4. **If a model fails to load or run, catch the error, log it, and move to the next step.**
5. **Always print progress** so the user can see what's happening: "Loading dataset...", "Running inference on 300 examples...", "Computing metrics...", etc.
6. **Save all intermediate outputs** to `results/` as CSV or JSON so nothing is lost if a later step fails.
7. **GPU setup:** The machine has an **NVIDIA RTX 4070 Mobile (8GB VRAM)** and **16GB system RAM**. Always try CUDA first: `device = "cuda" if torch.cuda.is_available() else "cpu"` and print which device is being used. If CUDA fails at runtime (OOM etc.), catch the error, halve the batch size, and retry. If still failing, fall back to CPU automatically.
8. **Batch sizes:** Use `batch_size=32` as default on GPU (HHEM/DeBERTa-base is ~86M params, fits easily in 8GB VRAM). On CPU fallback, use `batch_size=8`.

## Repository Structure

```
hallucination-project/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview and how to run
├── requirements.txt             # Python dependencies
├── setup.sh                     # One-command setup script
├── run_all.sh                   # One-command full pipeline
├── src/
│   ├── load_data.py             # Download and normalize HaluEval QA
│   ├── run_hhem.py              # Run Vectara HHEM on normalized data
│   ├── evaluate_results.py      # Compute metrics from predictions
│   ├── error_analysis.py        # Analyze false positives/negatives
│   └── generate_report.py       # Auto-generate markdown report
├── data/                        # Normalized datasets (generated)
├── results/                     # Predictions, metrics, plots (generated)
├── figures/                     # Charts and visualizations (generated)
└── report/
    └── report.md                # Final NeurIPS-style report (generated)
```

## Tech Stack

- **Python 3.10+**
- `datasets` — load HaluEval from Hugging Face Hub
- `transformers` — load and run pretrained models
- `sentence-transformers` — for cross-encoder models like HHEM
- `scikit-learn` — metrics (accuracy, precision, recall, F1, confusion matrix)
- `pandas` — data manipulation
- `matplotlib` / `seaborn` — plotting
- `torch` — ML backend (CUDA preferred, CPU fallback)

## Dataset: HaluEval QA

- **Source:** `pminervini/HaluEval` on Hugging Face Hub
- **Subset:** QA
- **Expected schema:** Each example has a question, some knowledge/context, a correct answer, and a hallucinated answer.
- **Normalization strategy:** Convert into pairs of (context, response, label) where label=1 means hallucinated and label=0 means faithful. This means each raw example produces TWO evaluation rows: one with the correct answer (label=0) and one with the hallucinated answer (label=1).
- **Sample size:** Use 500 raw examples → 1000 evaluation pairs. With GPU this runs fast (~2-3 min). Scripts should accept `--num_samples` via argparse to allow adjustment.

## Model: Vectara HHEM

- **Model ID:** `vectara/hallucination_evaluation_model`
- **Type:** Cross-encoder / text-classification model
- **Input:** Pair of (premise/context, hypothesis/response)
- **Output:** Score from 0 to 1 where higher = more consistent (less hallucinated)
- **Threshold:** 0.5 — scores below 0.5 are predicted as hallucinated
- **Loading:** Use `from sentence_transformers import CrossEncoder` and load the model. If that fails, try `transformers` pipeline with `text-classification`.
- **IMPORTANT:** HHEM scores CONSISTENCY. So score > 0.5 means NOT hallucinated (label=0), score <= 0.5 means hallucinated (label=1). Make sure the label mapping is correct or all metrics will be inverted.

## Metrics to Compute

- Accuracy
- Precision (for hallucination class)
- Recall (for hallucination class)
- F1 (for hallucination class)
- Confusion matrix (as heatmap figure)
- Score distribution histogram by true label
- Latency: total time and samples/second

## Error Analysis

- Print 10 false positives (predicted hallucinated but actually faithful)
- Print 10 false negatives (predicted faithful but actually hallucinated)
- For each error, show: question, context (truncated to 200 chars), response, true label, predicted label, score
- Save error analysis to `results/error_analysis.csv`

## Report Structure (NeurIPS-style Markdown)

1. Title + Abstract
2. Introduction — why hallucination detection matters
3. Related Work — HaluEval, PHANTOM, HHEM, other detectors
4. Task Definition — grounded vs hallucinated response classification
5. Dataset — HaluEval QA description and preprocessing
6. Method — HHEM as zero-shot hallucination detector
7. Experimental Setup — hardware, sample size, threshold, metrics
8. Results — table of metrics + figures
9. Error Analysis — patterns in false positives and false negatives
10. Discussion — strengths, weaknesses, when this approach works
11. Limitations — single dataset, single model, small sample, no fine-tuning comparison
12. Conclusion
13. References

## Pipeline Execution Order

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Load and normalize dataset
python src/load_data.py

# Step 3: Run HHEM inference
python src/run_hhem.py

# Step 4: Compute metrics
python src/evaluate_results.py

# Step 5: Run error analysis
python src/error_analysis.py

# Step 6: Generate report
python src/generate_report.py
```

Each script should be runnable independently (reads from files, writes to files) and should also work as part of the sequential pipeline via `python run_all.py` (cross-platform) or `./run_all.sh` (bash/Linux/WSL).

## Environment

- **OS:** Windows 10/11
- **GPU:** NVIDIA RTX 4070 Mobile (8GB VRAM)
- **RAM:** 16GB
- **IMPORTANT:** Use `python` commands, not bash scripts, since this is Windows. Use `python setup.py` instead of `./setup.sh`. Use `python run_all.py` instead of `./run_all.sh`. If running in WSL, bash scripts work fine.

## Fallback Strategies

| Problem | Fallback |
|---------|----------|
| HaluEval fails to load from Hub | Try `datasets.load_dataset("pminervini/HaluEval", "qa")` with different config names. If all fail, create a synthetic mini-dataset of 50 examples. |
| HHEM fails to load | Try `transformers` pipeline instead of `sentence_transformers`. If that fails, use a simple embedding similarity baseline with `all-MiniLM-L6-v2`. |
| CUDA OOM or not available | Halve batch_size and retry. If still OOM, fall back to CPU with batch_size=8. |
| Any script crashes | Wrap main() in try/except, print the traceback, save partial results, exit with code 1. |

## Style Guidelines

- All Python files should have docstrings at the top explaining what they do.
- Use `argparse` for any configurable parameters (sample size, threshold, paths).
- Use `logging` module, not bare print statements.
- Use type hints.
- Use `if __name__ == "__main__":` pattern.
- Set random seed to 42 everywhere for reproducibility.

## Token Efficiency (Important for Usage Limits)

- **Model switching:** The project settings.json defaults to `sonnet` which handles 90% of this project fine. If you need deeper reasoning for debugging a tricky failure, switch with `/model opus`, then switch back with `/model sonnet` when done. You can also use `/model opusplan` which automatically uses Opus for planning and Sonnet for execution.
- Claude Code WILL automatically fall back to Sonnet if Opus quota is exhausted — this is fine for this project.
- Use `/compact` between phases if the context window is getting large.
- Write multiple scripts in a single turn when possible before running them.
- If you hit a limit mid-project, the user will resume with a new session. All scripts read from and write to files, so you can pick up from any phase by checking which output files already exist.
- Check quota status anytime with `/status`.

## Phase 2: Improvement Phase

The baseline pipeline is complete. Phase 2 improves the project with:

### Baseline Results (for reference)
- Accuracy: 73.2%, Precision: 88.2%, Recall: 53.6%, F1: 66.7%, ROC-AUC: 71.7%
- 232 false negatives, 36 false positives — HHEM is too conservative

### PHANTOM Dataset
- **Source:** `seyled/Phantom_Hallucination_Detection` on Hugging Face
- **Loading:** `load_dataset("seyled/Phantom_Hallucination_Detection", data_files="PhantomDataset/Phantom_10K_seed.csv")`
- **Alternative files:** `Phantom_10K_seed.csv`, `Phantom_def14A_seed.csv`, `Phantom_497K_seed.csv` — use seed variants (short ~500 token contexts)
- **Domain:** Financial (SEC filings) — expect worse performance than HaluEval since models weren't trained on finance
- **Schema:** Has query-answer-document triplets with correct and hallucinated answers. Inspect actual columns before coding.
- **PHANTOM paper:** Ji et al., 2025. "PHANTOM: A Benchmark for Hallucination Detection in Financial Long-Context QA." NeurIPS 2025 Datasets and Benchmarks Track.

### Improvement Strategies
1. **Threshold optimization:** Sweep 0.1–0.9, find optimal by F1 and Youden's J
2. **Fine-tune DeBERTa-v3-base:** Same backbone as HHEM, supervised on HaluEval 80/20 split
3. **Ensemble:** Combine HHEM + fine-tuned DeBERTa predictions
4. **Cross-domain evaluation:** Test all models on PHANTOM (financial domain)

### Additional Output Files (Phase 2)
```
models/deberta-hallucination-detector/    # Fine-tuned model
results/optimal_threshold_metrics.json
results/hhem_predictions_optimized.csv
results/finetuned_predictions.csv
results/finetuned_metrics.json
results/phantom_hhem_predictions.csv
results/phantom_finetuned_predictions.csv
results/phantom_metrics.json
results/ensemble_metrics.json
results/comprehensive_error_analysis.csv
figures/threshold_optimization.png
figures/model_comparison_bar.png
figures/roc_curves.png
figures/error_overlap.png
figures/context_length_analysis.png
figures/cross_domain_comparison.png
figures/phantom_vs_halueval.png
report/final_report.md
```

### Fine-tuning Fallback Strategy
| Problem | Fallback |
|---------|----------|
| DeBERTa-v3-base OOM during training | Use gradient_accumulation_steps=4, batch_size=2. If still OOM, use deberta-v3-small. If still OOM, use distilbert-base-uncased. |
| PHANTOM dataset fails to load | Try different file patterns in PhantomDataset/ folder. If all fail, skip PHANTOM evaluation and note it in the report. |
| Fine-tuning takes too long | Reduce to 2 epochs. Use fp16=True. Reduce max_length to 256 tokens. |
| Ensemble gives worse results than best single model | That's a valid finding — report it honestly. |
