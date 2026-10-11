# FastHalluCheck v2: length-robust hallucination detection

**Research question.** How do hallucination detectors degrade as the context grows from a ~500-token passage to
30K nominal tokens and as the evidence moves to the beginning, middle or end of the document, and does chunked or
retrieval-based verification recover the loss?

PHANTOM (Ji et al., NeurIPS 2025 D&B) makes this a controlled experiment. Its 10-K and DEF 14A filings come in
seed, 2K, 5K, 10K, 20K and 30K-token versions, each with the evidence passage at the beginning, middle or end,
and all versions reuse the seed's questions and answers. The paper evaluated generative LLMs only; encoder
detectors on this grid have not been published.

Every detector is reported next to **answer-only shortcut baselines**, because all three benchmarks can be partly
solved without reading the context:

| Test set | Answer-only length model (AUROC) | Answer-only TF-IDF (AUROC) |
|---|---|---|
| HaluEval QA | 0.982 | 0.968 |
| RAGTruth | 0.685 | 0.808 |
| PHANTOM 10-K + DEF 14A, all lengths (trained on 8-K/497K) | 0.835 | 0.957 |
| PHANTOM hard subset (faithful answer >= 224 chars) | 0.732 | 0.888 |

(`artifacts/metrics/shortcut_baselines.csv`, `metrics_long.csv`; reproduce with `python -m src.cli baselines && python -m src.cli evaluate`.)

## Team

Vaishnav Ajai, Kritesh Singh and Prajwal Prasad, Dartmouth College, Thayer School of Engineering.

## Why v2

The v1 pipeline (kept in `legacy/`) reported F1 0.964 for a fine-tuned DeBERTa and 0.982 for an ensemble on
HaluEval, and a cross-domain drop on PHANTOM. An audit found that these numbers don't hold:

- **HaluEval shortcut.** A 3-feature logistic regression on the answer alone (length, word count, trailing
  period) reaches 0.97 accuracy, so the fine-tuned score is indistinguishable from a length rule.
- **Contaminated ensemble.** The ensemble was scored on all 1,000 pairs, 800 of which were DeBERTa's training rows.
- **Checkpoint chosen on the test set.**
- **Label peeking.** Score polarity was set from the gold labels (`validate_scores`).
- **Unknown PHANTOM data.** The loader requested file names with the wrong case and mapped every PHANTOM label to 0,
  so the data behind the PHANTOM numbers is unknown.
- **Truncated contexts.** The PHANTOM contexts were cut to 512 tokens, so the "cross-domain" drop is confounded
  with never seeing the evidence.
- **Not runnable from a clone.** `src/data/` was swallowed by an unanchored `data/` rule in `.gitignore`.

## Design

- **One score convention.** Every detector outputs `p_halluc` (higher = more likely hallucinated), converted once
  inside its adapter.
- **Same verification wrapper at every length.** `truncate` (first window, the v1 behaviour), `chunk` (overlapping
  windows; the answer is faithful if any window supports it), `retrieve` (BM25 top-k windows). Window overlaps are
  set from measured passage lengths so that the evidence passage fits whole in some window.
- **Leak-free splits grouped by pair**, committed in `artifacts/splits.json`. PHANTOM 10-K/DEF 14A are test-only at
  every length. Thresholds and model selection use validation slices only.
- **Statistics.** Pair-clustered bootstrap CIs, paired bootstrap differences, McNemar tests and ECE.
- **Reproducibility.** Pinned dataset and model revisions (`src/config.py`). Predictions store ids and scores only, so
  they are committed and can be evaluated on any machine.

## Detectors

| Role | Model |
|---|---|
| Shortcut controls | answer-length LR, answer TF-IDF LR, lexical support; context ablation of each model |
| Zero-shot encoders | Vectara HHEM-2.1-open; `tasksource/ModernBERT-base-nli`; `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; LettuceDetect (ModernBERT) |
| Fine-tuned (main) | ModernBERT-base from the NLI checkpoint, positions extended to 8K, distractor augmentation, 3 seeds; DeBERTa-v3-base NLI (512-token comparison) |
| Decision model (API) | **TypeSafe Jev** (`jev-1.13.0`): a System One model that returns calibrated probabilities instead of text. Each claim is asked as a yes/no *noul* ("is every detail supported by the document?") and as a 3-way *choice* (supported / contradicted / unsupported) |
| LLM judges | Qwen3-4B-Instruct-2507 (4-bit, laptop), Qwen3-8B (Kaggle); API: Dartmouth Chat, Gemini (secondary) |

## Quick start

```bash
pip install -r requirements.txt     # CPU; for the RTX 4070 see HANDOFF.md
python -m src.cli prepare           # ~5 min: datasets at pinned revisions, splits check, data card
python -m pytest
python -m src.cli status            # what is done, what is left, and where each stage must run
python -m src.cli baselines         # shortcut audit (CPU, ~2 min)
python -m src.cli smoke             # every stage on a few rows (CPU, ~15 min)
```

Jev needs no GPU and runs anywhere with `TYPESAFE_API_KEY` set:

```bash
python -m src.cli probe-jev                                  # < $0.01 live check
python -m src.cli score --config configs/jev.yaml --dry-run  # cost estimate (~$2.60), nothing sent
python -m src.cli score --config configs/jev.yaml            # budget guard stops at $4.50
```

GPU stages (fine-tuning, the zero-shot and fine-tuned grids, the Qwen judge) run on the laptop. **See
[HANDOFF.md](HANDOFF.md)** for the Windows runbook, priorities and time estimates. Afterwards:

```bash
python -m src.cli evaluate && python -m src.cli report    # -> artifacts/metrics/, report/final_report.md
```

## Repository layout

```
configs/      experiment configs (YAML)          artifacts/   committed results: splits, data card, predictions, metrics, figures
src/data/     loaders, schema, splits            report/      generated report
src/detectors detector adapters + baselines      tests/       offline pytest suite
src/verify/   chunking, slices, resumable runner notebooks/   Kaggle notebook for the 8B judge
src/train/    fine-tuning + augmentation         legacy/      v1 pipeline (reference only)
src/eval/     metrics, thresholds, evaluation
src/display/  report generator
```

`data/`, `models/` and `results/` are regenerated and git-ignored. Fine-tuned weights are published to the Hugging
Face Hub with `scripts/upload_models.py`.

## Data notes

- **PHANTOM labels** are the strings "hallucination" / "not hallucination". File names are case-sensitive
  (`Phantom_10k_seed.csv`, `Phantom_def14A_2000tokens_middle.csv`).
- **DEF 14A long files** regenerate the answers for about a third of the pairs relative to the seed; these rows are tagged
  `variant=reanswered`. The 10-K long files match the seed exactly.
- **Seed passages** are about 600 tokens at the median and up to ~1,700 under encoder tokenizers, so they don't fit a
  512-token window. The "30000-token" files are 35-45K tokens. See `artifacts/data_card.md`.

## License

Academic use (Dartmouth College coursework). Datasets and models keep their own licenses.
