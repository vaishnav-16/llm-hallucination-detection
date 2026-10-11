# CLAUDE.md: FastHalluCheck v2 (length-robust hallucination detection)

## What this project is

**Research question:** how do hallucination detectors degrade as the context grows from a ~500-token passage
to 30K nominal tokens and as the evidence moves (beginning / middle / end), and does chunked or
retrieval-based verification recover the loss? Every detector is judged against answer-only shortcut baselines.

- **Main benchmark:** PHANTOM (`seyled/Phantom_Hallucination_Detection`, Ji et al., NeurIPS 2025 D&B).
  10-K and DEF 14A are test-only at every length; 8-K and 497K are train/val (the paper's split).
- **Also:** RAGTruth (`wandb/RAGTruth-processed`, real LLM responses) and HaluEval QA (`pminervini/HaluEval`).
  HaluEval is a shortcut case study: answer length alone gives about 0.97 accuracy.
- v1 (HHEM + DeBERTa-small on 500 HaluEval questions) lives in `legacy/` for reference only. Its headline numbers
  are not valid (see README "Why v2").

## Critical rules

1. **Never ask the user for input mid-task.** Pick the most reasonable default and document it.
2. **Never peek at test labels.** Thresholds, calibration, ensemble weights and checkpoint selection use
   validation slices only. Score direction is fixed in each detector adapter (`p_halluc`, higher = hallucinated),
   never inferred from labels.
3. **No silent fallbacks.** A model that fails to load, a batch that fails at batch size 1, or a schema surprise
   must raise (`SchemaError`, `ScoringError`). Never substitute default scores or another model under the same name.
4. **Never train on PHANTOM 10-K or DEF 14A rows.** Their long variants reuse the seed questions and answers.
   Splits are grouped by `pair_id` and committed in `artifacts/splits.json`. Do not rebuild them casually.
5. **Every detector result is reported next to the answer-only baselines and on the hard subset.**
6. **Commit small results, never data or weights:** `artifacts/` and `report/` are committed; `data/`,
   `models/` and `results/` are git-ignored. Weights go to the HF Hub (`scripts/upload_models.py`).
7. **Log progress** with `logging` (not print) so long runs show pairs/s and the current cell.

## Two environments

| | Cloud session (Claude Code on the web) | Local laptop (Windows) |
|---|---|---|
| Hardware | 4 CPU cores, ~15 GB RAM, **no GPU** | RTX 4070 Mobile 8 GB, 16 GB RAM |
| Runs | code, `prepare`, `baselines`, API stages, `evaluate`, `report`, CPU `smoke` | `finetune`, GPU `score` configs, Qwen judge |
| Persistence | ephemeral: commit and push everything worth keeping | normal |

`python -m src.cli status` lists each stage, what it needs (cpu / gpu / api), whether it is done, and the
exact command. **`HANDOFF.md` is the step-by-step runbook for the GPU side.** On Windows use `python -m ...`
commands; there are no bash-only steps.

## Layout

```
configs/            YAML experiment configs (smoke, zeroshot_grid, finetune_*, finetuned_grid, judge_*, judges_api, providers)
src/config.py       paths, pinned HF revisions (datasets + models), seeds, device selection
src/data/           loaders (halueval, phantom, ragtruth), schema validation, splits, data card, answer rewrite
src/detectors/      base (p_halluc contract, OOM-halving batch loop), hf_models (HHEM, seq-cls, LettuceDetect),
                    llm_judge (local Qwen), api_judge (Dartmouth Chat / Gemini, cached), jev (TypeSafe Jev), baselines, stub
src/verify/         chunking (truncate / chunk / retrieve; min/mean/top-k aggregation), slices, runner (resumable)
src/train/          finetune (NLI-initialised binary head, val-only selection), augment (distractor padding)
src/eval/           metrics (AUROC, F1, ECE, pair-clustered bootstrap, McNemar), threshold, evaluate
src/display/        report.py (figures + report/final_report.md from artifacts only); figures.py (legacy helpers)
src/dashboard/, app.py   Streamlit dashboard (still reads v1 result files; to be ported to artifacts/metrics)
src/cli/            `python -m src.cli <command>`
tests/              offline pytest suite (no network)
artifacts/          committed: splits.json, data_card.md, predictions/*.parquet + manifest.json, metrics/, figures/, training/
legacy/             v1 scripts, untouched
```

## Pipeline

```
prepare -> baselines -> [rewrite-halueval] -> score (smoke | zeroshot_grid | judges_api)
        -> finetune -> score (finetuned_grid | judge_qwen4b) -> evaluate -> report
```

Scoring is config-driven and resumable per PHANTOM cell. A run's results are tied to a hash of its config, so a
changed config reruns only that run. `--limit` and `scratch: true` configs write to `results/scratch_*`, never to `artifacts/`.

## Key facts from the data (see `artifacts/data_card.md`)

- PHANTOM labels are the strings "hallucination" / "not hallucination". The v1 loader mapped every label to 0,
  and its file names had the wrong case (`10K` vs `10k`).
- Seed passages are ~560-640 tokens at the median and ~800-925 at p95 (max ~1,700) under encoder tokenizers.
  They do not fit DeBERTa's 512 window. The "30000-token" files are ~35-45K tokens.
- `tasksource/ModernBERT-base-nli` is configured for 2,048 positions. Fine-tuning extends it to 8,192
  (`extend_positions`, RoPE).
- DEF 14A long files regenerate the answers for about a third of the pairs (`variant=reanswered`). The 10-K long files match the seed exactly.
- Answer-only TF-IDF reaches about 0.96 AUROC on PHANTOM test filings and 0.81 on RAGTruth. The hard subset
  (faithful answer >= 224 chars) lowers PHANTOM's to about 0.89.

## TypeSafe Jev

- `src/detectors/jev.py`, config `configs/jev.yaml`, key `TYPESAFE_API_KEY` (in `.env` or the environment). No GPU.
  Docs: https://docs.typesafe.ai/llms.txt (append `.md` to any page path). Read the live API and
  jaggedness pages before changing the adapter.
- `POST https://api.typesafe.ai/v1/systemone`; model pinned to `jev-1.13.0` (`jev-latest` is a moving alias).
  32K tokens for state plus the longest question, 64K per request; $0.042 per million input tokens.
- Every request asks a `noul` and a `choice` per claim. All claims sharing a document chunk go in one request:
  questions are independent and the state is billed once. `jev_noul` and `jev_choice` send identical requests,
  so the second run is free from `artifacts/api_cache/jev.jsonl`.
- Spend is tracked in `artifacts/api_cache/jev_spend.json`; `max_usd` (4.5) stops runs before the $5 credit runs out.
  Always run `score --config configs/jev.yaml --dry-run` before a live run, and commit the cache with the predictions.

## Style

Module docstrings that explain why, `argparse` / CLI flags for anything configurable, `logging`, type hints,
`pathlib`, seed 42 (`src.config.seed_everything`), `num_workers=0` (Windows). Run `python -m pytest` before committing.
