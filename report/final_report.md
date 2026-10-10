# Length-robust hallucination detection on PHANTOM, RAGTruth and HaluEval

_Generated 2026-10-10 by `python -m src.cli report` from the committed `artifacts/`._

**Pipeline status.** Done: prepare, baselines, score-smoke, evaluate, report. Pending: rewrite-halueval, probe-api, score-zeroshot, judges-api, finetune, score-finetuned, judge-local. Sections that depend on pending stages say so explicitly.

## Research question

How do hallucination detectors degrade as the context grows from a ~500-token passage to 30K nominal tokens and as the evidence moves to the beginning, middle or end, and does chunked or retrieval-based verification recover the loss? Every detector is evaluated next to answer-only shortcut baselines, because all three benchmarks can be partly solved without reading the context.

## Shortcut audit

AUROC of baselines trained on each benchmark's training split. `answer_length` and `answer_tfidf` never see the context; `lexical_support` is 1 minus the share of the answer's content words found in the context. PHANTOM baselines are trained on 8-K/497K and tested on 10-K/DEF 14A at every length.

| eval | answer_length | answer_tfidf | lexical_support |
|---|---|---|---|
| halueval_test | 0.982 | 0.968 | 0.895 |
| phantom_grid_full | 0.835 | 0.957 | 0.836 |
| ragtruth_test | 0.685 | 0.808 | 0.783 |

A detector only demonstrates that it uses the context where it beats the answer-only line, or on the length-matched hard subset where the answer-only signal is weaker.

## Data

- HaluEval QA: 10000 questions (20000 rows). RAGTruth: 14942 train / 2675 test responses.
- PHANTOM: 4 seed files plus 30 long-context files (10-K and DEF 14A at 2K-30K tokens x 3 positions). Long rows are joined to their seed pair; 2086 of 13462 long-file pairs have answers regenerated relative to the seed (`variant=reanswered`; 10k: 0 of 6757 (0%); def14a: 2086 of 6705 (31%)).
- Splits are grouped by pair: PHANTOM 10-K and DEF 14A are test-only at every length; 8-K/497K are train/val; HaluEval 70/15/15; RAGTruth official test, train split 85/15. `grid_sub` = 200 pairs per test filing present in every cell.
- Full statistics and tokenizer measurements: [`artifacts/data_card.md`](../artifacts/data_card.md).

## Results

AUROC with 95% pair-clustered bootstrap CIs. Thresholds for F1 and balanced accuracy are chosen on validation slices only.

| detector | strategy | halueval_test | phantom_grid_full | phantom_grid_full (hard) | ragtruth_test |
|---|---|---|---|---|---|
| answer_length | none | 0.982 [0.977, 0.986] | 0.835 [0.820, 0.850] | 0.732 [0.713, 0.755] | 0.685 [0.663, 0.707] |
| answer_tfidf | none | 0.968 [0.962, 0.974] | 0.957 [0.950, 0.965] | 0.888 [0.869, 0.908] | 0.808 [0.787, 0.828] |
| lexical_support | full_context | 0.895 [0.881, 0.907] | 0.836 [0.819, 0.852] | 0.814 [0.796, 0.832] | 0.783 [0.764, 0.803] |

## Fine-tuning

_Pending: GPU stage `finetune` (see HANDOFF.md)._

## Discussion

_To be written once the GPU stages are complete. It must discuss each result against the answer-only line and the hard subset, not in isolation._

## Limitations

- All three benchmarks contain answer-only signal (see the shortcut audit); absolute scores overstate grounding.
- PHANTOM's long variants are built around a single 500-token evidence passage, so `min_phalluc` (any window supports the answer) suits it; claims needing evidence from several windows are not tested.
- 512-token models (DeBERTa) cannot hold most seed passages in one window; their results are partial-evidence results.
- `grid_sub` uses 200 pairs per filing per cell; CIs reflect this.
- API judges change over time; the API results are a secondary comparison and are pinned to the model ids in the manifest.

## Reproducibility

Pinned revisions:

- `pminervini/HaluEval` @ `12a856119f03`
- `seyled/Phantom_Hallucination_Detection` @ `11406c015385`
- `wandb/RAGTruth-processed` @ `eb4f4b9d1b68`
- `vectara/hallucination_evaluation_model` @ `de9a5e6ba9ba`
- `tasksource/ModernBERT-base-nli` @ `de4ab7e77845`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` @ `6f5cf0a2b59c`
- `KRLabsOrg/lettucedect-base-modernbert-en-v1` @ `bbd77832f52f`
- `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c0`
- `Qwen/Qwen3-8B` @ `b968826d9c46`

Run `python -m src.cli status` to see how each artifact is produced.
