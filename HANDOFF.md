# Handoff: running the GPU stages on the RTX 4070 laptop (Windows)

The cloud session (CPU only) builds the code, prepares the data, runs the shortcut baselines and any API
judges, and writes the report. This laptop runs everything that needs a GPU. The two sides exchange work
through git: the laptop commits small result files under `artifacts/`, never datasets or weights.

`python -m src.cli status` always prints what is left, what each stage needs (cpu / gpu / api) and the
exact command to run it.

## 1. One-time setup (PowerShell)

```powershell
git clone https://github.com/vaishnav-16/llm-hallucination-detection.git
cd llm-hallucination-detection
git checkout claude/nifty-turing-2tmmro        # or: git pull, if already cloned

python -m venv .venv
.venv\Scripts\Activate.ps1

# 1) A CUDA build of PyTorch. Pick the CUDA version from the selector at https://pytorch.org/get-started/locally/
#    (an RTX 4070 works with any current CUDA 12.x build), for example:
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2) The project with GPU extras (accelerate, bitsandbytes), API client, dashboard and pytest:
pip install -e ".[gpu,api,dashboard,dev]"

# Optional but recommended: a Hugging Face token makes the ~2 GB download faster and is needed to upload models.
$env:HF_TOKEN = "hf_..."            # or: huggingface-cli login
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

Check that CUDA is visible: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

## 2. Prepare data and check the install (~5 min)

```powershell
python -m src.cli prepare        # downloads HaluEval, RAGTruth, all 34 PHANTOM files; keeps the committed splits
python -m pytest                 # offline unit tests, ~10 s
python -m src.cli smoke --device cuda   # every stage on a few rows, scratch output only, ~5 min
python -m src.cli status
```

`prepare` refuses to continue if the downloaded data no longer matches `artifacts/splits.json`. Never pass
`--rebuild-splits` unless you mean to change the committed splits.

## 3. GPU stages, in priority order

All scoring runs resume where they stopped: Ctrl+C is safe, and rerunning the same command skips finished
cells. Times are rough estimates for the RTX 4070 laptop. After the first cell of each config, compare the
logged pairs/s with the estimate.

| # | Stage | Command | Est. time | Priority |
|---|---|---|---|---|
| 1 | Fine-tune ModernBERT-NLI (3 seeds) | `python -m src.cli finetune --config configs/finetune_modernbert.yaml --device cuda` | 1-2 h / seed | core |
| 2 | Zero-shot grid (HHEM, ModernBERT-NLI, LettuceDetect, DeBERTa-NLI + ablations) | `python -m src.cli score --config configs/zeroshot_grid.yaml --device cuda` | 6-7 h | core |
| 3 | Fine-tune DeBERTa-NLI (512-token comparison) | `python -m src.cli finetune --config configs/finetune_deberta.yaml --device cuda` | 0.5-1 h | core |
| 4 | Score fine-tuned models | `python -m src.cli score --config configs/finetuned_grid.yaml --device cuda` | 8-15 h | core (`--only modernbert_ft_s42 modernbert_ft_s42_ablate deberta_ft_s42` first) |
| 5 | Qwen3-4B judge (4-bit) | `python -m src.cli score --config configs/judge_qwen4b.yaml --device cuda` | 5-7 h | optional |
| 6 | Qwen3-8B judge on 20K/30K cells | Kaggle: `notebooks/kaggle_judge.ipynb` | ~13 h T4 | optional |

Seeds 43/44 and `modernbert_ft_s42_full` (every PHANTOM row) are the expensive parts of stage 4. Run them last
or overnight. To try a single cheap job first: `python -m src.cli score --config configs/zeroshot_grid.yaml --device cuda --only deberta_nli`.

**Before stage 1, run a 20-step fine-tune** to check memory: `python -m src.cli finetune --config configs/finetune_modernbert.yaml --device cuda --seeds 42 --max-steps 20`.
If it runs out of memory, lower `max_length` to 3072 in the config, or `batch_size` to 1 with `grad_accum` 16.

## 4. After each stage: commit results, upload weights

```powershell
git add artifacts/
git commit -m "Add <stage> predictions"
git push
python scripts/upload_models.py --username <your-hf-username>     # after fine-tuning only
```

Only `artifacts/` (predictions, manifests, training logs, metrics) goes into git. `data/`, `models/` and
`results/` are git-ignored on purpose.

## 5. Evaluate and report (any machine, CPU is fine)

```powershell
python -m src.cli evaluate      # metrics with CIs, paired comparisons -> artifacts/metrics/
python -m src.cli report        # figures + report/final_report.md
streamlit run app.py            # dashboard
```

## 6. Optional API stages (no GPU; can run in the cloud session)

1. Create `.env` in the repo root (git ignores it):
   ```
   DARTMOUTH_CHAT_API_KEY=...     # chat.dartmouth.edu > Settings > Account > API keys
   GEMINI_API_KEY=...             # https://aistudio.google.com/apikey
   ```
2. `python -m src.cli probe-api` lists the available models. Pin `model:` for each provider in `configs/providers.yaml`
   (prefer a dated id), then run `probe-api` again to check logprobs and long-context support.
3. `python -m src.cli rewrite-halueval --provider dartmouth` builds the normalized HaluEval variant (~10K calls).
   Then rerun `python -m src.cli baselines`: the answer-only baseline on `halueval_norm_test` should fall close to 0.5.
4. `python -m src.cli score --config configs/judges_api.yaml`.

## Troubleshooting

- **CUDA out of memory while scoring:** the runner halves the batch size automatically. If it still fails at batch
  size 1, lower `max_length` or `window` for that run in the config. Changing a run's config invalidates only that run's
  cached results.
- **`bitsandbytes` errors on Windows:** `pip install -U bitsandbytes` (0.43+ ships Windows wheels). Only the optional
  Qwen judge uses it.
- **A stage fails with `SchemaError`:** the data no longer matches what the loaders expect. Don't silence it; report it.
- **Slow first run:** model weights download on first use and go into the Hugging Face cache.
