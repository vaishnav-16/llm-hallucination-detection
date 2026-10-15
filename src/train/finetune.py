"""Fine-tune an encoder as a binary hallucination classifier (label 1 = hallucinated).

Design choices (each fixes a problem in the old pipeline):
- Model selection uses the validation slices only; test slices are never loaded here.
- Training starts from an NLI checkpoint. The 3-way NLI head is mapped to a 2-way head
  (faithful <- entailment, hallucinated <- mean of neutral and contradiction), so step 0 already
  behaves like the zero-shot NLI model instead of a random classifier.
- The original HaluEval answers are excluded by default, because they teach the answer-length shortcut.
  The normalized HaluEval variant is used if it exists.
- Distractor augmentation (``augment.py``) pads training contexts to 2-3.5K tokens with unrelated
  training passages.
- Training sequences are bucketed by length (``train_sampling_strategy='group_by_length'``), with bf16
  and gradient checkpointing, so 4K-token training fits in 8 GB.

Outputs ``models/<name>/seed<k>/`` (weights + tokenizer + ``fhc_meta.json``) and
``artifacts/training/<name>_seed<k>.json`` (config, data counts, validation curve).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import ARTIFACTS_DIR, MODELS, MODELS_DIR, RESULTS_DIR, ROOT, get_device, seed_everything
from src.data.store import write_json
from src.detectors.base import fit_claim
from src.eval.metrics import auroc, threshold_metrics
from src.train.augment import add_distractors
from src.verify.chunking import hypothesis_text
from src.verify.slices import get_slice

logger = logging.getLogger(__name__)

SCRATCH_MODELS_DIR = RESULTS_DIR / "scratch_models"  # step-capped runs; git-ignored, never scored or uploaded
MIN_VAL_AUROC = 0.6  # default floor; the NLI-initialised head already starts well above chance


class TrainingError(RuntimeError):
    """A fine-tune finished but its best validation AUROC is at chance (a collapsed or broken model)."""


def check_val_auroc(best: dict, label: str, floor: float = MIN_VAL_AUROC, history: list[dict] | None = None) -> None:
    """Raise if the selected checkpoint's validation AUROC is missing, NaN or at most ``floor``.

    The first DeBERTa fine-tune collapsed to a constant output (val AUROC 0.5) and was still saved, uploaded
    and scored. A run that fails this check is not saved under ``models/``, so it can't be scored or uploaded.
    """
    auc = best.get("eval_auroc")
    if auc is None or not auc > floor:  # `not >` also catches NaN
        curve = [round(h["eval_auroc"], 4) for h in history or [] if "eval_auroc" in h]
        raise TrainingError(f"{label}: best val AUROC {auc} is not above {floor} (per-eval curve {curve}); "
                            "model not saved. Checkpoints are left in the run's checkpoints/ folder for inspection.")


def _load_slices(names: list[str], max_rows: int | None = None) -> pd.DataFrame:
    frames = []
    for name in names:
        try:
            frames += [p.loader() for p in get_slice(name)]
        except FileNotFoundError as e:
            logger.warning("skipping slice %s: %s", name, e)
    df = pd.concat(frames, ignore_index=True)
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=0)
    return df


def _binary_head_from_nli(model, nli_labels: dict[int, str]) -> None:
    """Replace a 3-way NLI classifier with a 2-way one initialised from it (in place)."""
    import torch

    head = model.classifier
    ent = [i for i, v in nli_labels.items() if "entail" in v and "not" not in v][0]
    others = [i for i in nli_labels if i != ent]
    new = torch.nn.Linear(head.in_features, 2, bias=head.bias is not None, dtype=head.weight.dtype)
    with torch.no_grad():
        new.weight[0] = head.weight[ent]
        new.weight[1] = head.weight[others].mean(dim=0)
        if head.bias is not None:
            new.bias[0] = head.bias[ent]
            new.bias[1] = head.bias[others].mean()
    model.classifier = new
    model.config.num_labels = 2
    model.config.id2label = {0: "faithful", 1: "hallucinated"}
    model.config.label2id = {"faithful": 0, "hallucinated": 1}
    model.num_labels = 2


def load_model_for_training(base: str, extend_positions: int | None = None):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    revision = None
    if base in MODELS:
        revision, base = MODELS[base]["revision"], MODELS[base]["repo_id"]
    cfg = AutoConfig.from_pretrained(base, revision=revision)
    if extend_positions:
        if cfg.model_type != "modernbert":
            raise ValueError("extend_positions is only supported for ModernBERT (RoPE)")
        cfg.max_position_embeddings = extend_positions
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    # fp32 master weights: transformers 5 loads the stored dtype, and the DeBERTa NLI checkpoint is stored in fp16.
    # Optimiser steps on fp16 weights collapsed that fine-tune to a constant output. bf16 autocast still runs the math.
    model = AutoModelForSequenceClassification.from_pretrained(base, revision=revision, config=cfg, dtype=torch.float32)
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if model.config.num_labels == 3 and any("entail" in v for v in labels.values()):
        _binary_head_from_nli(model, labels)
        logger.info("Initialised binary head from NLI head (%s)", labels)
    elif model.config.num_labels != 2:
        raise ValueError(f"{base}: expected an NLI (3-label) or binary checkpoint, got {labels}")
    return tok, model, {"repo_id": base, "revision": revision}


def ablate_contexts(cfg: dict, *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Answer-only control (``ablate_context: true``): blank every context so the model sees only question + answer.

    This is the same input an ``ablate_context`` score run gives a detector, so the control is scored the same way.
    It measures how far a fine-tuned encoder gets on answer-side cues alone, a stronger line than answer TF-IDF.
    """
    if not cfg.get("ablate_context"):
        return list(frames)
    if cfg.get("augment", {}).get("fraction", 0) > 0:
        raise ValueError(f"{cfg['name']}: ablate_context and distractor augmentation are mutually exclusive")
    return [f.assign(context="") for f in frames]


class _PairDataset:
    def __init__(self, df: pd.DataFrame, tok, max_length: int, hyp_fmt: str):
        hyps, n_cut = [], 0
        for q, r, t in zip(df["question"], df["response"], df["task_type"]):
            h, cut = fit_claim(tok, hypothesis_text(q, r, t, hyp_fmt), max_length)
            hyps.append(h)
            n_cut += cut
        if n_cut:
            logger.warning("%d of %d claims longer than the %d-token window were cut to fit", n_cut, len(hyps), max_length)
        self.enc = tok(df["context"].tolist(), hyps, truncation="only_first", max_length=max_length)
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=float)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
    m = threshold_metrics(labels, p, 0.5)
    return {"auroc": auroc(labels, p), "f1": m["f1"], "balanced_accuracy": m["balanced_accuracy"]}


def _compat_args(**kw) -> dict:
    """Translate argument names between transformers 4.x and 5.x TrainingArguments."""
    import inspect

    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    warmup = kw.pop("warmup")
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = warmup
    else:  # v5: a float in [0, 1) is a ratio
        kw["warmup_steps"] = warmup
    if kw.pop("length_bucketing"):
        if "train_sampling_strategy" in params:
            kw["train_sampling_strategy"] = "group_by_length"
        else:
            kw["group_by_length"] = True
    return {k: v for k, v in kw.items() if k in params or k == "output_dir"}


def finetune(cfg: dict, seed: int, device: str = "auto", max_steps: int | None = None, models_dir: Path = MODELS_DIR) -> Path:
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    seed_everything(seed)
    dev = get_device(device)
    name = cfg["name"]
    out_dir = Path(models_dir) / name / f"seed{seed}"
    tok, model, base_info = load_model_for_training(cfg["base_model"], cfg.get("extend_positions"))

    train = _load_slices(cfg["train_slices"], cfg.get("max_train_rows"))
    oversample = cfg.get("oversample", {})
    for ds, k in oversample.items():
        extra = train[train["dataset"] == ds]
        train = pd.concat([train] + [extra] * (int(k) - 1), ignore_index=True)
    aug = cfg.get("augment", {})
    if aug.get("fraction", 0) > 0:
        train = add_distractors(train, aug["fraction"], tuple(aug.get("target_tokens", (2000, 3500))), seed)
    val = _load_slices(cfg["val_slices"], cfg.get("max_val_rows"))
    train, val = ablate_contexts(cfg, train, val)
    if (train["split"] != "train").any() or (val["split"] != "val").any():
        raise RuntimeError("finetune: train/val slices must contain only train/val rows")
    logger.info("train rows %d (%s), val rows %d", len(train), train["dataset"].value_counts().to_dict(), len(val))

    ml, fmt = int(cfg.get("max_length", 4096)), cfg.get("hypothesis", "qa")
    train_ds, val_ds = _PairDataset(train, tok, ml, fmt), _PairDataset(val, tok, ml, fmt)
    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    args = TrainingArguments(**_compat_args(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=float(cfg.get("epochs", 2)),
        max_steps=max_steps or -1,
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 4)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 8)),
        learning_rate=float(cfg.get("learning_rate", 3e-5)),
        warmup=float(cfg.get("warmup_ratio", 0.1)),
        weight_decay=0.01,
        bf16=bf16,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", dev == "cuda")),
        length_bucketing=True,
        eval_strategy="steps" if max_steps else "epoch",
        eval_steps=max_steps or None,
        save_strategy="steps" if max_steps else "epoch",
        save_steps=max_steps or None,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,  # Windows-safe
        use_cpu=dev == "cpu",
    ))
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok), processing_class=tok, compute_metrics=_compute_metrics,
    )
    t0 = time.time()
    trainer.train()
    best = trainer.evaluate()
    if not max_steps:  # a 20-step memory check isn't expected to learn anything
        check_val_auroc(best, f"{name} seed {seed}", float(cfg.get("min_val_auroc", MIN_VAL_AUROC)), trainer.state.log_history)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    meta = {
        "name": name, "seed": seed, "base": base_info, "config": cfg, "device": dev, "bf16": bf16,
        "train_rows": int(len(train)), "train_by_dataset": train["dataset"].value_counts().to_dict(),
        "val_rows": int(len(val)), "best_val": best, "train_minutes": round((time.time() - t0) / 60, 1),
        "max_steps": max_steps, "log_history": trainer.state.log_history,
    }
    (out_dir / "fhc_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "log_history"}, indent=2, default=str))
    if not max_steps:  # smoke runs do not write committed artifacts
        write_json(meta, ARTIFACTS_DIR / "training" / f"{name}_seed{seed}.json")
    import shutil

    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    logger.info("Saved %s (val auroc %.4f) in %.1f min", out_dir, best.get("eval_auroc", float("nan")), meta["train_minutes"])
    return out_dir


def is_trained(out_dir: Path) -> bool:
    """True if ``out_dir`` holds a finished full run (not a step-capped smoke run)."""
    meta = Path(out_dir) / "fhc_meta.json"
    return meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("max_steps") is None


def finetune_from_config(path: str, device: str = "auto", seeds: list[int] | None = None, max_steps: int | None = None,
                         force: bool = False, models_dir: Path = MODELS_DIR) -> None:
    """Train every seed in a config. Rerunning the same command skips seeds that already finished.

    Step-capped runs (``max_steps``) are memory/smoke checks: they go to ``results/scratch_models`` so a
    20-step checkpoint can never be scored or uploaded in place of the real model under ``models/``.
    """
    p = Path(path)
    if not p.exists():
        p = ROOT / path
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    out_root = SCRATCH_MODELS_DIR if max_steps else Path(models_dir)
    for seed in seeds or cfg.get("seeds", [42]):
        if not max_steps and not force and is_trained(out_root / cfg["name"] / f"seed{seed}"):
            logger.info("Skipping %s seed %d: already trained (pass --force to retrain)", cfg["name"], seed)
            continue
        finetune(cfg, seed, device, max_steps, models_dir=out_root)
