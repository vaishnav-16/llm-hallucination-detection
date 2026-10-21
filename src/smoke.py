"""`smoke` stage: a tiny end-to-end run of every stage, writing only to scratch folders under results/.

Run it on CPU in the cloud session before handing off, and with ``--device cuda`` on the laptop before
the real GPU stages. It checks plumbing, not quality: the numbers it produces are meaningless.
"""

from __future__ import annotations

import logging

import yaml

from src.config import RESULTS_DIR, ROOT

logger = logging.getLogger(__name__)


def run_smoke(device: str = "cpu") -> None:
    from src.eval.evaluate import evaluate_all
    from src.train.finetune import finetune
    from src.verify.runner import run_config

    logger.info("[smoke 1/3] scoring configs/smoke.yaml")
    run_config(ROOT / "configs" / "smoke.yaml", device=device)

    logger.info("[smoke 2/3] evaluating scratch predictions")
    out = RESULTS_DIR / "scratch_metrics"
    out.mkdir(parents=True, exist_ok=True)
    evaluate_all(n_boot=50, n_boot_cells=20, pred_dir=RESULTS_DIR / "scratch_predictions", out_dir=out)

    logger.info("[smoke 3/3] 4-step fine-tune of ModernBERT-NLI on 64 rows")
    cfg = yaml.safe_load((ROOT / "configs" / "finetune_modernbert.yaml").read_text(encoding="utf-8"))
    cfg.update({"name": "smoke-ft", "max_length": 512, "max_train_rows": 64, "max_val_rows": 32, "batch_size": 2,
                "eval_batch_size": 4, "grad_accum": 1, "augment": {"fraction": 0.5, "target_tokens": [400]}})
    finetune(cfg, seed=42, device=device, max_steps=4, models_dir=RESULTS_DIR / "scratch_models")
    logger.info("Smoke run complete. Outputs are under %s", RESULTS_DIR)
