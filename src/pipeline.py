"""Stage registry: what each pipeline stage needs (cpu / gpu / api), how to run it, and whether it is done.

``python -m src.cli status`` prints this table so that whichever session picks the project up
(the cloud CPU session or the local RTX 4070 session) knows exactly what is left and where it must run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import ARTIFACTS_DIR, CONFIGS_DIR, DATA_DIR, MODELS_DIR, REPORT_DIR

PRED_DIR = ARTIFACTS_DIR / "predictions"


@dataclass
class Stage:
    name: str
    requires: str            # cpu | gpu | api | cpu+api
    command: str
    outputs: list[Path] = field(default_factory=list)
    note: str = ""
    done_fn: Callable[[], bool] | None = None

    def done(self) -> bool:
        if self.done_fn is not None:
            return self.done_fn()
        return bool(self.outputs) and all(p.exists() for p in self.outputs)


def _preds_for(config: str) -> Callable[[], bool]:
    """A scoring config is done when the run manifest lists every run in it as complete."""
    def check() -> bool:
        from src.verify.runner import config_runs_complete

        return config_runs_complete(CONFIGS_DIR / config)
    return check


def _newer_than_predictions(path: Path) -> Callable[[], bool]:
    """Done only if ``path`` exists and was written after the latest prediction or baseline file."""
    def check() -> bool:
        if not path.exists():
            return False
        inputs = list(PRED_DIR.glob("*.parquet")) + [PRED_DIR / "manifest.json"]
        latest = max((p.stat().st_mtime for p in inputs if p.exists()), default=0.0)
        return path.stat().st_mtime >= latest
    return check


STAGES: list[Stage] = [
    Stage("prepare", "cpu", "python -m src.cli prepare",
          [DATA_DIR / "normalized" / "halueval.parquet", ARTIFACTS_DIR / "splits.json", ARTIFACTS_DIR / "data_card.md"],
          "downloads ~2 GB from the HF Hub; needed on every machine (data/ is not committed)"),
    Stage("baselines", "cpu", "python -m src.cli baselines", [ARTIFACTS_DIR / "metrics" / "shortcut_baselines.csv"]),
    Stage("rewrite-halueval", "api", "python -m src.cli rewrite-halueval --provider dartmouth",
          [ARTIFACTS_DIR / "rewrites" / "halueval_rewrites.jsonl"], "needs DARTMOUTH_CHAT_API_KEY (or GEMINI_API_KEY) in .env"),
    Stage("probe-api", "api", "python -m src.cli probe-api", [ARTIFACTS_DIR / "api_probe.json"]),
    Stage("score-smoke", "cpu", "python -m src.cli score --config configs/smoke.yaml", done_fn=_preds_for("smoke.yaml")),
    Stage("score-zeroshot", "gpu", "python -m src.cli score --config configs/zeroshot_grid.yaml --device cuda",
          done_fn=_preds_for("zeroshot_grid.yaml"), note="HHEM, ModernBERT-NLI, DeBERTa-NLI, LettuceDetect on grid_sub + test sets"),
    Stage("probe-jev", "api", "python -m src.cli probe-jev", [ARTIFACTS_DIR / "jev_probe.json"],
          "needs TYPESAFE_API_KEY; costs < $0.01"),
    Stage("judge-jev", "api", "python -m src.cli score --config configs/jev.yaml", done_fn=_preds_for("jev.yaml"),
          note="TypeSafe Jev; ~$2.60 (check with --dry-run); no GPU, runs in either environment"),
    Stage("judges-api", "api", "python -m src.cli score --config configs/judges_api.yaml", done_fn=_preds_for("judges_api.yaml")),
    Stage("finetune", "gpu", "python -m src.cli finetune --config configs/finetune_modernbert.yaml --device cuda",
          [MODELS_DIR / "modernbert-nli-ft" / "seed42" / "config.json"], "then finetune_deberta.yaml; push with scripts/upload_models.py"),
    Stage("score-finetuned", "gpu", "python -m src.cli score --config configs/finetuned_grid.yaml --device cuda",
          done_fn=_preds_for("finetuned_grid.yaml")),
    Stage("judge-local", "gpu", "python -m src.cli score --config configs/judge_qwen4b.yaml --device cuda",
          done_fn=_preds_for("judge_qwen4b.yaml"), note="optional; Qwen3-8B at 30K runs on Kaggle (notebooks/kaggle_judge.ipynb)"),
    Stage("evaluate", "cpu", "python -m src.cli evaluate", note="rerun after any new predictions",
          done_fn=_newer_than_predictions(ARTIFACTS_DIR / "metrics" / "metrics_long.csv")),
    Stage("report", "cpu", "python -m src.cli report", note="rerun after evaluate",
          done_fn=_newer_than_predictions(REPORT_DIR / "final_report.md")),
]


def status_rows() -> list[dict]:
    rows = []
    for s in STAGES:
        try:
            done = s.done()
        except Exception as e:  # a missing optional module should not break `status`
            done = False
            s = Stage(s.name, s.requires, s.command, s.outputs, f"{s.note} [status check failed: {e}]".strip())
        rows.append({"stage": s.name, "requires": s.requires, "done": done, "command": s.command, "note": s.note})
    return rows
