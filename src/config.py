"""Project-wide constants: paths, pinned Hugging Face revisions, seeds and logging setup.

Everything that identifies *which* data or weights a result came from lives here,
so that a result can always be traced back to an exact revision.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FHC_DATA_DIR", ROOT / "data"))          # regenerated, git-ignored
ARTIFACTS_DIR = Path(os.environ.get("FHC_ARTIFACTS_DIR", ROOT / "artifacts"))  # committed
MODELS_DIR = Path(os.environ.get("FHC_MODELS_DIR", ROOT / "models"))      # git-ignored, pushed to HF Hub
RESULTS_DIR = Path(os.environ.get("FHC_RESULTS_DIR", ROOT / "results"))   # scratch shards, git-ignored
CONFIGS_DIR = ROOT / "configs"
REPORT_DIR = ROOT / "report"

SEED = 42

# Pinned dataset revisions (commit SHAs on the Hugging Face Hub, checked 2026-10-10).
DATASETS = {
    "halueval": {"repo_id": "pminervini/HaluEval", "revision": "12a856119f03975a94509091e8cada3e6be6ead7"},
    "phantom": {"repo_id": "seyled/Phantom_Hallucination_Detection", "revision": "11406c015385efae07977feecd7b7e753e0b1d30"},
    "ragtruth": {"repo_id": "wandb/RAGTruth-processed", "revision": "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b"},
}

# Pinned model revisions.
MODELS = {
    "hhem": {"repo_id": "vectara/hallucination_evaluation_model", "revision": "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"},
    "modernbert_nli": {"repo_id": "tasksource/ModernBERT-base-nli", "revision": "de4ab7e77845098b7fab7f6ab9d370ddff27b19c"},
    "deberta_nli": {"repo_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "revision": "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"},
    "lettucedetect": {"repo_id": "KRLabsOrg/lettucedect-base-modernbert-en-v1", "revision": "bbd77832f52f9bd87546a3924c032467921f5c34"},
    "qwen3_4b": {"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
    "qwen3_8b": {"repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
}


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with timestamps."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")


def get_device(requested: str = "auto") -> str:
    """Return 'cuda' when available (or requested), else 'cpu'. Logs the choice."""
    log = logging.getLogger(__name__)
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"Unknown device {requested!r}")
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = "cuda" if (requested in ("auto", "cuda") and has_cuda) else "cpu"
    log.info("Using device: %s", device)
    return device
