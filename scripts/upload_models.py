"""Publish fine-tuned checkpoints from models/<name>/seed<k>/ to the Hugging Face Hub.

Every checkpoint written by `python -m src.cli finetune` is uploaded as
``<username>/fasthallucheck-<name>-seed<k>``, together with its fhc_meta.json (base model revision,
training config, validation metrics). Collaborators can then score without retraining by pointing a
scoring config's ``model:`` at the Hub id.

Usage:
    huggingface-cli login                                         # one-time auth (token with write access)
    python scripts/upload_models.py --username <HF_USERNAME>                  # every checkpoint found
    python scripts/upload_models.py --username <HF_USERNAME> --only modernbert-nli-ft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
IGNORE_PATTERNS = ["checkpoints/*", "checkpoint-*", "training_args.bin"]


def find_checkpoints(only: list[str] | None) -> list[Path]:
    dirs = sorted(p.parent for p in MODELS_DIR.glob("*/seed*/config.json"))
    return [d for d in dirs if not only or d.parent.name in only]


def upload(ckpt: Path, username: str, private: bool) -> str:
    from huggingface_hub import HfApi, create_repo

    repo_id = f"{username}/fasthallucheck-{ckpt.parent.name}-{ckpt.name}"
    create_repo(repo_id, exist_ok=True, private=private)
    HfApi().upload_folder(folder_path=str(ckpt), repo_id=repo_id, ignore_patterns=IGNORE_PATTERNS,
                          commit_message=f"Upload {ckpt.parent.name}/{ckpt.name}")
    print(f"uploaded {ckpt.relative_to(PROJECT_ROOT)} -> https://huggingface.co/{repo_id}")
    return repo_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", required=True, help="Hugging Face Hub username or organisation")
    parser.add_argument("--only", nargs="*", default=None, help="model names under models/ to upload")
    parser.add_argument("--private", action="store_true", help="create private repos")
    args = parser.parse_args()
    ckpts = find_checkpoints(args.only)
    if not ckpts:
        print("error: no checkpoints under models/*/seed*/ — run `python -m src.cli finetune ...` first", file=sys.stderr)
        return 1
    for c in ckpts:
        upload(c, args.username, args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
