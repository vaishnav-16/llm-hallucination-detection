"""Publish fine-tuned FastHalluCheck models to the Hugging Face Hub.

Owner-side utility — run once after fine-tuning to share weights with
collaborators so they can skip the ~15-minute training step.

Usage:
    huggingface-cli login                                    # one-time auth
    python scripts/upload_models.py --username <HF_USERNAME>
    python scripts/upload_models.py --username <HF_USERNAME> --model halueval
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "halueval": {
        "local_path": PROJECT_ROOT / "models" / "deberta-hallucination-detector",
        "repo_suffix": "fasthallucheck-deberta-halueval",
    },
    "phantom": {
        "local_path": PROJECT_ROOT / "models" / "deberta-phantom-finetuned",
        "repo_suffix": "fasthallucheck-deberta-phantom",
    },
}

IGNORE_PATTERNS = ["checkpoint-*", "training_args.bin"]


def upload(model_key: str, username: str) -> None:
    cfg = MODELS[model_key]
    local_path = cfg["local_path"]
    repo_id = f"{username}/{cfg['repo_suffix']}"

    if not local_path.exists():
        print(f"[skip] {model_key}: {local_path} does not exist")
        return

    print(f"[{model_key}] creating repo {repo_id}")
    create_repo(repo_id, exist_ok=True)

    print(f"[{model_key}] uploading {local_path} -> {repo_id}")
    api = HfApi()
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload FastHalluCheck fine-tuned DeBERTa",
    )
    print(f"[{model_key}] done -> https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", required=True, help="Hugging Face Hub username")
    parser.add_argument(
        "--model",
        choices=["halueval", "phantom", "both"],
        default="both",
        help="Which fine-tuned model to upload (default: both)",
    )
    args = parser.parse_args()

    keys = ["halueval", "phantom"] if args.model == "both" else [args.model]
    missing = [k for k in keys if not MODELS[k]["local_path"].exists()]
    if missing:
        print(f"error: missing model folders: {missing}", file=sys.stderr)
        print("Run `python run_all.py` first to fine-tune the models.", file=sys.stderr)
        return 1

    for k in keys:
        upload(k, args.username)
    return 0


if __name__ == "__main__":
    sys.exit(main())
