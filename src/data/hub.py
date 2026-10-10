"""Download files from the Hugging Face Hub at pinned revisions."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import DATASETS

logger = logging.getLogger(__name__)


def download_dataset_file(dataset: str, filename: str) -> Path:
    """Download ``filename`` from the pinned revision of ``dataset`` (a key of config.DATASETS).

    Uses the standard HF cache, so repeated calls are free.
    """
    from huggingface_hub import hf_hub_download

    spec = DATASETS[dataset]
    logger.info("Fetching %s/%s @ %s", spec["repo_id"], filename, spec["revision"][:8])
    return Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=filename,
            repo_type="dataset",
            revision=spec["revision"],
        )
    )
