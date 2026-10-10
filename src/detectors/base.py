"""Detector interface and the shared batching loop.

Score convention, used everywhere: ``p_halluc`` in [0, 1], where higher means more likely hallucinated.
Adapters convert their model's native output once, in their own code. Nothing in the pipeline
ever inspects gold labels to decide which way a score points.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from src.verify.chunking import SpanFn, whitespace_spans

logger = logging.getLogger(__name__)


class ScoringError(RuntimeError):
    """A batch could not be scored even at batch size 1."""


class Detector(ABC):
    """A model that scores (premise, hypothesis) pairs."""

    name: str = "detector"
    requires: str = "cpu"          # cpu | gpu | api: where full-size runs are expected to happen
    uses_context: bool = True      # False for answer-only shortcut baselines
    max_length: int = 512          # model input limit in tokens (premise + hypothesis + specials)

    def span_fn(self) -> SpanFn:
        """Token spans used for chunking. Model detectors override this with their own tokenizer."""
        return whitespace_spans

    def count_tokens(self, text: str) -> int:
        return len(self.span_fn()(text))

    @abstractmethod
    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Return p_halluc for each pair."""

    def describe(self) -> dict:
        return {"name": self.name, "requires": self.requires, "max_length": self.max_length}


def run_batched(
    items: Sequence,
    fn: Callable[[Sequence], np.ndarray],
    batch_size: int,
    lengths: Sequence[int] | None = None,
    desc: str = "scoring",
) -> np.ndarray:
    """Apply ``fn`` to batches of ``items``, halving the batch size on CUDA OOM.

    Items are processed in descending length order (less padding) and results are returned in the
    original order. A batch that fails at batch size 1 raises ScoringError: no silent default scores.
    """
    n = len(items)
    out = np.full(n, np.nan, dtype=float)
    order = np.argsort(-np.asarray(lengths), kind="stable") if lengths is not None else np.arange(n)
    i, bs = 0, max(1, batch_size)
    t0, last_log = time.time(), time.time()
    while i < n:
        idx = order[i : i + bs]
        try:
            out[idx] = np.asarray(fn([items[j] for j in idx]), dtype=float)
            i += len(idx)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError subclasses RuntimeError
            if "out of memory" not in str(e).lower():
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
            if bs == 1:
                raise ScoringError(f"{desc}: out of memory at batch size 1 (item length {lengths[idx[0]] if lengths is not None else '?'})") from e
            bs = max(1, bs // 2)
            logger.warning("%s: CUDA OOM, batch size -> %d", desc, bs)
        if time.time() - last_log > 30:
            rate = i / max(1e-9, time.time() - t0)
            logger.info("%s: %d/%d pairs (%.1f pairs/s, batch %d)", desc, i, n, rate, bs)
            last_log = time.time()
    if np.isnan(out).any():
        raise ScoringError(f"{desc}: {int(np.isnan(out).sum())} pairs produced NaN scores")
    return out
