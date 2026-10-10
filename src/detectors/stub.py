"""Deterministic, dependency-free detector for smoke tests and unit tests.

Scores a pair by the share of hypothesis words missing from the premise, so it behaves like a weak
lexical verifier: supported answers score low, unsupported ones high.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from src.detectors.base import Detector

_W = re.compile(r"[a-z0-9]+")


class StubDetector(Detector):
    requires = "cpu"

    def __init__(self, name: str = "stub", max_length: int = 512):
        self.name, self.max_length = name, max_length

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = []
        for p, h in zip(premises, hypotheses):
            words = _W.findall(h.lower())
            vocab = set(_W.findall(p.lower()))
            out.append(1.0 - (sum(w in vocab for w in words) / len(words) if words else 0.0))
        return np.asarray(out)
