"""Length-agnostic verification: split a premise into token windows, optionally retrieve, aggregate.

Every detector runs through the same ``ChunkSpec`` at every context length, including PHANTOM seeds,
so a length curve varies only the context, never the scoring method.

Strategies
- ``truncate``  keep the first ``window`` tokens (the old pipeline's behaviour; the baseline).
- ``chunk``     overlapping windows over the whole premise; score every window, then aggregate.
- ``retrieve``  overlapping windows, keep the ``top_k`` windows by BM25 against question + answer, then aggregate.

Retrievers (``retriever``, only used by ``retrieve``)
- ``bm25``           BM25Okapi fitted on the overlapping windows themselves (the original behaviour, the default).
                     Its IDF counts a passage once per window that contains it, so with few, heavily overlapping
                     windows (a 2K-token PHANTOM context is ~4 windows of 1,500 with stride 750) a term in 2 of 3
                     windows gets a negative IDF: the evidence passage, sitting in the overlap, counts against the
                     windows that contain it. On PHANTOM 2K/middle the top-2 windows held the whole evidence
                     passage for 13% of pairs.
- ``bm25_blockidf``  the same BM25 window scoring, with IDF from non-overlapping blocks of ``stride // 3`` units,
                     so document frequency no longer depends on the overlap (2K/middle: 92%; longer contexts
                     unchanged within a few points).

Containment guarantee
With window W and overlap O (stride S = W - O), any contiguous span of at most O + 1 tokens lies
entirely inside at least one window (see ``guaranteed_span``). Choose O >= the measured p95 length
of the evidence passage to guarantee that the passage is seen whole.

Aggregation is named by its effect on ``p_halluc`` (higher = more likely hallucinated):
- ``min_phalluc``   the answer is faithful if ANY window supports it (MiniCheck / SummaC style "max support").
- ``mean_phalluc``  average over windows.
- ``topk_mean``     mean of the ``top_k`` most supportive (lowest) window scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

STRATEGIES = ("truncate", "chunk", "retrieve")
RETRIEVERS = ("bm25", "bm25_blockidf")
AGGREGATIONS = ("min_phalluc", "mean_phalluc", "topk_mean")

Span = tuple[int, int]


class SpanFn(Protocol):
    def __call__(self, text: str) -> list[Span]: ...


@dataclass(frozen=True)
class ChunkSpec:
    strategy: str = "chunk"
    window: int = 512
    overlap: int | None = None      # default: window // 2
    top_k: int = 3
    aggregation: str = "min_phalluc"
    retriever: str = "bm25"
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}")
        if self.retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}, got {self.retriever!r}")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if not 0 <= self.effective_overlap < self.window:
            raise ValueError("overlap must satisfy 0 <= overlap < window")

    @property
    def effective_overlap(self) -> int:
        return self.window // 2 if self.overlap is None else self.overlap

    @property
    def stride(self) -> int:
        return self.window - self.effective_overlap

    def tag(self) -> str:
        """Short identifier used in run names, e.g. chunk-w8000-o2000-min_phalluc."""
        if self.strategy == "truncate":
            return f"truncate-w{self.window}"
        base = f"{self.strategy}-w{self.window}-o{self.effective_overlap}-{self.aggregation}"
        base += f"-k{self.top_k}" if self.strategy == "retrieve" or self.aggregation == "topk_mean" else ""
        return base + ("-bidf" if self.strategy == "retrieve" and self.retriever == "bm25_blockidf" else "")


def guaranteed_span(window: int, overlap: int) -> int:
    """Longest span (in tokens) guaranteed to fall entirely inside some window."""
    return overlap + 1


def whitespace_spans(text: str) -> list[Span]:
    """Fallback 'tokenizer': one span per run of non-whitespace characters."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def hf_span_fn(tokenizer) -> SpanFn:
    """Character spans of each token under a Hugging Face *fast* tokenizer."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("chunking needs a fast tokenizer (offset mappings)")

    def spans(text: str) -> list[Span]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        return [(a, b) for a, b in enc["offset_mapping"] if b > a]

    return spans


def window_starts(n_tokens: int, window: int, stride: int) -> list[int]:
    """Start indices of windows covering [0, n_tokens). The last window is aligned to the end."""
    if n_tokens <= window:
        return [0]
    starts = list(range(0, n_tokens - window + 1, stride))
    if starts[-1] + window < n_tokens:
        starts.append(n_tokens - window)
    return starts


def split_windows(text: str, spans: Sequence[Span], window: int, stride: int) -> list[str]:
    """Slice ``text`` into token windows, preserving the original characters."""
    n = len(spans)
    if n <= window:
        return [text]
    return [text[spans[s][0] : spans[min(s + window, n) - 1][1]] for s in window_starts(n, window, stride)]


_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def _bm25_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def retrieve_top_k(chunks: Sequence[str], query: str, k: int) -> list[int]:
    """Indices (in original order) of the k chunks with the highest BM25 score for ``query``."""
    if len(chunks) <= k:
        return list(range(len(chunks)))
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([_bm25_tokens(c) or ["_"] for c in chunks])
    scores = bm25.get_scores(_bm25_tokens(query))
    top = np.argsort(-scores, kind="stable")[:k]
    return sorted(int(i) for i in top)


def retrieve_top_k_blockidf(chunks: Sequence[str], blocks: Sequence[str], query: str, k: int,
                            k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Like ``retrieve_top_k``, but document frequencies come from non-overlapping ``blocks`` of the same text.

    IDF uses the BM25+ style ``log(1 + (N - df + 0.5) / (df + 0.5))``, which is never negative.
    """
    if len(chunks) <= k:
        return list(range(len(chunks)))
    q = set(_bm25_tokens(query))
    block_sets = [set(_bm25_tokens(t)) for t in blocks]
    n = len(block_sets)
    idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t in q for df in [sum(t in bs for bs in block_sets)]}
    counts = [Counter(_bm25_tokens(c)) for c in chunks]
    lengths = [sum(c.values()) for c in counts]
    avg = max(1e-9, sum(lengths) / len(lengths))
    scores = [sum(idf[t] * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * ln / avg)) for t in q) for c, ln in zip(counts, lengths)]
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return sorted(int(i) for i in top)


def build_premises(context: str, query: str, spec: ChunkSpec, span_fn: SpanFn) -> list[str]:
    """Premise chunk(s) for one example under ``spec``. ``query`` is used only by ``retrieve``."""
    spans = span_fn(context)
    if spec.strategy == "truncate":
        if len(spans) <= spec.window:
            return [context]
        # Same text as the first `chunk` window, so API judges can serve truncate from the chunk cache.
        return [context[spans[0][0] : spans[spec.window - 1][1]]]
    chunks = split_windows(context, spans, spec.window, spec.stride)
    if spec.strategy == "retrieve":
        if spec.retriever == "bm25_blockidf":
            block = max(1, spec.stride // 3)
            idx = retrieve_top_k_blockidf(chunks, split_windows(context, spans, block, block), query, spec.top_k)
        else:
            idx = retrieve_top_k(chunks, query, spec.top_k)
        chunks = [chunks[i] for i in idx]
    return chunks


def aggregate(p_halluc: Sequence[float], how: str, k: int = 3) -> float:
    """Combine per-window hallucination probabilities into one per-example probability."""
    p = np.asarray(p_halluc, dtype=float)
    if p.size == 0:
        raise ValueError("no window scores to aggregate")
    if how == "min_phalluc":
        return float(p.min())
    if how == "mean_phalluc":
        return float(p.mean())
    if how == "topk_mean":
        return float(np.sort(p)[: max(1, k)].mean())
    raise ValueError(f"unknown aggregation {how!r}")


def hypothesis_text(question: str, response: str, task_type: str, fmt: str = "qa") -> str:
    """The claim to verify. Summaries and data-to-text responses are verified without the instruction."""
    if fmt == "answer" or task_type in ("Summary", "Data2txt"):
        return response
    if fmt == "qa":
        return f"{question.strip()} {response.strip()}"
    raise ValueError(f"unknown hypothesis format {fmt!r}")


TokenCounter = Callable[[str], int]
