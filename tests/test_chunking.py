"""Tests for the chunking step: containment guarantee, truncation, retrieval and aggregation direction."""

import random

import pytest

from src.verify.chunking import (
    ChunkSpec,
    aggregate,
    build_premises,
    guaranteed_span,
    hypothesis_text,
    retrieve_top_k,
    split_windows,
    whitespace_spans,
    window_starts,
)


def _doc(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize("window,overlap,n", [(512, 256, 5000), (8000, 1300, 60000), (1024, 900, 3000), (64, 10, 700)])
def test_any_span_up_to_guarantee_is_contained(window, overlap, n):
    starts = window_starts(n, window, window - overlap)
    assert starts[0] == 0 and starts[-1] + window >= n
    p = guaranteed_span(window, overlap)
    rng = random.Random(0)
    for _ in range(2000):
        a = rng.randrange(0, n - p + 1)
        assert any(s <= a and a + p <= s + window for s in starts), (a, p)


def test_span_longer_than_guarantee_can_be_missed():
    window, overlap, n = 100, 10, 1000
    starts = window_starts(n, window, window - overlap)
    p = guaranteed_span(window, overlap) + 20
    missed = [a for a in range(n - p + 1) if not any(s <= a and a + p <= s + window for s in starts)]
    assert missed, "a passage longer than overlap+1 should not always fit"


def test_split_windows_preserves_text_and_covers_document():
    text = _doc(1000)
    spans = whitespace_spans(text)
    chunks = split_windows(text, spans, 100, 60)
    assert chunks[0].startswith("w0 ") and chunks[-1].endswith("w999")
    assert all(c in text for c in chunks)
    assert all(len(c.split()) == 100 for c in chunks)


def test_short_context_is_one_chunk_for_every_strategy():
    text = _doc(50)
    for strat in ("truncate", "chunk", "retrieve"):
        assert build_premises(text, "q", ChunkSpec(strategy=strat, window=100), whitespace_spans) == [text]


def test_truncate_keeps_prefix_only():
    text = _doc(300)
    (prem,) = build_premises(text, "q", ChunkSpec(strategy="truncate", window=100), whitespace_spans)
    assert prem.split() == [f"w{i}" for i in range(100)]


def test_retrieve_finds_the_needle():
    filler = [" ".join(["revenue", "segment", "risk"] * 30) for _ in range(9)]
    needle = "the dividend payout ratio was 41 percent in fiscal 2023 " * 3
    chunks = filler[:5] + [needle] + filler[5:]
    assert 5 in retrieve_top_k(chunks, "What was the dividend payout ratio? 41 percent", k=2)


def test_aggregation_direction():
    # Faithful if ANY window supports the answer: one supportive window (p=0.05) must win.
    assert aggregate([0.9, 0.05, 0.8], "min_phalluc") == pytest.approx(0.05)
    assert aggregate([0.9, 0.05, 0.8], "mean_phalluc") == pytest.approx((0.9 + 0.05 + 0.8) / 3)
    assert aggregate([0.9, 0.05, 0.8, 0.1], "topk_mean", k=2) == pytest.approx(0.075)
    with pytest.raises(ValueError):
        aggregate([], "min_phalluc")


def test_spec_validation_and_tags():
    with pytest.raises(ValueError):
        ChunkSpec(strategy="bogus")
    with pytest.raises(ValueError):
        ChunkSpec(window=100, overlap=100)
    assert ChunkSpec(strategy="truncate", window=512).tag() == "truncate-w512"
    assert ChunkSpec(window=8000, overlap=2000).tag() == "chunk-w8000-o2000-min_phalluc"


def test_hypothesis_format():
    assert hypothesis_text("Who won?", "Alice.", "QA") == "Who won? Alice."
    assert hypothesis_text("Summarize:", "A summary.", "Summary") == "A summary."
    assert hypothesis_text("Who won?", "Alice.", "QA", fmt="answer") == "Alice."
