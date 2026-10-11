"""fit_claim: only claims that cannot fit a pair window are cut, keeping their beginning."""

import re

from src.detectors.base import MIN_PREMISE_TOKENS, fit_claim


class _WordTokenizer:
    """One token per word; pairs add 3 special tokens ([CLS] a [SEP] b [SEP])."""

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3 if pair else 2

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, verbose=True):
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def test_claim_that_fits_is_unchanged():
    claim = " ".join(["w"] * 100)
    assert fit_claim(_WordTokenizer(), claim, 512) == (claim, False)


def test_overlong_claim_keeps_its_beginning_and_leaves_room_for_the_premise():
    words = [f"w{i}" for i in range(800)]
    cut, was_cut = fit_claim(_WordTokenizer(), " ".join(words), 512)
    assert was_cut
    budget = 512 - MIN_PREMISE_TOKENS - 3
    assert cut.split() == words[:budget]


def test_cuda_device_caps_the_allocator(monkeypatch):
    """get_device('cuda') must cap PyTorch's allocator so a Windows sysmem spill raises OOM instead."""
    import torch

    from src import config

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, *a: calls.append(f))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: type("P", (), {"total_memory": 8 * 2**30})())
    assert config.get_device("cuda") == "cuda"
    assert calls == [config.GPU_MEMORY_FRACTION]
