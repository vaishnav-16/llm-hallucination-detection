"""Open-weight LLM judge run locally (Qwen3 4B on the RTX 4070 in 4-bit; Qwen3 8B on Kaggle).

Same Yes/No prompt as the API judge. ``p_halluc`` = P(No) / (P(Yes) + P(No)) read directly from the
next-token distribution, so there is no sampling noise and no parsing. Pinned weights make it reproducible.
Qwen3 hybrid models are run with thinking disabled.

Memory (rough): Qwen3-4B in 4-bit is about 2.5 GB of weights plus about 0.15 MB of KV cache per token,
so about 4.4 GB at 30K tokens. Contexts beyond ``max_length`` must use ``chunk`` or ``retrieve``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.config import MODELS
from src.detectors.api_judge import JUDGE_SYSTEM, JUDGE_USER
from src.detectors.base import Detector, run_batched
from src.verify.chunking import SpanFn, hf_span_fn

logger = logging.getLogger(__name__)


class LocalLLMJudge(Detector):
    requires = "gpu"

    def __init__(self, model: str = "qwen3_4b", device: str = "cuda", load_in_4bit: bool = True,
                 max_length: int = 16384, batch_size: int = 1, name: str | None = None, device_map: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = MODELS.get(model, {"repo_id": model, "revision": None})
        self.name = name or spec["repo_id"].split("/")[-1]
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
        self.tokenizer.padding_side = "left"
        kw: dict = {"revision": spec["revision"]}
        # T4 GPUs (Kaggle) have no bf16; fall back to fp16 there.
        half = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=half)
            kw["device_map"] = device_map or {"": 0}
        elif device == "cuda":
            kw["dtype"] = half
            if device_map:
                kw["device_map"] = device_map  # e.g. "auto" to split an fp16 8B model over Kaggle's 2 x T4
        else:
            kw["dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(spec["repo_id"], **kw).eval()
        if "device_map" not in kw:
            self.model.to(device)
        self.yes_ids = self._token_ids(["Yes", " Yes", "yes"])
        self.no_ids = self._token_ids(["No", " No", "no"])

    def _token_ids(self, words: list[str]) -> list[int]:
        ids = set()
        for w in words:
            t = self.tokenizer.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        if not ids:
            raise ValueError(f"{self.name}: none of {words} is a single token")
        return sorted(ids)

    def span_fn(self) -> SpanFn:
        return hf_span_fn(self.tokenizer)

    def describe(self) -> dict:
        return {**super().describe(), "model": self.name}

    def _prompt(self, premise: str, hypothesis: str) -> str:
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(premise=premise, hypothesis=hypothesis)}]
        try:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _score_batch(self, pairs) -> np.ndarray:
        import torch

        texts = [self._prompt(p, h) for p, h in pairs]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[:, self.yes_ids].sum(dim=1)
        p_no = probs[:, self.no_ids].sum(dim=1)
        return (p_no / (p_yes + p_no).clamp_min(1e-12)).cpu().numpy()

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        pairs = list(zip(premises, hypotheses))
        return run_batched(pairs, self._score_batch, self.batch_size, [len(p) + len(h) for p, h in pairs], desc=self.name)
