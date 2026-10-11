"""Hugging Face model detectors: HHEM-2.1-open, NLI / binary sequence classifiers, and LettuceDetect.

All load at pinned revisions with strict weight loading; a failed load raises instead of falling
back to another model under the same name.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.config import MODELS, ROOT
from src.detectors.base import Detector, fit_claim, fit_premise, run_batched
from src.verify.chunking import SpanFn, hf_span_fn

logger = logging.getLogger(__name__)


def _torch():
    import torch

    return torch


def _dtype(device: str, precision: str):
    torch = _torch()
    if precision == "auto":
        return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


class _HFDetector(Detector):
    tokenizer = None

    def span_fn(self) -> SpanFn:
        return hf_span_fn(self.tokenizer)

    def _len(self, pair: tuple[str, str]) -> int:
        return len(pair[0]) + len(pair[1])

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        pairs = list(zip(premises, hypotheses))
        return run_batched(pairs, self._score_batch, self.batch_size, [self._len(p) for p in pairs], desc=self.name)

    def _score_batch(self, pairs):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


T5_ATTENTION = "t5_bias_sdpa"
_t5_bias_cache: dict = {}


def t5_bias_sdpa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, position_bias=None, **kwargs):
    """T5 attention with the relative position bias and padding mask folded into one additive mask for SDPA.

    Stock T5 + ``sdpa`` rebuilds a (batch, heads, L, L) mask in every layer and falls back to SDPA's fp32 math
    kernel (4.4 pairs/s for HHEM at 2K tokens on the 4070); ``eager`` is 6.7. T5 passes the same position-bias
    tensor to all 12 layers, so the combined mask is built once per forward in the model dtype and SDPA's
    fused kernel runs: 24 pairs/s, scores within 0.003 of ``eager`` (bf16 noise; batch composition alone moves
    them 0.01).
    """
    torch = _torch()
    refs = _t5_bias_cache.get("refs")
    if refs is not None and refs[0] is position_bias and refs[1] is attention_mask:
        bias = _t5_bias_cache["bias"]
    else:
        bias = position_bias.to(query.dtype)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                bias = bias.masked_fill(~attention_mask, torch.finfo(query.dtype).min)
            else:
                bias = bias + attention_mask.to(query.dtype)
        # Holding the inputs keeps their ids from being reused by the next batch's tensors.
        _t5_bias_cache.update(refs=(position_bias, attention_mask), bias=bias)
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def _register_t5_attention() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(T5_ATTENTION, t5_bias_sdpa)
    AttentionMaskInterface.register(T5_ATTENTION, sdpa_mask)


class HHEMDetector(_HFDetector):
    """Vectara HHEM-2.1-open (flan-t5-base). Its consistency score s becomes p_halluc = 1 - s."""

    requires = "gpu"

    def __init__(self, device: str = "cpu", batch_size: int = 8, max_length: int = 4096, precision: str = "fp32", name: str = "hhem"):
        from transformers import AutoModelForSequenceClassification

        spec = MODELS["hhem"]
        self.name, self.device, self.batch_size, self.max_length = name, device, batch_size, max_length
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec["repo_id"], revision=spec["revision"], trust_remote_code=True, dtype=_dtype(device, precision)
        )
        self.model.t5.to(device).eval()
        _register_t5_attention()
        self.model.t5.config._attn_implementation = T5_ATTENTION
        self.tokenizer = self.model.tokenzier  # (sic) attribute name in HHEM's remote code
        self.prompt = self.model.prompt
        self.n_truncated = 0

    def _format(self, premise: str, hypothesis: str) -> str:
        """HHEM prompt with the premise cut to fit, so truncation never removes the hypothesis."""
        overhead = len(self.tokenizer(self.prompt.format(text1="", text2=hypothesis))["input_ids"])
        premise, cut = fit_premise(self.tokenizer, premise, self.max_length - overhead - 8)  # margin: boundary tokens can merge differently
        if cut:
            self.n_truncated += 1
        return self.prompt.format(text1=premise, text2=hypothesis)

    def _score_batch(self, pairs):
        torch = _torch()
        texts = [self._format(p, h) for p, h in pairs]
        # Multiple-of-8 lengths keep SDPA's fused kernel aligned (4.9 -> 7.9 pairs/s before the bias cache).
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, pad_to_multiple_of=8).to(self.device)
        with torch.no_grad():
            logits = self.model.t5(**enc).logits[:, 0, :]
        consistency = torch.softmax(logits.float(), dim=-1)[:, 1]
        return (1.0 - consistency).cpu().numpy()


    def describe(self) -> dict:
        return {**super().describe(), "premises_truncated": self.n_truncated, "attention": T5_ATTENTION}


class SequenceClassifierDetector(_HFDetector):
    """Any AutoModelForSequenceClassification checkpoint.

    ``output='nli'``: p_halluc = 1 - P(entailment), with the entailment index read from ``id2label``.
    ``output='binary'``: p_halluc = P(label 1), for checkpoints fine-tuned in this project (1 = hallucinated).
    """

    requires = "gpu"

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        output: str = "auto",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        precision: str = "auto",
        name: str | None = None,
        extend_positions: int | None = None,
    ):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        if model in MODELS:  # short name from config.MODELS
            revision = revision or MODELS[model]["revision"]
            model = MODELS[model]["repo_id"]
        elif (ROOT / model).exists():  # a local fine-tuned checkpoint, e.g. models/modernbert-nli-ft/seed42
            model = str(ROOT / model)
        self.name = name or model.replace("\\", "/").rstrip("/").split("/")[-1]
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.claims_truncated: set[str] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        cfg = AutoConfig.from_pretrained(model, revision=revision)
        if extend_positions:
            # Only for rotary-position models (ModernBERT): there is no learned position table to outgrow.
            if cfg.model_type != "modernbert":
                raise ValueError(f"{self.name}: extend_positions is only supported for ModernBERT, not {cfg.model_type}")
            logger.warning("%s: extending max_position_embeddings %d -> %d (RoPE)", self.name, cfg.max_position_embeddings, extend_positions)
            cfg.max_position_embeddings = extend_positions
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, config=cfg, dtype=_dtype(device, precision)
        ).to(device).eval()
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        if output == "auto":
            output = "nli" if any("entail" in v for v in labels.values()) else "binary"
        self.output = output
        if output == "nli":
            ent = [i for i, v in labels.items() if "entail" in v and "not" not in v]
            if len(ent) != 1:
                raise ValueError(f"{self.name}: cannot find a unique entailment label in {labels}")
            self.entail_idx = ent[0]
        elif output == "binary":
            if self.model.config.num_labels != 2:
                raise ValueError(f"{self.name}: binary output needs 2 labels, got {labels}")
        else:
            raise ValueError(f"unknown output mode {output!r}")
        model_max = getattr(self.model.config, "max_position_embeddings", max_length)
        if max_length > model_max:
            raise ValueError(f"{self.name}: max_length {max_length} exceeds the model's {model_max} positions")
        logger.info("%s loaded (%s output, labels %s, max_length %d)", self.name, output, labels, max_length)

    def _score_batch(self, pairs):
        torch = _torch()
        hyps = []
        for _, claim in pairs:
            h, cut = fit_claim(self.tokenizer, claim, self.max_length)
            if cut and claim not in self.claims_truncated:  # a claim recurs in every window of its context
                self.claims_truncated.add(claim)
                logger.warning("%s: claim longer than the %d-token window, cut to fit (%d distinct so far)",
                               self.name, self.max_length, len(self.claims_truncated))
            hyps.append(h)
        enc = self.tokenizer(
            [p for p, _ in pairs], hyps, return_tensors="pt", padding=True,
            truncation="only_first", max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
        if self.output == "nli":
            return (1.0 - probs[:, self.entail_idx]).cpu().numpy()
        return probs[:, 1].cpu().numpy()

    def describe(self) -> dict:
        return {**super().describe(), "claims_truncated": len(self.claims_truncated)}


LETTUCE_QA_PROMPT = (
    "Briefly answer the following question:\n{question}\nBear in mind that your response should be strictly "
    "based on the following 1 passages:\n{context}\nIn case the passages do not contain the necessary information "
    'to answer the question, please reply with: "Unable to answer based on given passages."\noutput:'
)
LETTUCE_SUMMARY_PROMPT = "Summarize the following text:\n{context}\noutput:"


class LettuceDetector(_HFDetector):
    """LettuceDetect (ModernBERT token classifier). Response score = max hallucination prob over answer tokens.

    Uses the library's own English prompt templates (lettucedetect 0.2.3). The runner passes
    hypotheses as ``question \\x1e answer`` (see ``pack_question``) so the question can go in the prompt.
    """

    requires = "gpu"

    def __init__(self, device: str = "cpu", batch_size: int = 4, max_length: int = 4096, precision: str = "auto", name: str = "lettucedetect"):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        spec = MODELS["lettucedetect"]
        self.name, self.device, self.batch_size, self.max_length = name, device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
        self.model = AutoModelForTokenClassification.from_pretrained(
            spec["repo_id"], revision=spec["revision"], dtype=_dtype(device, precision)
        ).to(device).eval()

    @staticmethod
    def pack_question(question: str, answer: str, task_type: str) -> str:
        return f"{'' if task_type in ('Summary', 'Data2txt') else question}\x1e{answer}"

    def _score_batch(self, pairs):
        torch = _torch()
        prompts, answers = [], []
        for premise, packed in pairs:
            question, _, answer = packed.partition("\x1e")
            tmpl = LETTUCE_QA_PROMPT if question else LETTUCE_SUMMARY_PROMPT
            prompts.append(tmpl.format(question=question, context=f"passage 1: {premise}"))
            answers.append(answer)
        enc = self.tokenizer(prompts, answers, return_tensors="pt", padding=True, truncation="only_first", max_length=self.max_length)
        answer_mask = torch.tensor(
            [[sid == 1 for sid in enc.sequence_ids(i)] for i in range(len(pairs))], dtype=torch.bool
        )
        enc = enc.to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, :, 1].cpu()
        probs = probs.masked_fill(~answer_mask, 0.0)
        return probs.max(dim=1).values.numpy()
