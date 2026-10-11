"""TypeSafe Jev (System One decision model) as a hallucination detector.

API facts this adapter relies on (docs.typesafe.ai, read 2026-10-11):
- ``POST https://api.typesafe.ai/v1/systemone`` with ``Authorization: Bearer $TYPESAFE_API_KEY`` and a body
  ``{"state": ..., "model": ..., "questions": {id: Question}}``. A ``noul`` answer is P(yes) in [0, 1]; a
  ``choice`` answer has a probability for every option.
- ``jev-1.13.0`` (pinned here; ``jev-latest`` is an alias that moves) accepts 64K tokens per request and 32K for
  the state plus the longest question. Input costs $0.042 per million tokens; output is free.
- Questions in one request share the state, are evaluated in parallel and cannot see each other. The state is
  billed once. So every claim that shares a document chunk goes into one request: the two answers of a PHANTOM or
  HaluEval pair, or the 6 model responses to one RAGTruth source.
- TypeSafe's own "jaggedness" notes for jev-1.13 list literal reading, numbers, and large states full of irrelevant
  detail as weak spots. PHANTOM's long-context grid measures exactly the last two.

Every request asks two questions per claim:
- ``noul``   "is every detail of `response` supported by `document`?" -> p_halluc = 1 - noul
- ``choice`` supported / contradicted / unsupported (TypeSafe's citation-check cookbook)
             -> p_halluc = 1 - P(supported)
The ``jev_noul`` and ``jev_choice`` runs send byte-identical requests, so the second run is served entirely from
the cache.

Responses are cached in ``artifacts/api_cache/jev.jsonl`` (key hash + answers + usage, never the prompt text), and
spend is tracked in ``artifacts/api_cache/jev_spend.json``. A ``max_usd`` guard stops a run before it overspends.
Failures raise: there are no default scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from src.config import ARTIFACTS_DIR, ROOT
from src.detectors.base import Detector, ScoringError
from src.verify.chunking import Span, SpanFn

logger = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-1.13.0"
USD_PER_INPUT_TOKEN = 0.042 / 1e6
CACHE_PATH = ARTIFACTS_DIR / "api_cache" / "jev.jsonl"
SPEND_PATH = ARTIFACTS_DIR / "api_cache" / "jev_spend.json"
PIECE_CHARS = 4  # chunking unit: ~4 characters ~ 0.9 tokens on PHANTOM (measured against Qwen3/ModernBERT tokenizers)
SEP = "\x1e"     # question / response separator inside packed hypotheses (same convention as LettuceDetector)

NOUL_INSTRUCTIONS = (
    "Is every factual detail in `response` (names, numbers, amounts, dates, causes and qualifiers) stated in or "
    "directly implied by `document`? `response` answers `question_asked`."
)
NOUL_CRITERIA = {
    "true": "Every detail of `response` is supported by `document`.",
    "false": "At least one detail of `response` is missing from `document` or conflicts with it.",
}
CHOICE_INSTRUCTIONS = "How does `document` relate to the facts stated in `response`? `response` answers `question_asked`."
CHOICE_CRITERIA = OrderedDict(
    supported="Every fact in `response` is stated in or directly implied by `document`.",
    contradicted="`document` states something that conflicts with at least one fact in `response`.",
    unsupported="At least one fact in `response` is not in `document`, although nothing in `document` contradicts it.",
)
SUMMARY_SUFFIX = " `response` is a summary or description written from `document`."


def piece_spans(text: str) -> list[Span]:
    """Chunking units for Jev: each word split into pieces of at most 4 characters (~0.9 real tokens each)."""
    out: list[Span] = []
    for m in re.finditer(r"\S+", text):
        s, e = m.span()
        out.extend((i, min(i + PIECE_CHARS, e)) for i in range(s, e, PIECE_CHARS))
    return out


def approx_tokens(text: str) -> int:
    return len(text) // PIECE_CHARS + 1


class JevClient:
    """HTTP client for /v1/systemone with retries, an append-only response cache and a spend ledger."""

    def __init__(self, model: str = DEFAULT_MODEL, max_usd: float | None = 4.5, timeout: float = 120.0):
        self.model, self.max_usd, self.timeout = model, max_usd, timeout
        self._key: str | None = None
        self._lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
        self.spend = json.loads(SPEND_PATH.read_text()) if SPEND_PATH.exists() else {"input_tokens": 0, "usd": 0.0, "requests": 0}
        self.session_tokens = 0
        self.served_models: set[str] = set()

    def _api_key(self) -> str:
        if self._key is None:
            try:
                from dotenv import load_dotenv

                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            self._key = os.environ.get("TYPESAFE_API_KEY", "")
            if not self._key:
                raise ScoringError("TYPESAFE_API_KEY is not set (put it in .env or the environment)")
        return self._key

    @staticmethod
    def request_key(body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def cached(self, body: dict) -> bool:
        return self.request_key(body) in self.cache

    def _post(self, path: str, body: dict | None) -> tuple[int, dict, dict]:
        """One HTTP call. Returns (status, json body, headers). Separate method so tests can replace it."""
        import requests

        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        if body is None:
            r = requests.get(f"{API_URL}{path}", headers=headers, timeout=self.timeout)
        else:
            r = requests.post(f"{API_URL}{path}", headers=headers, json=body, timeout=self.timeout)
        try:
            payload = r.json()
        except ValueError:
            payload = {"raw": r.text[:500]}
        return r.status_code, payload, dict(r.headers)

    def list_models(self) -> list[dict]:
        status, payload, _ = self._post("/models", None)
        if status != 200:
            raise ScoringError(f"Jev /models: HTTP {status}: {payload}")
        return payload.get("models", [])

    def systemone(self, body: dict, use_cache: bool = True) -> dict:
        key = self.request_key(body)
        if use_cache and key in self.cache:
            return self.cache[key]
        if self.max_usd is not None and self.spend["usd"] >= self.max_usd:
            raise ScoringError(f"Jev budget guard: ${self.spend['usd']:.3f} spent >= max_usd {self.max_usd}")
        for attempt in range(7):
            status, payload, headers = self._post("/systemone", body)
            if status == 200:
                break
            if status in (429, 529, 500, 502, 503, 504):
                wait = max(float(headers.get("retry-after", 0) or 0), min(60.0, 2.0 ** attempt))
                logger.warning("Jev HTTP %d, retrying in %.1fs", status, wait)
                time.sleep(wait)
                continue
            raise ScoringError(f"Jev HTTP {status}: {json.dumps(payload)[:400]}")
        else:
            raise ScoringError("Jev: still failing after retries (rate limit / overload)")
        slim = {"model": payload.get("model"), "answers": payload.get("answers", {}), "usage": payload.get("usage", {})}
        tokens = int(slim["usage"].get("input_tokens", 0))
        with self._lock:
            self.cache[key] = slim
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": slim}) + "\n")
            self.spend["input_tokens"] += tokens
            self.spend["usd"] = round(self.spend["input_tokens"] * USD_PER_INPUT_TOKEN, 6)
            self.spend["requests"] += 1
            SPEND_PATH.write_text(json.dumps(self.spend, indent=2))
            self.session_tokens += tokens
            if slim["model"]:
                self.served_models.add(slim["model"])
        return slim


class JevJudge(Detector):
    """Zero-shot Jev judge. ``answer`` selects which of the two questions becomes p_halluc."""

    requires = "api"

    def __init__(self, answer: str = "noul", model: str = DEFAULT_MODEL, name: str | None = None, max_usd: float | None = 4.5,
                 concurrency: int = 6, max_claims_per_request: int = 12, max_length: int = 28000, device: str | None = None):
        if answer not in ("noul", "choice"):
            raise ValueError("answer must be 'noul' or 'choice'")
        self.answer, self.name = answer, name or f"jev_{answer}"
        self.client = JevClient(model, max_usd)
        self.concurrency, self.max_claims = concurrency, max_claims_per_request
        self.max_length = max_length  # in ~4-char pieces; keep windows well under it
        self.n_shrunk = 0

    def span_fn(self) -> SpanFn:
        return piece_spans

    @staticmethod
    def pack_question(question: str, response: str, task_type: str) -> str:
        return f"{'' if task_type in ('Summary', 'Data2txt') else question}{SEP}{response}"

    def describe(self) -> dict:
        return {
            **super().describe(), "model_requested": self.client.model, "models_served": sorted(self.client.served_models),
            "answer": self.answer, "session_input_tokens": self.client.session_tokens,
            "session_usd": round(self.client.session_tokens * USD_PER_INPUT_TOKEN, 4), "premises_shrunk": self.n_shrunk,
        }

    # --- request construction -------------------------------------------------
    def _questions(self, claims: Sequence[str]) -> dict:
        qs: dict = {}
        for i, packed in enumerate(claims):
            question, _, response = packed.partition(SEP)
            data = {"question_asked": question or "(none: summary or data-to-text task)", "response": response}
            suffix = "" if question else SUMMARY_SUFFIX
            qs[f"c{i}_noul"] = {"type": "noul", "instructions": {**data, "question": NOUL_INSTRUCTIONS + suffix}, "criteria": NOUL_CRITERIA}
            qs[f"c{i}_choice"] = {"type": "choice", "instructions": {**data, "question": CHOICE_INSTRUCTIONS + suffix}, "criteria": dict(CHOICE_CRITERIA)}
        return qs

    def build_body(self, premise: str, claims: Sequence[str]) -> dict:
        return {"model": self.client.model, "state": {"document": premise}, "questions": self._questions(claims)}

    def _groups(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[tuple[str, list[int]]]:
        """Group pair indices by premise (deduplicated claims), in chunks of at most max_claims."""
        by_premise: OrderedDict[str, OrderedDict[str, list[int]]] = OrderedDict()
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            by_premise.setdefault(p, OrderedDict()).setdefault(h, []).append(i)
        groups = []
        for p, claims in by_premise.items():
            items = list(claims.items())
            for k in range(0, len(items), self.max_claims):
                groups.append((p, items[k : k + self.max_claims]))
        return groups

    # --- scoring --------------------------------------------------------------
    def _answer(self, resp: dict, i: int) -> float:
        a = resp["answers"].get(f"c{i}_{self.answer}")
        if a is None:
            raise ScoringError(f"Jev response missing answer c{i}_{self.answer}: {list(resp['answers'])[:6]}")
        if self.answer == "noul":
            return 1.0 - float(a["noul"])
        return 1.0 - float(a["probabilities"]["supported"])

    def _run_group(self, premise: str, claims: list[tuple[str, list[int]]]) -> list[tuple[list[int], float]]:
        texts = [c for c, _ in claims]
        for attempt in range(3):
            body = self.build_body(premise, texts)
            try:
                resp = self.client.systemone(body)
                break
            except ScoringError as e:
                # The live API answers 400 {"detail": {"error_type": "max_tokens_exceeded"}} (seen 2026-10-11 on
                # dense 10-K tables, ~2.3 tokens per piece); 422 is kept for other "too long" phrasings.
                too_long = re.search(r"HTTP (400|422).*(max_tokens|token|context|length|too long|exceed)", str(e), re.I | re.S)
                if not too_long or attempt == 2:
                    raise
                self.n_shrunk += 1
                premise = premise[: int(len(premise) * 0.75)]
                logger.warning("Jev: request too long, retrying with the document cut to %d chars", len(premise))
        return [(idx, self._answer(resp, i)) for i, (_, idx) in enumerate(claims)]

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = np.full(len(premises), np.nan)
        groups = self._groups(premises, hypotheses)
        n_live = sum(not self.client.cached(self.build_body(p, [c for c, _ in g])) for p, g in groups)
        logger.info("%s: %d requests (%d not cached) for %d pairs; spent so far $%.3f", self.name, len(groups), n_live, len(premises), self.client.spend["usd"])
        with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as pool:
            for k, res in enumerate(pool.map(lambda g: self._run_group(*g), groups), 1):
                for idx, p in res:
                    out[idx] = p
                if k % 100 == 0:
                    logger.info("%s: %d/%d requests, spent $%.3f", self.name, k, len(groups), self.client.spend["usd"])
        if np.isnan(out).any():
            raise ScoringError(f"{self.name}: {int(np.isnan(out).sum())} pairs without a score")
        return out

    def estimate(self, premises: Sequence[str], hypotheses: Sequence[str]) -> dict:
        """Requests, approximate input tokens and USD for uncached requests, without calling the API."""
        groups = self._groups(premises, hypotheses)
        live = [(p, g) for p, g in groups if not self.client.cached(self.build_body(p, [c for c, _ in g]))]
        tokens = sum(approx_tokens(json.dumps(self.build_body(p, [c for c, _ in g]))) for p, g in live)
        return {"requests": len(groups), "uncached_requests": len(live), "approx_tokens": tokens, "approx_usd": tokens * USD_PER_INPUT_TOKEN}


def probe(sample_cell: str = "phantom_10k_500_seed", long_cell: str = "phantom_10k_30000_middle") -> dict:
    """Small live check (well under $0.01): models, a pair grouped vs alone, and one ~20K-piece document."""
    from src.data.store import write_json

    judge = JevJudge("noul", max_usd=None)
    res: dict = {"model_requested": judge.client.model}
    res["models"] = judge.client.list_models()
    try:
        from src.data.prepare import load_normalized

        seed = load_normalized(sample_cell).sort_values(["pair_id", "label"]).head(2)
        long_doc = load_normalized(long_cell)["context"].iloc[0]
    except FileNotFoundError:
        logger.warning("normalized data missing; probing with synthetic text (run `prepare` for a real probe)")
        import pandas as pd

        seed = pd.DataFrame({"question": ["What was the payout ratio?"] * 2, "response": ["41%.", "73%, driven by Asia."],
                             "context": ["The dividend payout ratio was 41% in fiscal 2023."] * 2, "label": [0, 1], "task_type": ["QA"] * 2})
        long_doc = "The company reported revenue growth in fiscal year 2023. " * 1500
    claims = [JevJudge.pack_question(q, r, t) for q, r, t in zip(seed["question"], seed["response"], seed["task_type"])]
    doc = seed["context"].iloc[0]
    grouped = judge.client.systemone(judge.build_body(doc, claims), use_cache=False)
    alone = [judge.client.systemone(judge.build_body(doc, [c]), use_cache=False) for c in claims]
    res["grouped_vs_alone_noul"] = [
        {"label": int(lab), "grouped": grouped["answers"][f"c{i}_noul"]["noul"], "alone": a["answers"]["c0_noul"]["noul"]}
        for i, (lab, a) in enumerate(zip(seed["label"], alone))
    ]
    res["grouped_input_tokens"] = grouped["usage"].get("input_tokens")
    res["alone_input_tokens"] = [a["usage"].get("input_tokens") for a in alone]
    spans = piece_spans(long_doc)
    window = long_doc[: spans[min(len(spans), 20000) - 1][1]]
    t0 = time.time()
    try:
        r = judge.client.systemone(judge.build_body(window, claims[:1]), use_cache=False)
        res["long"] = {"pieces": min(len(spans), 20000), "input_tokens": r["usage"].get("input_tokens"),
                       "tokens_per_piece": round(r["usage"].get("input_tokens", 0) / min(len(spans), 20000), 3), "latency_s": round(time.time() - t0, 2)}
    except ScoringError as e:
        res["long"] = {"pieces": min(len(spans), 20000), "error": str(e)[:300]}
    res["models_served"] = sorted(judge.client.served_models)
    res["spend_after_probe"] = judge.client.spend
    write_json(res, ARTIFACTS_DIR / "jev_probe.json")
    logger.info("Jev probe: %s", json.dumps({k: v for k, v in res.items() if k != "models"}, default=str)[:800])
    return res
