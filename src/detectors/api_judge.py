"""LLM-as-judge over OpenAI-compatible HTTP APIs (Dartmouth Chat, Gemini), with a response cache.

The judge sees one (document chunk, claim) pair and answers Yes/No to "is the claim supported?".
How ``p_halluc`` is obtained depends on the provider's ``probability`` setting:
- ``logprobs``   P(No) / (P(Yes) + P(No)) from the first token's top log-probabilities.
- ``sampling``   share of "No" among ``samples`` completions at temperature 1.
- ``verbalized`` the model states a 0-100 confidence that the claim is supported.

Every response is cached in ``artifacts/api_cache/<provider>.jsonl`` keyed by a SHA-256 of
(provider, model, prompt, parameters). Only the key hash and the response are stored, never the prompt
(contexts can be 30K tokens). Reruns and resumed runs cost nothing. A request that still fails after
retries raises: there are no default scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from src.config import ARTIFACTS_DIR, CONFIGS_DIR
from src.detectors.base import Detector

logger = logging.getLogger(__name__)

CACHE_DIR = ARTIFACTS_DIR / "api_cache"

JUDGE_SYSTEM = "You are a meticulous fact-checker. You judge only whether a claim is supported by the given document."
JUDGE_USER = (
    "Document:\n\"\"\"\n{premise}\n\"\"\"\n\nClaim:\n\"\"\"\n{hypothesis}\n\"\"\"\n\n"
    "Is every part of the claim supported by the document? A claim with any unsupported or contradicted detail is NOT supported. "
    "Answer with exactly one word: Yes or No."
)
VERBAL_USER = (
    "Document:\n\"\"\"\n{premise}\n\"\"\"\n\nClaim:\n\"\"\"\n{hypothesis}\n\"\"\"\n\n"
    "How confident are you, from 0 to 100, that every part of the claim is supported by the document? Reply with only the number."
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(CONFIGS_DIR.parent / ".env")
    except ImportError:
        pass


def load_providers() -> dict:
    with open(CONFIGS_DIR / "providers.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ApiClient:
    """Minimal OpenAI-compatible chat client with rate limiting, retries and an append-only cache."""

    def __init__(self, provider: str, model: str | None = None, overrides: dict | None = None):
        _load_env()
        cfg = dict(load_providers()[provider])
        cfg.update(overrides or {})
        self.provider, self.cfg = provider, cfg
        self.model = model or cfg.get("model")
        key = os.environ.get(cfg["api_key_env"], "")
        if not key:
            raise RuntimeError(f"{provider}: set {cfg['api_key_env']} in .env or the environment")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.base = cfg["base_url"].rstrip("/")
        self.min_interval = 60.0 / float(cfg.get("requests_per_minute", 30))
        self._last = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"{provider}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]

    def list_models(self) -> list[str]:
        import requests

        r = requests.get(f"{self.base}/models", headers=self.headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        items = data.get("data", data if isinstance(data, list) else [])
        return sorted(str(m.get("id", m)) if isinstance(m, dict) else str(m) for m in items)

    def chat(self, messages: list[dict], use_cache: bool = True, **params) -> dict:
        import requests

        if not self.model:
            raise RuntimeError(f"{self.provider}: no model pinned in configs/providers.yaml (run probe-api)")
        body = {"model": self.model, "messages": messages, **params}
        key = hashlib.sha256(json.dumps([self.provider, body], sort_keys=True).encode()).hexdigest()
        if use_cache and key in self.cache:
            return self.cache[key]
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = requests.post(f"{self.base}/chat/completions", headers=self.headers, json=body, timeout=300)
            except requests.RequestException as e:
                logger.warning("%s: request error %s (attempt %d)", self.provider, e, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                resp = r.json()
                slim = {"choices": resp.get("choices", []), "model": resp.get("model"), "usage": resp.get("usage")}
                self.cache[key] = slim
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": slim}) + "\n")
                return slim
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = float(r.headers.get("retry-after", 0) or 0)
                time.sleep(max(retry_after, 2 ** attempt * 2))
                continue
            raise RuntimeError(f"{self.provider}: HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"{self.provider}: request failed after retries")


def _first_text(resp: dict, i: int = 0) -> str:
    return (resp["choices"][i]["message"].get("content") or "").strip()


def _yes_no(text: str) -> str | None:
    m = re.match(r"\W*(yes|no)\b", text.lower())
    return m.group(1) if m else None


def p_no_from_logprobs(resp: dict) -> float | None:
    """P(No) / (P(Yes) + P(No)) from the first generated token's top logprobs, if present."""
    try:
        first = resp["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return None
    mass = {"yes": 0.0, "no": 0.0}
    for cand in first.get("top_logprobs", []) or [first]:
        tok = cand.get("token", "").strip().lower()
        if tok in mass:
            mass[tok] += math.exp(cand["logprob"])
    total = mass["yes"] + mass["no"]
    return mass["no"] / total if total > 0 else None


class ApiJudge(Detector):
    """Zero-shot LLM judge over an API provider. Chunking uses whitespace words as the token unit."""

    requires = "api"

    def __init__(self, provider: str, model: str | None = None, name: str | None = None, max_length: int = 6000, samples: int | None = None):
        self.client = ApiClient(provider, model)
        self.name = name or f"{provider}:{self.client.model}"
        self.max_length = max_length  # in whitespace words; set from the probe's context limit
        self.mode = self.client.cfg.get("probability", "logprobs")
        self.samples = samples or int(self.client.cfg.get("samples", 5))

    def describe(self) -> dict:
        return {**super().describe(), "provider": self.client.provider, "model": self.client.model, "probability": self.mode}

    def _one(self, premise: str, hypothesis: str) -> float:
        if self.mode == "verbalized":
            resp = self.client.chat(
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": VERBAL_USER.format(premise=premise, hypothesis=hypothesis)}],
                temperature=0, max_tokens=8,
            )
            m = re.search(r"\d+(?:\.\d+)?", _first_text(resp))
            if not m:
                raise RuntimeError(f"{self.name}: unparseable confidence {_first_text(resp)!r}")
            return 1.0 - min(100.0, float(m.group())) / 100.0
        msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": JUDGE_USER.format(premise=premise, hypothesis=hypothesis)}]
        if self.mode == "logprobs":
            resp = self.client.chat(msgs, temperature=0, max_tokens=1, logprobs=True, top_logprobs=10)
            p = p_no_from_logprobs(resp)
            if p is not None:
                return p
            ans = _yes_no(_first_text(resp))
            if ans is None:
                raise RuntimeError(f"{self.name}: no logprobs and unparseable answer {_first_text(resp)!r}")
            return 1.0 if ans == "no" else 0.0
        votes = []
        for s in range(self.samples):  # sampling: separate cached calls (seeded by index)
            resp = self.client.chat(msgs, temperature=1.0, max_tokens=3, seed=s)
            ans = _yes_no(_first_text(resp))
            if ans is not None:
                votes.append(ans == "no")
        if not votes:
            raise RuntimeError(f"{self.name}: no parseable Yes/No in {self.samples} samples")
        return float(np.mean(votes))

    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = np.empty(len(premises))
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            out[i] = self._one(p, h)
            if (i + 1) % 50 == 0:
                logger.info("%s: %d/%d pairs", self.name, i + 1, len(premises))
        return out


def probe_providers(providers: list[str] | None = None) -> dict:
    """List models, then check logprobs support, a long-context request and latency for each provider."""
    from src.data.store import write_json

    results = {}
    # claude_cli has no HTTP endpoint to probe (rewrite only), so only base_url providers are listed by default.
    for name in providers or [k for k, v in load_providers().items() if v.get("base_url")]:
        res: dict = {"provider": name}
        try:
            client = ApiClient(name)
            res["models"] = client.list_models()
            if not client.model:
                res["note"] = "pin `model` in configs/providers.yaml, then rerun probe-api"
                results[name] = res
                continue
            t0 = time.time()
            r = client.chat([{"role": "user", "content": "Answer Yes or No: is Paris in France?"}], use_cache=False,
                            temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)
            res["latency_s"] = round(time.time() - t0, 2)
            res["answer"] = _first_text(r)
            res["logprobs"] = p_no_from_logprobs(r) is not None
            for words in (2000, 8000, 24000):
                doc = " ".join(["The company reported revenue growth in fiscal year 2023."] * (words // 9))
                try:
                    client.chat([{"role": "user", "content": doc + "\n\nDid the company report revenue growth? Yes or No."}],
                                use_cache=False, temperature=0, max_tokens=1)
                    res[f"ok_{words}_words"] = True
                except Exception as e:
                    res[f"ok_{words}_words"] = f"failed: {str(e)[:120]}"
                    break
        except Exception as e:
            res["error"] = str(e)[:300]
        results[name] = res
        logger.info("probe %s: %s", name, {k: v for k, v in res.items() if k != "models"})
    write_json(results, ARTIFACTS_DIR / "api_probe.json")
    return results
