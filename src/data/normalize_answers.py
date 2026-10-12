"""Label-blind rewrite of HaluEval answers into one declarative style (the ``normalized`` variant).

Safeguards (see the plan's "Safeguards for the HaluEval answer rewrite"):
- The rewriter never learns which answer is correct: every answer gets the identical prompt as a
  separate call, and calls are issued in a seeded random order that interleaves both labels.
- Meaning check: the rewrite must keep the original answer's content words (all of them for short
  answers of 4 words or fewer, at least 80% otherwise). A failed check is retried once with a stricter
  prompt; if it fails again the whole pair is dropped and counted.
- The rewriter model must not also be used as a judge on HaluEval (``evaluate`` warns if it is).

By default only the val/test pairs plus 2,000 train pairs are rewritten (about 10K calls).

Providers: an OpenAI-compatible API from ``configs/providers.yaml`` (``dartmouth``, ``gemini``), or
``claude_cli`` (headless ``claude -p`` on a Claude subscription, one process per answer; see
``src/data/claude_cli.py``). ``claude -p`` has no temperature control, so those rewrites are reproducible
only from the committed jsonl. The jsonl holds one provider/model; adding records from another raises.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, SEED
from src.data.store import write_table

logger = logging.getLogger(__name__)

REWRITES_PATH = ARTIFACTS_DIR / "rewrites" / "halueval_rewrites.jsonl"
AUDIT_PATH = ARTIFACTS_DIR / "rewrite_audit.md"

PROMPT = (
    "Question: {question}\nAnswer: {answer}\n\n"
    "Rewrite the answer as one complete, natural declarative sentence that answers the question. "
    "Keep exactly the same facts: do not add, remove, correct or hedge any information, even if the answer seems wrong. "
    "Output only the sentence."
)
STRICT_SUFFIX = " Reuse the answer's key words verbatim."

_STOP = set("the a an of to in and or for on at by with is are was were be been it its this that as from which who whom what when where how did does do".split())
_TOK = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> list[str]:
    return [w for w in _TOK.findall(text.lower()) if w not in _STOP]


def meaning_preserved(original: str, rewrite: str) -> bool:
    words = content_words(original)
    if not words:
        return True
    have = set(_TOK.findall(rewrite.lower()))
    kept = sum(w in have for w in words) / len(words)
    return kept == 1.0 if len(words) <= 4 else kept >= 0.8


def _select_pairs(df: pd.DataFrame, train_pairs: int) -> set[str]:
    from src.data import splits

    doc = splits.load_splits()
    s = df.assign(split=splits.assign(df, doc)).drop_duplicates("pair_id")
    keep = set(s.loc[s["split"].isin(["val", "test"]), "pair_id"])
    train = sorted(s.loc[s["split"] == "train", "pair_id"])
    rng = np.random.default_rng(SEED)
    keep |= set(rng.choice(train, size=min(train_pairs, len(train)), replace=False).tolist())
    return keep


def _rewrite_answer(response: str, question: str, complete: Callable[[str], str]) -> tuple[str, bool]:
    """Identical prompt for every answer; one stricter retry if the meaning check fails."""
    prompt = PROMPT.format(question=question, answer=response)
    text, ok = "", False
    for attempt in range(2):
        text = (complete(prompt + (STRICT_SUFFIX if attempt else "")) or "").strip().strip('"')
        ok = bool(text) and meaning_preserved(response, text)
        if ok:
            break
    return text, ok


def load_records(path: Path | None = None) -> dict:
    path = path or REWRITES_PATH
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["example_id"]] = rec
    return done


def rewrite_halueval(provider: str = "dartmouth", limit: int | None = None, train_pairs: int = 2000, workers: int = 4,
                     out_path: Path | None = None, model: str | None = None, max_new: int | None = None) -> None:
    """Rewrite selected HaluEval answers through ``provider``; resumable from the jsonl.

    ``provider`` is an OpenAI-compatible API from configs/providers.yaml (one call at a time), or
    ``claude_cli``: one headless ``claude -p`` process per answer, ``workers`` of them in parallel.
    ``out_path`` other than the default writes a scratch file (its audit sits next to it, and the
    normalized table is not rebuilt). ``model`` overrides the pinned model. ``max_new`` stops after that
    many new answers so a long run can be committed in chunks.
    """
    from src.data.halueval import load_halueval

    out_path = Path(out_path) if out_path else REWRITES_PATH
    scratch = out_path.resolve() != REWRITES_PATH.resolve()
    df = load_halueval("original")
    pairs = sorted(_select_pairs(df, train_pairs))
    if limit:
        pairs = pairs[:limit]
    rows = df[df["pair_id"].isin(set(pairs))].sample(frac=1.0, random_state=SEED)  # interleave labels
    done = load_records(out_path)
    todo = [r for r in rows.itertuples(index=False) if r.example_id not in done]
    if max_new is not None:
        todo = todo[:max_new]

    if provider == "claude_cli":
        from src.data.claude_cli import ClaudeCliClient
        from src.detectors.api_judge import load_providers

        cfg = load_providers().get("claude_cli") or {}
        client = ClaudeCliClient(model or cfg.get("model"), timeout_s=cfg.get("timeout_s", 180),
                                 max_retries=cfg.get("max_retries", 5), backoff_s=cfg.get("backoff_s", 15))
        meta = {"provider": provider, "model": client.model, "cli_version": client.version}
    else:
        from src.detectors.api_judge import ApiClient

        api = ApiClient(provider, model=model)
        workers = 1
        meta = {"provider": provider, "model": api.model}

        def api_complete(prompt: str) -> tuple[str, str]:
            resp = api.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=120)
            return resp["choices"][0]["message"].get("content") or "", api.model

        client = SimpleNamespace(model=api.model, complete=api_complete)
    _check_single_model(done, meta, out_path)
    logger.info("Rewriting %d answers with %s/%s (%d already on file, %d worker(s))",
                len(todo), meta["provider"], meta["model"], len(done), workers)

    def one(r) -> dict:
        served = set()

        def complete(prompt: str) -> str:
            text, m = client.complete(prompt)
            served.add(m)
            return text

        text, ok = _rewrite_answer(r.response, r.question, complete)
        if served != {meta["model"]}:
            raise RuntimeError(f"rewrite of {r.example_id} served by {sorted(served)}, expected {meta['model']}")
        return {"example_id": r.example_id, "pair_id": r.pair_id, "rewrite": text, "ok": ok, **meta}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(one, r) for r in todo]  # FIFO: calls start in the seeded order
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[rec["example_id"]] = rec
                n_new += 1
                if n_new % 200 == 0:
                    logger.info("rewrote %d/%d answers (%d on file)", n_new, len(todo), len(done))
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)
            logger.error("Rewrite stopped after %d new answers; rerun the same command to resume", n_new)
            raise
    sub = {k: v for k, v in done.items() if k in set(rows["example_id"])}
    logger.info("Rewrite finished: %d answers on file, %d ok", len(done), sum(v["ok"] for v in done.values()))
    if scratch:
        write_audit(df, sub, path=out_path.with_suffix(".audit.md"))
        return
    write_audit(df, done)
    from src.data.prepare import NORM_DIR

    write_table(load_halueval("normalized"), NORM_DIR / "halueval_normalized.parquet")


def _check_single_model(done: dict, meta: dict, path: Path) -> None:
    """The rewrite file must hold one provider/model only; refuse to mix in another."""
    seen = {(v.get("provider"), v.get("model")) for v in done.values()}
    if seen - {(meta["provider"], meta["model"])}:
        raise RuntimeError(f"{path} already holds rewrites from {sorted(seen)}; refusing to add {meta['provider']}/{meta['model']}")


def apply_rewrites(df: pd.DataFrame) -> pd.DataFrame:
    """Replace responses with their rewrites; keep only pairs where both answers passed the check."""
    if not REWRITES_PATH.exists():
        raise FileNotFoundError(f"{REWRITES_PATH} missing; run `python -m src.cli rewrite-halueval`")
    recs = load_records()
    out = df[df["example_id"].isin(recs)].copy()
    out["ok"] = out["example_id"].map(lambda e: recs[e]["ok"])
    good_pairs = out.groupby("pair_id")["ok"].transform("all") & (out.groupby("pair_id")["ok"].transform("size") == 2)
    dropped = out.loc[~good_pairs, "pair_id"].nunique()
    if dropped:
        logger.warning("HaluEval normalized: dropping %d pairs whose rewrite failed the meaning check", dropped)
    out = out[good_pairs].drop(columns="ok")
    out["response"] = out["example_id"].map(lambda e: recs[e]["rewrite"])
    out["variant"] = "normalized"
    return out


def write_audit(df: pd.DataFrame, recs: dict, n: int = 50, path: Path | None = None) -> None:
    """Spot-check file with 50 random rewritten pairs and summary statistics."""
    sub = df[df["example_id"].isin(recs)].copy()
    sub["rewrite"] = sub["example_id"].map(lambda e: recs[e]["rewrite"])
    sub["ok"] = sub["example_id"].map(lambda e: recs[e]["ok"])
    words = sub.assign(w=sub["rewrite"].str.split().str.len()).groupby("label")["w"].mean()
    rewriters = sorted({(v.get("provider"), v.get("model"), v.get("cli_version")) for v in recs.values()}, key=str)
    lines = [
        "# HaluEval rewrite audit", "",
        "- rewriter: " + "; ".join(f"{p}/{m}" + (f" (claude CLI {v})" if v else "") for p, m, v in rewriters),
        f"- answers rewritten: {len(sub)}; passed meaning check: {int(sub['ok'].sum())} ({sub['ok'].mean():.1%})",
        f"- pairs: {sub['pair_id'].nunique()}; dropped by the meaning check (either answer failed twice): "
        f"{int((~sub.groupby('pair_id')['ok'].all()).sum())}",
        f"- mean words after rewrite: faithful {words.get(0, float('nan')):.1f}, hallucinated {words.get(1, float('nan')):.1f}",
        f"- multi-line rewrites (possible meta-text, kept as returned): "
        f"{sub['rewrite'].str.contains(chr(10)).groupby(sub['label']).sum().astype(int).to_dict()}",
        f"- ends with a period after rewrite: {sub['rewrite'].str.strip().str.endswith('.').groupby(sub['label']).mean().round(3).to_dict()}",
        *(["- `claude -p` has no temperature control: the rewrites are reproducible only from the committed "
            "`artifacts/rewrites/halueval_rewrites.jsonl`, not by rerunning the command."]
          if any(p == "claude_cli" for p, _, _ in rewriters) else []),
        "", "## Random sample", "", "| label | question | original | rewrite | ok |", "|---|---|---|---|---|",
    ]
    pick = sub.drop_duplicates("pair_id").sample(min(n, sub["pair_id"].nunique()), random_state=SEED)["pair_id"]
    for r in sub[sub["pair_id"].isin(set(pick))].sort_values(["pair_id", "label"]).itertuples(index=False):
        esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")[:200]
        lines.append(f"| {r.label} | {esc(r.question)} | {esc(r.response)} | {esc(r.rewrite)} | {r.ok} |")
    (path or AUDIT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")
