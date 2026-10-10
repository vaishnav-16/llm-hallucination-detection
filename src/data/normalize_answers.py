"""Label-blind rewrite of HaluEval answers into one declarative style (the ``normalized`` variant).

Safeguards (see the plan's "Safeguards for the HaluEval answer rewrite"):
- The rewriter never learns which answer is correct: every answer gets the identical prompt as a
  separate call, and calls are issued in a seeded random order that interleaves both labels.
- Meaning check: the rewrite must keep the original answer's content words (all of them for short
  answers of 4 words or fewer, at least 80% otherwise). A failed check is retried once with a stricter
  prompt; if it fails again the whole pair is dropped and counted.
- The rewriter model must not also be used as a judge on HaluEval (``evaluate`` warns if it is).

By default only the val/test pairs plus 2,000 train pairs are rewritten (about 10K calls).
"""

from __future__ import annotations

import json
import logging
import re

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


def rewrite_halueval(provider: str = "dartmouth", limit: int | None = None, train_pairs: int = 2000) -> None:
    """Rewrite selected HaluEval answers through ``provider``; resumable via the API cache."""
    from src.data.halueval import load_halueval
    from src.detectors.api_judge import ApiClient

    df = load_halueval("original")
    pairs = sorted(_select_pairs(df, train_pairs))
    if limit:
        pairs = pairs[:limit]
    rows = df[df["pair_id"].isin(set(pairs))].sample(frac=1.0, random_state=SEED)  # interleave labels
    client = ApiClient(provider)
    done = {}
    if REWRITES_PATH.exists():
        with open(REWRITES_PATH, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done[rec["example_id"]] = rec
    REWRITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with open(REWRITES_PATH, "a", encoding="utf-8") as out:
        for r in rows.itertuples(index=False):
            if r.example_id in done:
                continue
            prompt = PROMPT.format(question=r.question, answer=r.response)
            text, ok = "", False
            for attempt in range(2):
                resp = client.chat([{"role": "user", "content": prompt + (STRICT_SUFFIX if attempt else "")}], temperature=0, max_tokens=120)
                text = (resp["choices"][0]["message"].get("content") or "").strip().strip('"')
                ok = bool(text) and meaning_preserved(r.response, text)
                if ok:
                    break
            rec = {"example_id": r.example_id, "pair_id": r.pair_id, "rewrite": text, "ok": ok, "provider": provider, "model": client.model}
            out.write(json.dumps(rec) + "\n")
            done[r.example_id] = rec
            n_new += 1
            if n_new % 200 == 0:
                logger.info("rewrote %d answers (%d total)", n_new, len(done))
    logger.info("Rewrite finished: %d answers on file, %d ok", len(done), sum(v["ok"] for v in done.values()))
    write_audit(df, done)
    from src.data.prepare import NORM_DIR

    write_table(load_halueval("normalized"), NORM_DIR / "halueval_normalized.parquet")


def apply_rewrites(df: pd.DataFrame) -> pd.DataFrame:
    """Replace responses with their rewrites; keep only pairs where both answers passed the check."""
    if not REWRITES_PATH.exists():
        raise FileNotFoundError(f"{REWRITES_PATH} missing; run `python -m src.cli rewrite-halueval`")
    recs = {}
    with open(REWRITES_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            recs[rec["example_id"]] = rec
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


def write_audit(df: pd.DataFrame, recs: dict, n: int = 50) -> None:
    """Spot-check file with 50 random rewritten pairs and summary statistics."""
    sub = df[df["example_id"].isin(recs)].copy()
    sub["rewrite"] = sub["example_id"].map(lambda e: recs[e]["rewrite"])
    sub["ok"] = sub["example_id"].map(lambda e: recs[e]["ok"])
    words = sub.assign(w=sub["rewrite"].str.split().str.len()).groupby("label")["w"].mean()
    lines = [
        "# HaluEval rewrite audit", "",
        f"- answers rewritten: {len(sub)}; passed meaning check: {int(sub['ok'].sum())} ({sub['ok'].mean():.1%})",
        f"- mean words after rewrite: faithful {words.get(0, float('nan')):.1f}, hallucinated {words.get(1, float('nan')):.1f}",
        f"- ends with a period after rewrite: {sub['rewrite'].str.strip().str.endswith('.').groupby(sub['label']).mean().round(3).to_dict()}",
        "", "## Random sample", "", "| label | question | original | rewrite | ok |", "|---|---|---|---|---|",
    ]
    pick = sub.drop_duplicates("pair_id").sample(min(n, sub["pair_id"].nunique()), random_state=SEED)["pair_id"]
    for r in sub[sub["pair_id"].isin(set(pick))].sort_values(["pair_id", "label"]).itertuples(index=False):
        esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")[:200]
        lines.append(f"| {r.label} | {esc(r.question)} | {esc(r.response)} | {esc(r.rewrite)} | {r.ok} |")
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
