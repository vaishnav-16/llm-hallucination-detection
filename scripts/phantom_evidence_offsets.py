"""Where does the evidence sit in each long PHANTOM test document? Writes ``artifacts/phantom_evidence_offsets.csv``.

The long files are built by inserting the seed passage into filing text at the beginning, middle or end. The seed
passage is located in every long context by exact match on its first 200 characters. ``first_word`` is the number of
whitespace-separated words before its first occurrence. Every chunking unit in this project (HF tokens, Jev's 4-char
pieces) splits text at least as finely as words, so ``first_word >= W`` guarantees the passage starts after the first
W units of any detector: a ``truncate-wW`` window cannot contain it. ``evaluate`` uses this for the ``evid>=W`` groups.

About 0.5-4% of documents per cell contain the passage more than once (the filler is real filing text). The first
occurrence counts, so those rows are judged by their earliest copy. Rows where the passage is not found get no offset
and are left out of those groups.

    python scripts/phantom_evidence_offsets.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS_DIR  # noqa: E402
from src.data.store import write_table  # noqa: E402
from src.verify.slices import get_slice  # noqa: E402

logger = logging.getLogger(__name__)
KEY_CHARS = 200
OUT = ARTIFACTS_DIR / "phantom_evidence_offsets.csv"


def offsets(context: str, passage: str) -> tuple[int, int, int]:
    """(first_char, first_word, n_occurrences); -1 / -1 / 0 when the passage key is absent."""
    key = passage[:KEY_CHARS]
    first = context.find(key)
    if first < 0:
        return -1, -1, 0
    return first, len(context[:first].split()), context.count(key)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    seed = pd.concat([p.loader() for p in get_slice("phantom_seed_test")]).drop_duplicates("pair_id").set_index("pair_id")["context"]
    rows = []
    for part in get_slice("phantom_grid_full"):
        d = part.loader()
        if d.empty or d["position"].iloc[0] == "seed":
            continue
        d = d.drop_duplicates(["pair_id", "filing", "target_tokens", "position"])
        for r in d.itertuples(index=False):
            # variant='long_only' rows (a few DEF 14A groups) match no seed pair, so there is no passage to locate.
            c, w, n = offsets(r.context, seed[r.pair_id]) if r.pair_id in seed.index else (-1, -1, 0)
            rows.append({"pair_id": r.pair_id, "variant": r.variant, "filing": r.filing, "target_tokens": r.target_tokens, "position": r.position,
                         "first_char": c, "first_word": w, "n_occurrences": n, "context_words": len(r.context.split())})
        logger.info("%s: %d documents", part.part, len(d))
    out = pd.DataFrame(rows)
    missing = int((out["first_word"] < 0).sum())
    logger.info("%d documents, passage not found in %d", len(out), missing)
    write_table(out, OUT)


if __name__ == "__main__":
    main()
