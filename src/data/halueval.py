"""HaluEval QA loader (all 10,000 questions) at a pinned revision.

Each raw item has ``knowledge, question, right_answer, hallucinated_answer`` and becomes two rows:
the right answer (label 0) and the hallucinated answer (label 1), sharing one ``pair_id``.

Known artifact: right answers are short spans (about 2 words, 1% end with a period) while
hallucinated answers are full sentences (about 11 words, 95% end with a period). An answer-only
classifier reaches ~0.97 accuracy. The ``normalized`` variant replaces both answers with
label-blind rewrites into one declarative style (see ``src/data/normalize_answers.py``).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.hub import download_dataset_file
from src.data.schema import stable_id, validate

logger = logging.getLogger(__name__)

QA_FILE = "qa/data-00000-of-00001.parquet"


def load_halueval(variant: str = "original") -> pd.DataFrame:
    """Load HaluEval QA as canonical rows. ``variant`` is 'original' or 'normalized'."""
    raw = pd.read_parquet(download_dataset_file("halueval", QA_FILE))
    expected = {"knowledge", "question", "right_answer", "hallucinated_answer"}
    if not expected <= set(raw.columns):
        raise ValueError(f"HaluEval schema changed: {list(raw.columns)}")
    before = len(raw)
    raw = raw.drop_duplicates(subset=["knowledge", "question", "right_answer", "hallucinated_answer"])
    if len(raw) < before:
        logger.warning("HaluEval: dropped %d exact duplicate items", before - len(raw))

    rows = []
    for item in raw.itertuples(index=False):
        pid = stable_id("halueval", item.question, item.knowledge, item.right_answer)
        for label, resp in ((0, item.right_answer), (1, item.hallucinated_answer)):
            rows.append(
                {
                    "example_id": stable_id("halueval", pid, label),
                    "pair_id": pid,
                    "question": item.question,
                    "context": item.knowledge,
                    "response": resp,
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    # Items that share (question, knowledge, right_answer) collide on pair_id; keep the first item.
    dup = df["example_id"].duplicated()
    if dup.any():
        logger.warning("HaluEval: dropping %d rows whose pair key collides with an earlier item", int(dup.sum()))
        df = df[~dup].copy()
    df["dataset"] = "halueval"
    df["variant"] = "original"
    df["filing"] = ""
    df["target_tokens"] = 0
    df["position"] = ""
    df["task_type"] = "QA"
    df["source_file"] = QA_FILE

    if variant == "normalized":
        from src.data.normalize_answers import apply_rewrites

        df = apply_rewrites(df)
    elif variant != "original":
        raise ValueError(f"Unknown HaluEval variant {variant!r}")
    return validate(df, f"halueval[{variant}]")
