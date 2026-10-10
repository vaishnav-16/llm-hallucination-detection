"""PHANTOM loader (Ji et al., NeurIPS 2025 D&B): exact file manifest, strict labels, pair ids.

Facts this loader relies on, checked on the pinned revision:
- The CSVs have columns ``Unnamed: 0, query, context, answer, ground_truth_label``, where the label is
  the string "hallucination" or "not hallucination". The old loader parsed these strings as 0 for every
  row; here any unknown label value raises an error.
- In each seed file, every (query, context) group holds exactly one faithful and one hallucinated answer.
  Rows alternate by label, so the row index must never be used as a feature.
- The long-context files (only for 10-K and DEF 14A) reuse the seed's query and both answers, with the
  seed passage embedded at the beginning, middle or end of a longer excerpt of the filing. They are a
  subset of the seed pairs. Each long group is joined back to its seed pair so that ``pair_id`` is shared
  across every length and position, which is what makes leak-free splitting and paired length curves possible.
- File names are case-sensitive on the Hub (``Phantom_10k_seed.csv``, ``Phantom_def14a_seed.csv`` but
  ``Phantom_def14A_2000tokens_middle.csv``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from src.data.hub import download_dataset_file
from src.data.schema import SchemaError, stable_id, validate

logger = logging.getLogger(__name__)

LENGTHS: tuple[int, ...] = (2000, 5000, 10000, 20000, 30000)
POSITIONS: tuple[str, ...] = ("beginning", "middle", "end")
SEED_TARGET_TOKENS = 500

FILINGS: dict[str, dict[str, str | None]] = {
    "10k": {"seed": "Phantom_10k_seed.csv", "long": "Phantom_10k_{length}tokens_{position}.csv"},
    "def14a": {"seed": "Phantom_def14a_seed.csv", "long": "Phantom_def14A_{length}tokens_{position}.csv"},
    "8k": {"seed": "Phantom_8k_seed.csv", "long": None},
    "497k": {"seed": "Phantom_497k_seed.csv", "long": None},
}
TEST_FILINGS: tuple[str, ...] = ("10k", "def14a")   # have long variants -> test only, at every length
TRAIN_FILINGS: tuple[str, ...] = ("8k", "497k")     # seed only -> train/val (the paper's own split)

LABEL_MAP = {"hallucination": 1, "not hallucination": 0}
REQUIRED_COLUMNS = ("query", "context", "answer", "ground_truth_label")


@dataclass(frozen=True)
class Cell:
    """One PHANTOM file: a filing at a nominal context length and evidence position."""

    filing: str
    target_tokens: int = SEED_TARGET_TOKENS
    position: str = "seed"

    @property
    def filename(self) -> str:
        spec = FILINGS[self.filing]
        if self.position == "seed":
            return str(spec["seed"])
        if spec["long"] is None:
            raise ValueError(f"PHANTOM filing {self.filing!r} has no long-context files")
        return str(spec["long"]).format(length=self.target_tokens, position=self.position)

    @property
    def key(self) -> str:
        return f"phantom_{self.filing}_{self.target_tokens}_{self.position}"


def all_cells(filings: tuple[str, ...] = tuple(FILINGS)) -> list[Cell]:
    """Seed cells for every filing plus every long cell for filings that have them."""
    cells = [Cell(f) for f in filings]
    for f in filings:
        if FILINGS[f]["long"] is not None:
            cells += [Cell(f, n, p) for n in LENGTHS for p in POSITIONS]
    return cells


def parse_labels(values: pd.Series, source: str) -> pd.Series:
    """Map PHANTOM's string labels to 1 (hallucinated) / 0 (faithful). Unknown values raise."""
    norm = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(norm.unique()) - set(LABEL_MAP))
    if unknown:
        raise SchemaError(f"{source}: unknown ground_truth_label values {unknown[:5]}")
    return norm.map(LABEL_MAP).astype("int8")


def read_raw(cell: Cell) -> pd.DataFrame:
    """Download and read one PHANTOM CSV, with labels parsed and the index column dropped."""
    path = download_dataset_file("phantom", f"PhantomDataset/{cell.filename}")
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise SchemaError(f"{cell.filename}: missing columns {missing}; got {list(raw.columns)}")
    raw = raw[list(REQUIRED_COLUMNS)].copy()
    if raw[["query", "context", "answer"]].isna().any().any():
        raise SchemaError(f"{cell.filename}: null text fields")
    raw["label"] = parse_labels(raw.pop("ground_truth_label"), cell.filename)
    return raw


def _complete_groups(raw: pd.DataFrame, source: str, strict: bool) -> pd.DataFrame:
    """Keep only (query, context) groups that hold exactly one faithful and one hallucinated answer."""
    g = raw.groupby(["query", "context"], sort=False)["label"]
    ok = (g.transform("size") == 2) & (g.transform("sum") == 1)
    if not ok.all():
        n_bad = int((~ok).sum())
        if strict:
            raise SchemaError(f"{source}: {n_bad} rows in groups without exactly one answer per label")
        logger.warning("%s: dropping %d rows in incomplete pairs", source, n_bad)
    return raw[ok].copy()


def _finish(rows: pd.DataFrame, cell: Cell) -> pd.DataFrame:
    rows = rows.rename(columns={"query": "question", "answer": "response"})
    rows["example_id"] = [
        stable_id("phantom", pid, lab, cell.target_tokens, cell.position) for pid, lab in zip(rows["pair_id"], rows["label"])
    ]
    rows["dataset"] = "phantom"
    rows["variant"] = "original"
    rows["filing"] = cell.filing
    rows["target_tokens"] = cell.target_tokens
    rows["position"] = cell.position
    rows["task_type"] = "QA"
    rows["source_file"] = cell.filename
    return validate(rows, cell.filename)


def load_seed(filing: str) -> pd.DataFrame:
    """Load a seed file (~500-token passages). Every pair must be complete."""
    cell = Cell(filing)
    raw = _complete_groups(read_raw(cell), cell.filename, strict=True)
    raw["pair_id"] = [stable_id("phantom", filing, q, c) for q, c in zip(raw["query"], raw["context"])]
    return _finish(raw, cell)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_long(cell: Cell, seed: pd.DataFrame, max_unjoined: float = 0.05) -> pd.DataFrame:
    """Load a long-context file and attach the pair_id of the seed pair it was built from.

    Each long group (query, long context) is joined to a seed pair in two steps:
    1. exact (query, faithful answer, hallucinated answer) match. If several seed pairs share it, the one
       whose passage appears verbatim inside the long context wins. These rows get ``variant='original'``.
    2. otherwise, same query and the seed passage appears verbatim (whitespace-normalized) inside the long
       context, but the answers differ. The 10-K files never need this step; in the DEF 14A long files
       31% of pairs have answers regenerated relative to the seed (see the data card). These rows get
       ``variant='reanswered'``: same question and evidence, so the split is safe, but they are not
       answer-identical across lengths.
    Groups matching neither get their own pair id and ``variant='long_only'``; more than
    ``max_unjoined`` of them raises, because that would mean the file no longer matches its seed.
    """
    raw = _complete_groups(read_raw(cell), cell.filename, strict=False)

    seed_pairs = seed.pivot_table(index=["pair_id", "question", "context"], columns="label", values="response", aggfunc="first")
    seed_pairs = seed_pairs.reset_index().rename(columns={0: "a0", 1: "a1"})
    by_triple: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    by_query: dict[str, list[tuple[str, str]]] = {}
    for r in seed_pairs.itertuples(index=False):
        probe = _squash(r.context)[:300]
        by_triple.setdefault((r.question, r.a0, r.a1), []).append((r.pair_id, probe))
        by_query.setdefault(_squash(r.question), []).append((r.pair_id, probe))

    groups = raw.pivot_table(index=["query", "context"], columns="label", values="answer", aggfunc="first").reset_index()
    groups = groups.rename(columns={0: "a0", 1: "a1"})
    assigned: dict[tuple[str, str], tuple[str, str]] = {}
    counts = {"original": 0, "reanswered": 0, "long_only": 0}
    for g in groups.itertuples(index=False):
        ctx = _squash(g.context)
        cands = by_triple.get((g.query, g.a0, g.a1), [])
        inside = [pid for pid, probe in cands if probe in ctx]
        if len(cands) == 1 or len(inside) == 1:
            pid, variant = (cands[0][0] if len(cands) == 1 else inside[0]), "original"
        else:
            inside = [pid for pid, probe in by_query.get(_squash(g.query), []) if probe in ctx]
            if len(inside) == 1:
                pid, variant = inside[0], "reanswered"
            else:
                pid, variant = stable_id("phantom", cell.filing, "long_only", g.query, g.a0, g.a1), "long_only"
        counts[variant] += 1
        assigned[(g.query, g.context)] = (pid, variant)

    n = max(1, len(groups))
    if counts["long_only"] / n > max_unjoined:
        raise SchemaError(f"{cell.filename}: {counts['long_only']} of {len(groups)} groups match no seed pair")
    pids = [pid for pid, _ in assigned.values()]
    if len(set(pids)) != len(pids):
        raise SchemaError(f"{cell.filename}: two long groups map to the same seed pair")

    raw["pair_id"] = [assigned[(q, c)][0] for q, c in zip(raw["query"], raw["context"])]
    variants = [assigned[(q, c)][1] for q, c in zip(raw["query"], raw["context"])]
    logger.info("%s: %d pairs joined (%s)", cell.filename, len(groups), counts)
    out = _finish(raw, cell)
    out["variant"] = variants
    out.attrs["join_counts"] = counts
    return out
