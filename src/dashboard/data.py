"""Pure data helpers for the dashboard: read the committed v2 artifacts and slice them for display.

Why a separate, Streamlit-free module: the dashboard must work from ``artifacts/`` alone (on any machine,
without the git-ignored ``data/``), and these functions decide what numbers are shown, so they are unit-tested
offline on synthetic frames. Pages wrap the loaders in ``st.cache_data`` (see ``state.py``).

Conventions mirrored from ``src/eval/evaluate.py`` so the dashboard never disagrees with the report:
- ``run_id = <detector>__<strategy-tag>__<slice>[__ablate]``; the slice is the third ``__`` field and the
  ``__ablate`` suffix belongs to the strategy (it does in ``metrics_long`` and in the prediction files).
- a row is predicted hallucinated when ``p_halluc >= threshold``; the reported threshold is the one
  ``evaluate`` chose on the matching validation slice and stored in ``metrics_long``.

Every loader takes an optional ``root`` (an artifacts directory) and otherwise resolves the module-level
``ARTIFACTS_DIR`` at call time, so tests can point the whole dashboard at a temporary directory.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.eval.metrics import threshold_metrics

logger = logging.getLogger(__name__)

# Answer-only detectors: they never see the context, so they are drawn as reference lines, not competitors.
BASELINE_DETECTORS: tuple[str, ...] = ("answer_tfidf", "answer_length")
# Baseline fallback when a slice has no baseline rows (they are only run on the full PHANTOM grid).
BASELINE_FALLBACK_SLICE = "phantom_grid_full"
DATASET_ORDER = ("halueval", "ragtruth", "phantom")
POSITIONS = ("beginning", "middle", "end")
DEFAULT_THRESHOLD = 0.5

METRICS_COLUMNS = ["detector", "strategy", "slice", "group", "n", "n_pairs", "threshold", "threshold_source",
                   "metric", "value", "ci_lo", "ci_hi"]
COMPARISON_COLUMNS = ["detector", "slice", "a", "b", "group", "n", "metric", "diff", "ci_lo", "ci_hi", "p_value"]
SEED_COLUMNS = ["base", "strategy", "slice", "group", "metric", "n_seeds", "mean", "sd"]
RUN_COLUMNS = ["run_id", "detector", "strategy", "slice", "dataset", "path", "has_manifest"]
ERROR_COLUMNS = ["example_id", "pair_id", "dataset", "filing", "target_tokens", "position", "task_type", "variant",
                 "faithful_chars", "label", "p_halluc", "n_chunks", "part"]


# --------------------------------------------------------------------------- paths and status

def artifacts_root(root: Path | str | None = None) -> Path:
    """The artifacts directory to read from (explicit ``root`` wins over ``ARTIFACTS_DIR``)."""
    return Path(root) if root is not None else Path(ARTIFACTS_DIR)


def metrics_dir(root: Path | str | None = None) -> Path:
    return artifacts_root(root) / "metrics"


def predictions_dir(root: Path | str | None = None) -> Path:
    return artifacts_root(root) / "predictions"


def figures_dir(root: Path | str | None = None) -> Path:
    return artifacts_root(root) / "figures"


def artifact_status(root: Path | str | None = None) -> dict[str, str]:
    """Human-readable availability of each artifact the dashboard reads (for the sidebar expander)."""
    status: dict[str, str] = {}
    for name in ("metrics_long", "comparisons", "seed_summary", "headline", "shortcut_baselines"):
        path = metrics_dir(root) / f"{name}.csv"
        status[f"metrics/{name}.csv"] = "present" if path.exists() else "missing"
    pred = predictions_dir(root)
    n_pred = len(list(pred.glob("*.parquet"))) if pred.exists() else 0
    n_man = len(list((pred / "manifest").glob("*.json"))) if (pred / "manifest").exists() else 0
    status["predictions/*.parquet"] = f"{n_pred} files"
    status["predictions/manifest/*.json"] = f"{n_man} files" + (" + legacy manifest.json" if (pred / "manifest.json").exists() else "")
    fig = figures_dir(root)
    status["figures/*.png"] = f"{len(list(fig.glob('*.png'))) if fig.exists() else 0} files"
    return status


def file_mtime(path: Path) -> float:
    """Modification time used as a cache key, so the dashboard reloads when an artifact is rewritten."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- metrics tables

def dataset_of(slice_name: str) -> str:
    """'halueval_norm_test' -> 'halueval', 'phantom_grid_sub' -> 'phantom'."""
    return slice_name.split("_", 1)[0]


def is_baseline(detector: str) -> bool:
    return detector in BASELINE_DETECTORS


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        logger.warning("%s not found", path)
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    return df


def load_metrics_long(root: Path | str | None = None) -> pd.DataFrame:
    """metrics_long.csv with an added ``dataset`` column (empty frame if missing)."""
    df = _read_csv(metrics_dir(root) / "metrics_long.csv", METRICS_COLUMNS)
    for col in ("detector", "strategy", "slice", "group", "metric", "threshold_source"):
        if col in df:
            df[col] = df[col].astype(str)
    df["dataset"] = df["slice"].map(dataset_of) if len(df) else pd.Series(dtype=str)
    return df


def load_comparisons(root: Path | str | None = None) -> pd.DataFrame:
    return _read_csv(metrics_dir(root) / "comparisons.csv", COMPARISON_COLUMNS)


def load_seed_summary(root: Path | str | None = None) -> pd.DataFrame:
    return _read_csv(metrics_dir(root) / "seed_summary.csv", SEED_COLUMNS)


def ordered_datasets(slices: list[str] | pd.Series) -> list[str]:
    """Datasets present in ``slices``, in the canonical order, unknown ones last."""
    present = sorted({dataset_of(s) for s in slices})
    return [d for d in DATASET_ORDER if d in present] + [d for d in present if d not in DATASET_ORDER]


def group_sort_key(group: str) -> tuple[int, int, str]:
    """Order groups: all, hard, then len=<N> numerically, then everything else alphabetically."""
    if group == "all":
        return (0, 0, "")
    if group == "hard":
        return (1, 0, "")
    m = re.match(r"len=(\d+)(.*)$", group)
    if m:
        return (2, int(m.group(1)), m.group(2))
    if group.startswith("pos="):
        pos = group[4:]
        return (3, POSITIONS.index(pos) if pos in POSITIONS else 9, pos)
    return (4, 0, group)


def headline_table(long: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """One row per (detector, strategy): ``value [lo, hi]`` for each test slice's ``all`` and ``hard`` groups.

    Re-implemented here (rather than importing ``src.eval.evaluate.headline``) to keep the dashboard light,
    and extended with a ``type`` column so answer-only baselines are visibly marked.
    """
    h = long[(long["metric"] == metric) & (long["group"].isin(["all", "hard"]))].copy()
    if h.empty:
        return pd.DataFrame(columns=["type", "detector", "strategy"])
    h["col"] = h["slice"] + h["group"].map(lambda g: "" if g == "all" else f" ({g})")
    h["cell"] = [fmt_ci(v, lo, hi) for v, lo, hi in zip(h["value"], h["ci_lo"], h["ci_hi"])]
    out = h.pivot_table(index=["detector", "strategy"], columns="col", values="cell", aggfunc="first").reset_index()
    out.columns.name = None
    out.insert(0, "type", np.where(out["detector"].map(is_baseline), "answer-only baseline", "detector"))
    slice_cols = sorted([c for c in out.columns if c not in ("type", "detector", "strategy")],
                        key=lambda c: (DATASET_ORDER.index(dataset_of(c)) if dataset_of(c) in DATASET_ORDER else 9, c))
    out = out[["type", "detector", "strategy", *slice_cols]]
    order = out["detector"].map(is_baseline).map({True: 0, False: 1})  # baselines first
    return out.assign(_o=order).sort_values(["_o", "detector", "strategy"]).drop(columns="_o").reset_index(drop=True)


def fmt_ci(value: float, lo: float, hi: float, digits: int = 3) -> str:
    """'0.812 [0.790, 0.833]', or just the value when the CI is missing."""
    if pd.isna(value):
        return ""
    if pd.isna(lo) or pd.isna(hi):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def results_frame(long: pd.DataFrame, slice_name: str, group: str, metric: str,
                  detectors: list[str] | None = None, strategies: list[str] | None = None) -> pd.DataFrame:
    """Rows of metrics_long for one slice/group/metric, optionally filtered, sorted by value (best first)."""
    df = long[(long["slice"] == slice_name) & (long["group"] == group) & (long["metric"] == metric)]
    if detectors is not None:
        df = df[df["detector"].isin(detectors)]
    if strategies is not None:
        df = df[df["strategy"].isin(strategies)]
    df = df.copy()
    df["label"] = [_label(d, s) for d, s in zip(df["detector"], df["strategy"])]
    ascending = metric == "ece"
    return df.sort_values("value", ascending=ascending).reset_index(drop=True)


def _label(detector: str, strategy: str) -> str:
    return detector if strategy in ("", "none") else f"{detector} · {strategy}"


def baseline_reference(long: pd.DataFrame, slice_name: str, group: str, metric: str,
                       baselines: tuple[str, ...] = BASELINE_DETECTORS) -> tuple[pd.DataFrame, str | None]:
    """Answer-only baseline rows for (slice, group, metric).

    Returns ``(rows, source_slice)``. If the slice itself has no baseline rows and it is a PHANTOM slice,
    the same group on ``phantom_grid_full`` is used instead (a superset of the rows, so not a paired
    comparison); ``source_slice`` then differs from ``slice_name`` so the caller can say so.
    ``source_slice`` is None when no baseline is available at all.
    """
    def rows(sl: str) -> pd.DataFrame:
        return long[(long["slice"] == sl) & (long["group"] == group) & (long["metric"] == metric)
                    & (long["detector"].isin(baselines))]

    found = rows(slice_name)
    if len(found):
        return found.reset_index(drop=True), slice_name
    if dataset_of(slice_name) == "phantom" and slice_name != BASELINE_FALLBACK_SLICE:
        found = rows(BASELINE_FALLBACK_SLICE)
        if len(found):
            return found.reset_index(drop=True), BASELINE_FALLBACK_SLICE
    return found.reset_index(drop=True), None


def _len_of(group: str) -> int | None:
    m = re.fullmatch(r"len=(\d+)(\|hard)?", group)
    return int(m.group(1)) if m else None


def length_curve(long: pd.DataFrame, slice_name: str, metric: str, detectors: list[str] | None = None,
                 strategies: list[str] | None = None, hard: bool = False) -> pd.DataFrame:
    """Metric vs target context length from the pooled ``len=<N>`` (or ``len=<N>|hard``) groups."""
    want = r"len=\d+\|hard" if hard else r"len=\d+"
    df = long[(long["slice"] == slice_name) & (long["metric"] == metric) & long["group"].str.fullmatch(want)]
    if detectors is not None:
        df = df[df["detector"].isin(detectors)]
    if strategies is not None:
        df = df[df["strategy"].isin(strategies)]
    df = df.copy()
    df["target_tokens"] = df["group"].map(_len_of).astype(int) if len(df) else pd.Series(dtype=int)
    df["label"] = [_label(d, s) for d, s in zip(df["detector"], df["strategy"])]
    return df.sort_values(["label", "target_tokens"]).reset_index(drop=True)


def cell_curve(long: pd.DataFrame, slice_name: str, metric: str, filing: str, detectors: list[str] | None = None,
               strategies: list[str] | None = None) -> pd.DataFrame:
    """Metric vs length for one filing, one row per (length, position) from the ``cell=`` groups.

    Cells are per evidence position, so pooling them over positions would need a new bootstrap; the caller
    facets by ``position`` instead. The 500-token seed cell has position ``seed`` (whole document, no
    position) and is repeated in every position facet as the shortest-context anchor.
    """
    pat = re.compile(rf"cell={re.escape(filing)}\|(\d+)\|(\w+)")
    df = long[(long["slice"] == slice_name) & (long["metric"] == metric)]
    if detectors is not None:
        df = df[df["detector"].isin(detectors)]
    if strategies is not None:
        df = df[df["strategy"].isin(strategies)]
    parts = df["group"].str.extract(pat)
    df = df[parts[0].notna()].copy()
    parts = parts[parts[0].notna()]
    df["target_tokens"] = parts[0].astype(int)
    df["position"] = parts[1]
    seed = df[df["position"] == "seed"]
    rest = df[df["position"] != "seed"]
    positions = [p for p in POSITIONS if p in set(rest["position"])]
    out = pd.concat([rest, *[seed.assign(position=p) for p in positions]], ignore_index=True)
    out["label"] = [_label(d, s) for d, s in zip(out["detector"], out["strategy"])]
    return out.sort_values(["label", "position", "target_tokens"]).reset_index(drop=True)


def filings_with_cells(long: pd.DataFrame, slice_name: str) -> list[str]:
    groups = long.loc[long["slice"] == slice_name, "group"].drop_duplicates()
    return sorted(groups.str.extract(r"^cell=([^|]+)\|")[0].dropna().unique().tolist())


def position_frame(long: pd.DataFrame, slice_name: str, metric: str, detectors: list[str] | None = None,
                   strategies: list[str] | None = None) -> pd.DataFrame:
    """Rows of the ``pos=<position>`` groups, with an ordered ``position`` column."""
    df = long[(long["slice"] == slice_name) & (long["metric"] == metric) & long["group"].str.startswith("pos=")]
    if detectors is not None:
        df = df[df["detector"].isin(detectors)]
    if strategies is not None:
        df = df[df["strategy"].isin(strategies)]
    df = df.copy()
    df["position"] = pd.Categorical(df["group"].str[4:], categories=list(POSITIONS), ordered=True)
    df["label"] = [_label(d, s) for d, s in zip(df["detector"], df["strategy"])]
    return df.sort_values(["label", "position"]).reset_index(drop=True)


def baseline_curve(long: pd.DataFrame, slice_name: str, metric: str, kind: str = "length",
                   hard: bool = False) -> tuple[pd.DataFrame, str | None]:
    """Answer-only baseline curve (by length or position) with the same grid_full fallback as bars."""
    def build(sl: str) -> pd.DataFrame:
        if kind == "length":
            return length_curve(long, sl, metric, list(BASELINE_DETECTORS), None, hard=hard)
        return position_frame(long, sl, metric, list(BASELINE_DETECTORS), None)

    found = build(slice_name)
    if len(found):
        return found, slice_name
    if dataset_of(slice_name) == "phantom" and slice_name != BASELINE_FALLBACK_SLICE:
        found = build(BASELINE_FALLBACK_SLICE)
        if len(found):
            return found, BASELINE_FALLBACK_SLICE
    return found, None


# --------------------------------------------------------------------------- prediction runs

def parse_run_id(run_id: str) -> dict[str, str]:
    """Split ``<detector>__<strategy>__<slice>[__ablate]`` the way ``evaluate`` does."""
    parts = run_id.split("__")
    if len(parts) < 3:
        return {"detector": parts[0], "strategy": parts[1] if len(parts) > 1 else "", "slice": ""}
    return {"detector": parts[0], "strategy": "__".join([parts[1], *parts[3:]]), "slice": parts[2]}


def load_manifests(root: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """All run records: a legacy single ``manifest.json`` (if present) overlaid by ``manifest/<run_id>.json``.

    Mirrors ``src.verify.runner._manifest``; unreadable files are logged and skipped.
    """
    pred = predictions_dir(root)
    man: dict[str, dict[str, Any]] = {}
    legacy = pred / "manifest.json"
    if legacy.exists():
        try:
            man.update(json.loads(legacy.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            logger.warning("could not read %s: %s", legacy, e)
    for p in sorted((pred / "manifest").glob("*.json")):
        try:
            man[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("could not read %s: %s", p, e)
    return man


def list_runs(root: Path | str | None = None) -> pd.DataFrame:
    """One row per prediction file, with detector/strategy/slice parsed from its run_id."""
    pred = predictions_dir(root)
    files = sorted(pred.glob("*.parquet")) if pred.exists() else []
    man = load_manifests(root)
    rows = []
    for f in files:
        info = parse_run_id(f.stem)
        rows.append({"run_id": f.stem, **info, "dataset": dataset_of(info["slice"]), "path": str(f),
                     "has_manifest": f.stem in man})
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def load_predictions(run_id: str, root: Path | str | None = None) -> pd.DataFrame:
    """One run's predictions (raises FileNotFoundError if absent)."""
    path = predictions_dir(root) / f"{run_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def val_threshold(long: pd.DataFrame, detector: str, strategy: str, slice_name: str) -> tuple[float | None, str]:
    """The decision threshold ``evaluate`` stored for this run, and where it came from.

    - test slice with metrics rows: that row's threshold and ``threshold_source`` (a validation slice, or
      ``default`` when ``evaluate`` had no validation predictions and fell back to 0.5);
    - a validation slice itself (no metrics rows): the threshold that was fitted on it for this
      detector/strategy, if any;
    - otherwise ``(None, "")`` and the caller falls back to 0.5.
    """
    if long.empty:
        return None, ""
    base = long[(long["detector"] == detector) & (long["strategy"] == strategy)]
    rows = base[base["slice"] == slice_name]
    if len(rows):
        row = rows[rows["group"] == "all"].iloc[0] if (rows["group"] == "all").any() else rows.iloc[0]
        return float(row["threshold"]), str(row["threshold_source"])
    fitted = base[base["threshold_source"] == slice_name]
    if len(fitted):
        return float(fitted["threshold"].iloc[0]), slice_name
    return None, ""


def threshold_summary(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts plus accuracy/BA/precision/recall/F1 at ``threshold`` (``p >= t`` = hallucinated).

    Uses ``src.eval.metrics.threshold_metrics`` so numbers match metrics_long exactly.
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    out: dict[str, float] = dict(threshold_metrics(y, p, threshold))
    out.update({
        "tp": int(((pred == 1) & (y == 1)).sum()), "tn": int(((pred == 0) & (y == 0)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()),
    })
    return out


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> np.ndarray:
    """2x2 matrix [[TN, FP], [FN, TP]] (rows = true faithful/hallucinated, cols = predicted)."""
    s = threshold_summary(y, p, threshold)
    return np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]])


def top_errors(preds: pd.DataFrame, threshold: float, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(false positives, false negatives) at ``threshold``.

    False positives: faithful rows flagged as hallucinated, highest ``p_halluc`` first.
    False negatives: hallucinated rows passed as faithful, lowest ``p_halluc`` first.
    """
    cols = [c for c in ERROR_COLUMNS if c in preds.columns]
    fp = preds[(preds["label"] == 0) & (preds["p_halluc"] >= threshold)].sort_values("p_halluc", ascending=False)
    fn = preds[(preds["label"] == 1) & (preds["p_halluc"] < threshold)].sort_values("p_halluc", ascending=True)
    return fp[cols].head(n).reset_index(drop=True), fn[cols].head(n).reset_index(drop=True)


def attach_text(rows: pd.DataFrame, slice_name: str, max_context_chars: int = 400) -> pd.DataFrame | None:
    """Join question/response/context text onto prediction rows from the local (git-ignored) ``data/``.

    Text is not stored in the artifacts. Only the parts the rows belong to are loaded, through the
    project's own slice loaders. Returns None when ``data/`` (or anything it needs) is unavailable,
    so the caller can show a caption instead of failing.
    """
    if rows.empty:
        return rows
    try:
        from src.verify.slices import get_slice

        needed = set(rows["part"]) if "part" in rows else None
        texts = []
        for part in get_slice(slice_name):
            if needed is not None and part.part not in needed:
                continue
            df = part.loader()
            texts.append(df[["example_id", "question", "response", "context"]])
        if not texts:
            return None
        text = pd.concat(texts, ignore_index=True).drop_duplicates("example_id")
    except Exception as e:  # missing data/, missing splits, unknown slice: degrade, never crash the page
        logger.info("text join unavailable for %s: %s", slice_name, e)
        return None
    ctx = text["context"].astype(str)
    text["context"] = ctx.str.slice(0, max_context_chars) + np.where(ctx.str.len() > max_context_chars, " ...", "")
    return rows.merge(text, on="example_id", how="left")
