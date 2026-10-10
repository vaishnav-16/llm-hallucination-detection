"""`evaluate` stage: metrics for every committed prediction file, with CIs and paired comparisons.

For each (detector, strategy) and each test slice:
- the decision threshold is chosen on the matching validation slice (PHANTOM 8-K/497K val, HaluEval val,
  RAGTruth val) by balanced accuracy. If no validation predictions exist, 0.5 is used and flagged.
- metrics are computed for the whole slice and for groups: context length (pooled), each PHANTOM cell,
  evidence position, the length-matched hard subset (faithful answer >= 224 chars), seed-identical
  ("original") rows only, and RAGTruth task type.
- 95% CIs come from a pair-clustered bootstrap.

Paired comparisons (same rows, aligned by example_id): each chunking strategy vs ``truncate`` for the same
detector, and each detector vs the answer-only TF-IDF baseline, per context length.

Outputs ``artifacts/metrics/metrics_long.csv``, ``comparisons.csv`` and ``headline.csv``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.data.store import read_table, write_table
from src.eval.metrics import all_metrics, cluster_bootstrap, paired_bootstrap_diff
from src.eval.threshold import choose_threshold
from src.verify.slices import HARD_FAITHFUL_CHARS

logger = logging.getLogger(__name__)

METRICS_DIR = ARTIFACTS_DIR / "metrics"
VAL_FOR = {
    "phantom_grid_sub": "phantom_val", "phantom_grid_full": "phantom_val", "phantom_seed_test": "phantom_val",
    "halueval_test": "halueval_val", "halueval_norm_test": "halueval_norm_val", "ragtruth_test": "ragtruth_val",
}
CI_METRICS = ["auroc", "f1", "balanced_accuracy"]


def load_predictions(pred_dir=ARTIFACTS_DIR / "predictions") -> pd.DataFrame:
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no prediction files in {pred_dir}")
    df = pd.concat([read_table(f) for f in files], ignore_index=True)
    df["slice"] = df["run_id"].str.split("__").str[2]
    return df


def _groups(df: pd.DataFrame, slice_name: str) -> list[tuple[str, pd.Series]]:
    out = [("all", pd.Series(True, index=df.index))]
    if slice_name.startswith("phantom"):
        for n in sorted(df["target_tokens"].unique()):
            m = df["target_tokens"] == n
            out.append((f"len={n}", m))
            out.append((f"len={n}|hard", m & (df["faithful_chars"] >= HARD_FAITHFUL_CHARS)))
            out.append((f"len={n}|original", m & (df["variant"] == "original")))
        for pos in ("beginning", "middle", "end"):
            out.append((f"pos={pos}", df["position"] == pos))
        out.append(("hard", df["faithful_chars"] >= HARD_FAITHFUL_CHARS))
        for f in sorted(df["filing"].unique()):
            out.append((f"filing={f}", df["filing"] == f))
        if df["position"].nunique() > 1:
            for (f, n, pos), idx in df.groupby(["filing", "target_tokens", "position"]).groups.items():
                out.append((f"cell={f}|{n}|{pos}", df.index.isin(idx)))
    if slice_name.startswith("ragtruth"):
        for t in sorted(df["task_type"].unique()):
            out.append((f"task={t}", df["task_type"] == t))
    return [(g, pd.Series(m, index=df.index)) for g, m in out]


def evaluate_all(n_boot: int = 1000, n_boot_cells: int = 200, pred_dir=ARTIFACTS_DIR / "predictions", out_dir=METRICS_DIR) -> pd.DataFrame:
    preds = load_predictions(pred_dir)
    rows = []
    for (det, strat), d in preds.groupby(["detector", "strategy"]):
        for sl, test in d.groupby("slice"):
            if sl not in VAL_FOR:
                continue
            val = d[d["slice"] == VAL_FOR[sl]]
            if len(val) and val["label"].nunique() == 2:
                thr, src = choose_threshold(val["label"].to_numpy(), val["p_halluc"].to_numpy(), "balanced_accuracy"), VAL_FOR[sl]
            else:
                thr, src = 0.5, "default"
            test = test.reset_index(drop=True)
            for g, mask in _groups(test, sl):
                sub = test[mask.to_numpy()]
                if len(sub) < 20 or sub["label"].nunique() < 2:
                    continue
                y, p, grp = sub["label"].to_numpy(), sub["p_halluc"].to_numpy(), sub["pair_id"].to_numpy()
                m = all_metrics(y, p, thr)
                ci = cluster_bootstrap(y, p, grp, thr, CI_METRICS, n_boot=n_boot_cells if g.startswith("cell=") else n_boot)
                base = {"detector": det, "strategy": strat, "slice": sl, "group": g, "n": len(sub), "n_pairs": sub["pair_id"].nunique(),
                        "threshold": thr, "threshold_source": src}
                for k, v in m.items():
                    lo, hi = ci.get(k, (np.nan, np.nan))
                    rows.append({**base, "metric": k, "value": v, "ci_lo": lo, "ci_hi": hi})
            logger.info("evaluated %s / %s / %s (threshold %.3f from %s)", det, strat, sl, thr, src)
    long = pd.DataFrame(rows)
    if long.empty:
        raise RuntimeError("no (detector, strategy, test slice) combination had enough rows to evaluate")
    write_table(long, out_dir / "metrics_long.csv")
    write_table(_comparisons(preds), out_dir / "comparisons.csv")
    write_table(headline(long), out_dir / "headline.csv")
    write_table(seed_summary(long), out_dir / "seed_summary.csv")
    return long


def _paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a[["example_id", "pair_id", "label", "target_tokens", "p_halluc"]].merge(
        b[["example_id", "p_halluc"]], on="example_id", suffixes=("_a", "_b")
    )


def _comparisons(preds: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    rows = []
    ph = preds[preds["slice"].isin(["phantom_grid_sub", "phantom_grid_full"])]
    base = preds[(preds["detector"] == "answer_tfidf") & (preds["slice"] == "phantom_grid_full")]
    for (det, sl), d in ph.groupby(["detector", "slice"]):
        if det.startswith("answer_") or det == "lexical_support":
            continue
        strategies = d["strategy"].unique()
        ref = [s for s in strategies if s.startswith("truncate")]
        pairs = [(s, ref[0]) for s in strategies if ref and s != ref[0]] if ref else []
        for s in strategies:
            pairs.append((s, "answer_tfidf"))
        for sa, sb in pairs:
            a = d[d["strategy"] == sa]
            b = base if sb == "answer_tfidf" else d[d["strategy"] == sb]
            m = _paired(a, b)
            if m.empty:
                continue
            for n, g in [("all", m)] + [(f"len={k}", v) for k, v in m.groupby("target_tokens")]:
                if g["label"].nunique() < 2:
                    continue
                r = paired_bootstrap_diff(g["label"].to_numpy(), g["p_halluc_a"].to_numpy(), g["p_halluc_b"].to_numpy(),
                                          g["pair_id"].to_numpy(), "auroc", n_boot=n_boot)
                rows.append({"detector": det, "slice": sl, "a": sa, "b": f"{'answer_tfidf' if sb == 'answer_tfidf' else det}:{sb}",
                             "group": n, "n": len(g), "metric": "auroc", **r})
    cols = ["detector", "slice", "a", "b", "group", "n", "metric", "diff", "ci_lo", "ci_hi", "p_value"]
    return pd.DataFrame(rows, columns=cols)


def headline(long: pd.DataFrame) -> pd.DataFrame:
    """One row per detector x strategy: AUROC (CI) on each test slice's 'all' group plus PHANTOM hard."""
    h = long[(long["metric"] == "auroc") & (long["group"].isin(["all", "hard"]))].copy()
    h["col"] = h["slice"] + h["group"].map(lambda g: "" if g == "all" else f" ({g})")
    h["cell"] = h.apply(lambda r: f"{r['value']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1)
    return h.pivot_table(index=["detector", "strategy"], columns="col", values="cell", aggfunc="first").reset_index()


def seed_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over training seeds for runs named <base>_s<seed> (same strategy, slice and group)."""
    m = long[long["metric"].isin(["auroc", "f1", "balanced_accuracy"]) & long["group"].isin(["all", "hard"])].copy()
    ext = m["detector"].str.extract(r"^(?P<base>.+)_s(?P<seed>\d+)$")
    m = m.join(ext).dropna(subset=["base"])
    cols = ["base", "strategy", "slice", "group", "metric", "n_seeds", "mean", "sd"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    g = m.groupby(["base", "strategy", "slice", "group", "metric"])["value"]
    out = pd.DataFrame({"n_seeds": g.size(), "mean": g.mean(), "sd": g.std(ddof=1)}).reset_index()
    return out[out["n_seeds"] > 1][cols]
