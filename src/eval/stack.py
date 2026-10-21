"""`stack` stage: combine several answers of one detector into one score, fitted on validation slices only.

TypeSafe's guidance for Jev is to ask one snap judgment per question and combine the answers in code. The v2 Jev runs
(``questions: v2``) give several per-claim answers (support, numbers, entities, extra, choice2) from the same requests.
This stage fits a logistic regression on their logits:

- fitted on the pooled validation slices (PHANTOM 8-K/497K val, RAGTruth val, HaluEval val) only, each dataset
  weighted equally, so no test label is used;
- validation predictions are out-of-fold (5-fold, grouped by ``pair_id``), so the decision threshold that
  ``evaluate`` picks on validation is not chosen on rows the model was fitted on;
- the fitted model is applied unchanged to every other slice the members were scored on.

A stack is declared in a config's ``stacks:`` list::

    stacks:
      - name: jev_v2_stack
        strategy: retrieve-w1500-o750-min_phalluc-k2-bidf   # the members' strategy tag
        members: [jev_v2_support, jev_v2_numbers, jev_v2_entities, jev_v2_extra, jev_v2_choice2]
        fit_slices: [phantom_val, ragtruth_val, halueval_val]

Members may also come from different detectors, each with its own strategy (``{detector, strategy, aliases}``), e.g.
Jev plus the fine-tuned answer-only control. ``aliases`` maps a slice to another run of the same detector config that
covers more rows there (``jev_noul_full`` on ``phantom_grid_sub``). The first member defines the rows; every other
member must cover all of them, or the stage raises. ``apply_slices`` limits where a fitted model is applied, so a
model fitted on one dataset's validation slice is never applied to another dataset. In that form ``strategy`` is the
output tag only.

Outputs go next to the members' predictions as ``<name>__<strategy>__<slice>.parquet``, with a manifest record that
holds the coefficients and the members' config hashes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, SEED
from src.data.store import read_json, read_table, write_json, write_table

logger = logging.getLogger(__name__)

PRED_DIR = ARTIFACTS_DIR / "predictions"
EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _members(spec: dict) -> list[dict]:
    """Normalise members to {name, detector, strategy, aliases}; a plain string uses the stack's own strategy."""
    out = []
    for m in spec["members"]:
        m = {"detector": m} if isinstance(m, str) else dict(m)
        out.append({"name": m["detector"], "detector": m["detector"], "strategy": m.get("strategy", spec["strategy"]),
                    "aliases": dict(m.get("aliases") or {})})
    return out


def _run_stem(det: str, strategy: str, sl: str) -> str:
    """Prediction file stem; ablated runs put ``__ablate`` after the slice (``<det>__<strategy>__<slice>__ablate``)."""
    if strategy.endswith("__ablate"):
        return f"{det}__{strategy[:-len('__ablate')]}__{sl}__ablate"
    return f"{det}__{strategy}__{sl}"


def _member_frame(members: list[dict], sl: str, pred_dir: Path) -> pd.DataFrame | None:
    """Rows of one slice with one feature column per member, aligned on example_id. None if any member is missing.

    The first member defines the rows. Another member may cover more rows (it is restricted to these), but a row it
    lacks raises: members scored on different rows must never be silently intersected.
    """
    out = None
    for m in members:
        path = pred_dir / f"{_run_stem(m['aliases'].get(sl, m['detector']), m['strategy'], sl)}.parquet"
        if not path.exists():
            return None
        d = read_table(path)
        if out is None:
            out = d.drop(columns=["p_halluc", "detector", "strategy", "run_id"]).copy()
            out = out.merge(d[["example_id", "p_halluc"]].rename(columns={"p_halluc": m["name"]}), on="example_id", validate="1:1")
            continue
        n = len(out)
        out = out.merge(d[["example_id", "p_halluc"]].rename(columns={"p_halluc": m["name"]}), on="example_id", how="inner", validate="1:1")
        if len(out) != n:
            raise ValueError(f"{path.name}: covers {len(out)} of the {n} rows of the first member (members scored on different rows)")
    return out


def _slices(members: list[dict], pred_dir: Path) -> list[str]:
    m = members[0]
    prefix, suffix = _run_stem(m["detector"], m["strategy"], "{slice}").split("{slice}")
    names = {p.name[len(prefix):-len(suffix + ".parquet")] for p in pred_dir.glob(f"{prefix}*{suffix}.parquet")}
    return sorted(n for n in names if n and "__" not in n)


def _model(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=SEED))


def _weights(datasets: pd.Series) -> np.ndarray:
    counts = datasets.map(datasets.value_counts())
    return (len(datasets) / (datasets.nunique() * counts)).to_numpy(float)


def run_stack(spec: dict, pred_dir: Path = PRED_DIR, C: float = 1.0, n_folds: int = 5) -> list[Path]:
    name, strategy = spec["name"], spec["strategy"]
    mspec = _members(spec)
    members = [m["name"] for m in mspec]
    fit_slices = list(spec["fit_slices"])
    fit_parts = []
    for sl in fit_slices:
        f = _member_frame(mspec, sl, pred_dir)
        if f is None:
            raise FileNotFoundError(f"stack {name}: members missing predictions on fit slice {sl} ({strategy})")
        fit_parts.append(f.assign(_slice=sl))
    fit = pd.concat(fit_parts, ignore_index=True)
    X, y = _logit(fit[members].to_numpy(float)), fit["label"].to_numpy(int)
    w = _weights(fit["dataset"])

    oof = np.full(len(fit), np.nan)
    groups = fit["dataset"].astype(str) + ":" + fit["pair_id"].astype(str)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
        oof[te] = _model(C).fit(X[tr], y[tr], logisticregression__sample_weight=w[tr]).predict_proba(X[te])[:, 1]
    model = _model(C).fit(X, y, logisticregression__sample_weight=w)
    lr = model.named_steps["logisticregression"]
    coefs = dict(zip(members, np.round(lr.coef_[0], 4).tolist()))
    val_auroc = {sl: round(float(roc_auc_score(y[m], oof[m])), 4) for sl in fit_slices for m in [(fit["_slice"] == sl).to_numpy()]}
    member_auroc = {sl: {mem: round(float(roc_auc_score(y[m], fit.loc[m, mem])), 4) for mem in members}
                    for sl in fit_slices for m in [(fit["_slice"] == sl).to_numpy()]}
    logger.info("stack %s: standardized coefficients %s", name, coefs)
    logger.info("stack %s: out-of-fold val AUROC %s (members: %s)", name, val_auroc, member_auroc)

    man_dir = pred_dir / "manifest"
    member_hashes = {}
    for m in mspec:
        for det in sorted({m["detector"], *m["aliases"].values()}):
            prefix, suffix = _run_stem(det, m["strategy"], "{slice}").split("{slice}")
            for p in sorted(man_dir.glob(f"{prefix}*{suffix}.json")):
                member_hashes[p.stem] = read_json(p).get("hash")
    stack_hash = hashlib.sha1(json.dumps({"spec": spec, "C": C, "n_folds": n_folds, "members": member_hashes},
                                         sort_keys=True).encode()).hexdigest()[:12]
    written = []
    targets = list(spec["apply_slices"]) + fit_slices if spec.get("apply_slices") else _slices(mspec, pred_dir)
    for sl in sorted(set(targets)):
        if sl in fit_slices:
            f = fit[fit["_slice"] == sl].drop(columns="_slice")
            p = oof[(fit["_slice"] == sl).to_numpy()]
        else:
            f = _member_frame(mspec, sl, pred_dir)
            if f is None:
                logger.warning("stack %s: skipping %s (not every member scored it)", name, sl)
                continue
            p = model.predict_proba(_logit(f[members].to_numpy(float)))[:, 1]
        out = f.drop(columns=members).copy()
        out["p_halluc"] = p
        out["detector"], out["strategy"] = name, strategy
        out["run_id"] = f"{name}__{strategy}__{sl}"
        written.append(write_table(out, pred_dir / f"{out['run_id'].iloc[0]}.parquet"))
        write_json({
            "hash": stack_hash, "run": name, "slice": sl, "strategy": strategy, "ablate": False,
            "detector": {"name": name, "type": "stack", "members": spec["members"], "coefficients_standardized": coefs,
                         "intercept": round(float(lr.intercept_[0]), 4), "C": C, "fit_slices": fit_slices,
                         "out_of_fold_val_auroc": val_auroc, "member_val_auroc": member_auroc,
                         "fit_rows": int(len(fit)), "out_of_fold": sl in fit_slices},
            "member_hashes": member_hashes, "rows": int(len(out)),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
        }, man_dir / f"{out['run_id'].iloc[0]}.json")
    logger.info("stack %s: wrote %d prediction files", name, len(written))
    return written


def run_stacks(config_path: str | Path, pred_dir: Path = PRED_DIR) -> list[Path]:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    out: list[Path] = []
    for spec in cfg.get("stacks", []):
        out += run_stack(spec, pred_dir)
    if not cfg.get("stacks"):
        logger.warning("%s has no stacks: section", config_path)
    return out
