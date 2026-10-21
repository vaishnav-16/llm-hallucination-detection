"""The stack stage fits on validation slices only, predicts out-of-fold there, and applies one model elsewhere."""

import json

import numpy as np
import pytest
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.eval.stack import run_stack

STRAT = "retrieve-w1500-o750-min_phalluc-k2-bidf"
MEMBERS = ["m_good", "m_noise"]


def _write(pred_dir, sl, dataset, n_pairs, seed, flip=False):
    rng = np.random.default_rng(seed)
    label = np.tile([0, 1], n_pairs)
    base = pd.DataFrame({"example_id": [f"{sl}-{i}" for i in range(2 * n_pairs)], "pair_id": np.repeat(np.arange(n_pairs), 2).astype(str),
                         "dataset": dataset, "label": label})
    good = np.clip(0.3 + 0.4 * label + rng.normal(0, 0.15, len(label)), 0.01, 0.99)
    if flip:  # a test slice whose labels the stack must never see: flipping them must not change its scores
        base["label"] = 1 - label
    for m, p in [("m_good", good), ("m_noise", rng.uniform(0.01, 0.99, len(label)))]:
        d = base.assign(p_halluc=p, detector=m, strategy=STRAT, run_id=f"{m}__{STRAT}__{sl}")
        d.to_parquet(pred_dir / f"{m}__{STRAT}__{sl}.parquet")


def test_stack_fits_on_validation_and_ignores_test_labels(tmp_path):
    (tmp_path / "manifest").mkdir()
    _write(tmp_path, "a_val", "a", 80, 1)
    _write(tmp_path, "b_val", "b", 40, 2)
    _write(tmp_path, "a_test", "a", 60, 3)
    spec = {"name": "stk", "strategy": STRAT, "members": MEMBERS, "fit_slices": ["a_val", "b_val"]}
    run_stack(spec, pred_dir=tmp_path)
    test = pd.read_parquet(tmp_path / f"stk__{STRAT}__a_test.parquet")
    assert roc_auc_score(test["label"], test["p_halluc"]) > 0.9
    man = json.loads((tmp_path / "manifest" / f"stk__{STRAT}__a_test.json").read_text())
    coefs = man["detector"]["coefficients_standardized"]
    assert abs(coefs["m_good"]) > 3 * abs(coefs["m_noise"])
    assert json.loads((tmp_path / "manifest" / f"stk__{STRAT}__a_val.json").read_text())["detector"]["out_of_fold"]

    # Same features, flipped test labels: identical scores, because test labels are never read.
    _write(tmp_path, "a_test", "a", 60, 3, flip=True)
    run_stack(spec, pred_dir=tmp_path)
    again = pd.read_parquet(tmp_path / f"stk__{STRAT}__a_test.parquet")
    np.testing.assert_allclose(again["p_halluc"], test["p_halluc"])


def _write_member(pred_dir, det, strategy, sl, dataset, ids, label, p):
    stem = f"{det}__{strategy[:-len('__ablate')]}__{sl}__ablate" if strategy.endswith("__ablate") else f"{det}__{strategy}__{sl}"
    pd.DataFrame({"example_id": ids, "pair_id": [i // 2 for i in range(len(ids))], "dataset": dataset, "label": label,
                  "p_halluc": p, "detector": det, "strategy": strategy, "run_id": stem}).to_parquet(pred_dir / f"{stem}.parquet")


def test_stack_mixed_members_aliases_and_row_coverage(tmp_path):
    (tmp_path / "manifest").mkdir()
    rng = np.random.default_rng(0)
    ctl_strat = "truncate-w7168__ablate"
    for sl, ds, n, ev_det in (("a_val", "a", 200, "ev"), ("a_test", "a", 200, "ev_full"), ("b_test", "b", 200, "ev")):
        label = np.tile([0, 1], n // 2)
        ids = [f"{sl}-{i}" for i in range(n)]
        # The evidence member sees half the signal, the control the other half; the control also covers 50 extra rows.
        sig = rng.normal(0, 1, n)
        _write_member(tmp_path, ev_det, STRAT, sl, ds, ids, label, 1 / (1 + np.exp(-(2 * label - 1 + sig))))
        extra = [f"{sl}-x{i}" for i in range(50)]
        _write_member(tmp_path, "ctl", ctl_strat, sl, ds, ids + extra, np.r_[label, np.tile([0, 1], 25)],
                      1 / (1 + np.exp(-(2 * np.r_[label, np.tile([0, 1], 25)] - 1 + rng.normal(0, 1, n + 50)))))
    spec = {"name": "mix", "strategy": "stack", "fit_slices": ["a_val"], "apply_slices": ["a_test"],
            "members": [{"detector": "ev", "strategy": STRAT, "aliases": {"a_test": "ev_full"}},
                        {"detector": "ctl", "strategy": ctl_strat}]}
    run_stack(spec, pred_dir=tmp_path)
    test = pd.read_parquet(tmp_path / "mix__stack__a_test.parquet")
    assert len(test) == 200  # the first member's rows, through the alias; the control's extra rows are dropped
    ev = pd.read_parquet(tmp_path / f"ev_full__{STRAT}__a_test.parquet")
    assert roc_auc_score(test["label"], test["p_halluc"]) > roc_auc_score(ev["label"], ev["p_halluc"])
    assert not (tmp_path / "mix__stack__b_test.parquet").exists()  # apply_slices keeps the model on its own dataset

    # A control that misses some of the first member's rows must raise, not shrink the slice.
    d = pd.read_parquet(tmp_path / "ctl__truncate-w7168__a_test__ablate.parquet")
    d.iloc[10:].to_parquet(tmp_path / "ctl__truncate-w7168__a_test__ablate.parquet")
    with pytest.raises(ValueError, match="covers 190 of the 200 rows"):
        run_stack(spec, pred_dir=tmp_path)
