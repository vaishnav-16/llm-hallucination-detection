"""The stack stage fits on validation slices only, predicts out-of-fold there, and applies one model elsewhere."""

import json

import numpy as np
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
