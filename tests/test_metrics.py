"""Metric implementations agree with scikit-learn; bootstrap and McNemar behave sensibly."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.eval.metrics import all_metrics, auprc, auroc, cluster_bootstrap, ece, mcnemar, paired_bootstrap_diff
from src.eval.threshold import apply_platt, choose_threshold, fit_platt


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 600), 0, 1).round(2)  # rounding creates ties
    return y, p


def test_auroc_auprc_match_sklearn(data):
    y, p = data
    assert auroc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-9)
    assert auprc(y, p) == pytest.approx(average_precision_score(y, p), abs=0.01)


def test_threshold_metrics_match_sklearn(data):
    y, p = data
    m = all_metrics(y, p, 0.5)
    assert m["f1"] == pytest.approx(f1_score(y, (p >= 0.5).astype(int)))
    assert 0 <= m["ece"] <= 1


def test_bootstrap_ci_contains_point_and_is_reasonable(data):
    y, p = data
    groups = np.arange(len(y)) // 2
    ci = cluster_bootstrap(y, p, groups, 0.5, ["auroc"], n_boot=300)
    lo, hi = ci["auroc"]
    assert lo < auroc(y, p) < hi
    assert 0.02 < hi - lo < 0.2


def test_paired_diff_of_identical_scores_is_zero(data):
    y, p = data
    d = paired_bootstrap_diff(y, p, p, np.arange(len(y)), n_boot=100)
    assert d["diff"] == 0 and d["ci_lo"] == 0 and d["ci_hi"] == 0


def test_mcnemar():
    y = np.array([1] * 20 + [0] * 20)
    a = y.copy()
    b = y.copy()
    b[:10] = 0  # b wrong on 10 rows where a is right
    r = mcnemar(y, a, b)
    assert r["a_only_correct"] == 10 and r["b_only_correct"] == 0
    assert r["p_value"] < 0.01
    assert mcnemar(y, a, a)["p_value"] == 1.0


def test_threshold_and_platt(data):
    y, p = data
    t = choose_threshold(y, p)
    assert 0 < t < 1
    a, b = fit_platt(y, p)
    assert ece(y, apply_platt(p, a, b)) <= ece(y, p) + 1e-6


def test_cluster_sampling_keeps_groups_whole():
    from src.eval.metrics import _Clusters

    groups = np.array([5, 5, 1, 9, 9, 9, 1])
    cl = _Clusters(groups)
    idx = cl.sample(np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < len(groups)
    picked = groups[idx]
    for g in np.unique(picked):
        size = (groups == g).sum()
        assert (picked == g).sum() % size == 0  # every sampled group appears whole


def test_seed_summary_pools_seed_runs():
    import pandas as pd

    from src.eval.evaluate import seed_summary

    rows = [{"detector": f"mb_ft_s{s}", "strategy": "chunk", "slice": "phantom_grid_sub", "group": "all", "metric": "auroc", "value": v}
            for s, v in ((42, 0.80), (43, 0.84), (44, 0.82))]
    rows.append({"detector": "hhem", "strategy": "chunk", "slice": "phantom_grid_sub", "group": "all", "metric": "auroc", "value": 0.7})
    out = seed_summary(pd.DataFrame(rows))
    assert len(out) == 1 and out.iloc[0]["n_seeds"] == 3
    assert out.iloc[0]["mean"] == pytest.approx(0.82) and out.iloc[0]["sd"] == pytest.approx(0.02)
