"""Dashboard data helpers on synthetic artifacts, plus a smoke run of every page against the real artifacts.

The helpers decide which numbers and thresholds the dashboard shows, so they are tested offline on small
frames written to a temporary artifacts directory. The page smoke test uses Streamlit's AppTest and only
needs the committed ``artifacts/`` (the optional text join is off by default, so ``data/`` is not touched).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dashboard import data


def _metric_rows(det: str, strat: str, sl: str, groups: dict[str, float], thr: float = 0.4, src: str = "phantom_val",
                 metric: str = "auroc") -> list[dict]:
    return [{"detector": det, "strategy": strat, "slice": sl, "group": g, "n": 100, "n_pairs": 50, "threshold": thr,
             "threshold_source": src, "metric": metric, "value": v, "ci_lo": v - 0.05, "ci_hi": v + 0.05}
            for g, v in groups.items()]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = []
    rows += _metric_rows("nli", "truncate-w400", "phantom_grid_sub",
                         {"all": 0.70, "hard": 0.62, "len=500": 0.80, "len=2000": 0.72, "len=500|hard": 0.7,
                          "len=2000|hard": 0.6, "pos=end": 0.66, "pos=beginning": 0.74,
                          "cell=10k|500|seed": 0.81, "cell=10k|2000|beginning": 0.75, "cell=10k|2000|end": 0.69})
    rows += _metric_rows("nli", "truncate-w400", "phantom_grid_sub", {"all": 0.65}, metric="f1")
    rows += _metric_rows("nli", "truncate-w400__ablate", "phantom_seed_test", {"all": 0.55}, thr=0.9)
    rows += _metric_rows("answer_tfidf", "none", "phantom_grid_full", {"all": 0.95, "len=500": 0.97, "len=2000": 0.94, "pos=end": 0.93})
    rows += _metric_rows("answer_length", "none", "phantom_grid_full", {"all": 0.83})
    rows += _metric_rows("answer_tfidf", "none", "halueval_test", {"all": 0.97}, src="halueval_val")
    rows += _metric_rows("nli", "truncate-w400", "halueval_test", {"all": 0.80}, thr=0.3, src="halueval_val")
    rows += _metric_rows("nli", "truncate-w400", "ragtruth_test", {"all": 0.60}, thr=0.5, src="default")
    mdir = tmp_path / "metrics"
    mdir.mkdir()
    pd.DataFrame(rows).to_csv(mdir / "metrics_long.csv", index=False)
    pd.DataFrame(columns=data.SEED_COLUMNS).to_csv(mdir / "seed_summary.csv", index=False)

    pdir = tmp_path / "predictions"
    (pdir / "manifest").mkdir(parents=True)
    pred = pd.DataFrame({"example_id": [f"e{i}" for i in range(6)], "pair_id": ["p0", "p0", "p1", "p1", "p2", "p2"],
                         "label": [0, 1, 0, 1, 0, 1], "p_halluc": [0.9, 0.8, 0.1, 0.2, 0.5, 0.95], "part": "x"})
    for rid in ("nli__truncate-w400__halueval_test", "nli__truncate-w400__halueval_val",
                "nli__truncate-w16000__phantom_seed_test__ablate", "answer_tfidf__none__halueval_test"):
        pred.to_parquet(pdir / f"{rid}.parquet")
    (pdir / "manifest" / "nli__truncate-w400__halueval_test.json").write_text(json.dumps({"run": "nli", "rows": 6}))
    (pdir / "manifest.json").write_text(json.dumps({"nli__truncate-w400__halueval_val": {"run": "nli", "legacy": True}}))
    monkeypatch.setattr(data, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


def test_loaders_use_patched_root(root: Path) -> None:
    long = data.load_metrics_long()
    assert set(long["dataset"]) == {"phantom", "halueval", "ragtruth"}
    assert data.load_seed_summary().empty
    assert data.load_comparisons().empty  # missing file -> empty frame, not an exception
    status = data.artifact_status()
    assert status["metrics/comparisons.csv"] == "missing"
    assert status["predictions/*.parquet"] == "4 files"


def test_parse_run_id_keeps_ablate_on_strategy() -> None:
    assert data.parse_run_id("jev_noul_ablate__truncate-w16000__halueval_test__ablate") == {
        "detector": "jev_noul_ablate", "strategy": "truncate-w16000__ablate", "slice": "halueval_test"}
    assert data.parse_run_id("deberta_nli__chunk-w400-o128-min_phalluc__phantom_grid_sub")["slice"] == "phantom_grid_sub"
    assert data.dataset_of("halueval_norm_test") == "halueval"


def test_list_runs_and_manifests(root: Path) -> None:
    runs = data.list_runs()
    assert len(runs) == 4
    ab = runs[runs["run_id"].str.endswith("__ablate")].iloc[0]
    assert (ab["strategy"], ab["slice"]) == ("truncate-w16000__ablate", "phantom_seed_test")
    man = data.load_manifests()
    assert man["nli__truncate-w400__halueval_val"]["legacy"] is True  # legacy manifest.json is read
    assert man["nli__truncate-w400__halueval_test"]["rows"] == 6
    assert runs.set_index("run_id").loc["answer_tfidf__none__halueval_test", "has_manifest"] == False  # noqa: E712
    assert len(data.load_predictions("nli__truncate-w400__halueval_test")) == 6


def test_val_threshold_lookup(root: Path) -> None:
    long = data.load_metrics_long()
    assert data.val_threshold(long, "nli", "truncate-w400", "halueval_test") == (0.3, "halueval_val")
    assert data.val_threshold(long, "nli", "truncate-w400", "ragtruth_test") == (0.5, "default")
    # a validation slice has no metrics rows but is where the test threshold was fitted
    assert data.val_threshold(long, "nli", "truncate-w400", "halueval_val") == (0.3, "halueval_val")
    assert data.val_threshold(long, "nli", "unknown", "halueval_test") == (None, "")
    assert data.val_threshold(long.iloc[0:0], "nli", "truncate-w400", "halueval_test") == (None, "")


def test_baseline_reference_falls_back_to_grid_full(root: Path) -> None:
    long = data.load_metrics_long()
    base, src = data.baseline_reference(long, "phantom_grid_sub", "all", "auroc")
    assert src == "phantom_grid_full"
    assert set(base["detector"]) == {"answer_tfidf", "answer_length"}
    base, src = data.baseline_reference(long, "halueval_test", "all", "auroc")
    assert src == "halueval_test" and list(base["detector"]) == ["answer_tfidf"]
    base, src = data.baseline_reference(long, "ragtruth_test", "all", "auroc")
    assert src is None and base.empty
    base, src = data.baseline_reference(long, "phantom_grid_sub", "hard", "auroc")  # group absent on grid_full
    assert src is None


def test_results_frame_and_headline(root: Path) -> None:
    long = data.load_metrics_long()
    df = data.results_frame(long, "phantom_grid_sub", "all", "auroc", ["nli"], ["truncate-w400"])
    assert len(df) == 1 and df["label"].iloc[0] == "nli · truncate-w400"
    head = data.headline_table(long)
    assert head["type"].iloc[0] == "answer-only baseline"  # baselines listed first
    assert "phantom_grid_sub (hard)" in head.columns
    cell = head.set_index(["detector", "strategy"]).loc[("nli", "truncate-w400"), "phantom_grid_sub"]
    assert cell == "0.700 [0.650, 0.750]"
    assert data.fmt_ci(0.5, float("nan"), 0.6) == "0.500"


def test_length_position_and_cell_curves(root: Path) -> None:
    long = data.load_metrics_long()
    curve = data.length_curve(long, "phantom_grid_sub", "auroc")
    assert list(curve["target_tokens"]) == [500, 2000]
    hard = data.length_curve(long, "phantom_grid_sub", "auroc", hard=True)
    assert list(hard["value"]) == [0.7, 0.6]
    base, src = data.baseline_curve(long, "phantom_grid_sub", "auroc", "length")
    assert src == "phantom_grid_full" and list(base["target_tokens"]) == [500, 2000]
    pos = data.position_frame(long, "phantom_grid_sub", "auroc")
    assert [str(p) for p in pos["position"]] == ["beginning", "end"]
    assert data.filings_with_cells(long, "phantom_grid_sub") == ["10k"]
    cells = data.cell_curve(long, "phantom_grid_sub", "auroc", "10k")
    # seed cell repeated in each position facet
    assert sorted(cells.loc[cells["target_tokens"] == 500, "position"]) == ["beginning", "end"]
    assert sorted(data.group_sort_key(g) for g in ["len=2000", "all", "len=500"])[0] == (0, 0, "")


def test_threshold_summary_confusion_and_errors() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.5, 0.95])
    s = data.threshold_summary(y, p, 0.5)  # p >= 0.5 -> hallucinated, as in evaluate
    assert (s["tp"], s["tn"], s["fp"], s["fn"]) == (2, 1, 2, 1)
    assert s["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert data.confusion(y, p, 0.5).tolist() == [[1, 2], [1, 2]]
    preds = pd.DataFrame({"example_id": list("abcdef"), "label": y, "p_halluc": p})
    fp, fn = data.top_errors(preds, 0.5, n=10)
    assert list(fp["example_id"]) == ["a", "e"]  # highest-scoring faithful first
    assert list(fn["example_id"]) == ["d"]


def test_attach_text_degrades_without_data() -> None:
    rows = pd.DataFrame({"example_id": ["x"], "part": ["nope"]})
    assert data.attach_text(rows, "no_such_slice") is None


# --------------------------------------------------------------------------- page smoke test (real artifacts)

REAL_METRICS = Path(__file__).resolve().parents[1] / "artifacts" / "metrics" / "metrics_long.csv"


@pytest.mark.skipif(not REAL_METRICS.exists(), reason="committed artifacts not present")
def test_every_page_renders_on_real_artifacts() -> None:
    from streamlit.testing.v1 import AppTest

    from src.dashboard.config import PAGES

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)
        assert not at.error, (page, [e.value for e in at.error])
