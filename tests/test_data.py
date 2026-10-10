"""Offline tests for schema validation, PHANTOM label parsing / long-to-seed join, and split rules."""

import pandas as pd
import pytest

from src.data import phantom, splits
from src.data.schema import COLUMNS, SchemaError, stable_id, validate


def _rows(dataset="halueval", n_pairs=10, filing="", position="", target=0, ctx="ctx"):
    rows = []
    for i in range(n_pairs):
        pid = stable_id(dataset, filing, i)
        for lab in (0, 1):
            rows.append({
                "example_id": stable_id(pid, lab, target, position), "pair_id": pid, "dataset": dataset, "variant": "original",
                "filing": filing, "target_tokens": target, "position": position, "task_type": "QA",
                "question": f"q{i}", "context": f"{ctx} {i}", "response": f"a{i}{lab}", "label": lab, "source_file": "x",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def test_validate_accepts_good_frame():
    assert len(validate(_rows(), "ok")) == 20


def test_validate_rejects_single_class():
    df = _rows()
    df["label"] = 0
    with pytest.raises(SchemaError, match="one class"):
        validate(df, "bad")


def test_validate_rejects_duplicates_and_empty_text():
    df = _rows()
    with pytest.raises(SchemaError, match="duplicate"):
        validate(pd.concat([df, df.iloc[:1]]), "dup")
    df.loc[0, "context"] = "  "
    with pytest.raises(SchemaError, match="empty"):
        validate(df, "empty")


def test_phantom_label_parsing_is_strict():
    s = pd.Series(["hallucination", "Not Hallucination ", "hallucination"])
    assert phantom.parse_labels(s, "t").tolist() == [1, 0, 1]
    with pytest.raises(SchemaError):
        phantom.parse_labels(pd.Series(["hallucinated", "0"]), "t")  # the strings the old loader expected


def test_phantom_file_names_are_case_exact():
    assert phantom.Cell("10k").filename == "Phantom_10k_seed.csv"
    assert phantom.Cell("def14a").filename == "Phantom_def14a_seed.csv"
    assert phantom.Cell("def14a", 2000, "middle").filename == "Phantom_def14A_2000tokens_middle.csv"
    with pytest.raises(ValueError):
        phantom.Cell("8k", 2000, "middle").filename
    assert len(phantom.all_cells()) == 4 + 2 * 15


def _raw_seed():
    return pd.DataFrame({
        "query": ["q1", "q1", "q2", "q2"],
        "context": ["passage one about revenue " * 20, "passage one about revenue " * 20, "passage two about risk " * 20, "passage two about risk " * 20],
        "answer": ["41%", "41% due to Asia growth.", "$5m", "$7m, rising."],
        "label": [0, 1, 0, 1],
    })


def test_long_join_original_reanswered_and_long_only(monkeypatch):
    raw_seed = _raw_seed()
    monkeypatch.setattr(phantom, "read_raw", lambda cell: raw_seed.copy())
    seed = phantom.load_seed("def14a")
    pid = dict(zip(seed["question"], seed["pair_id"]))

    long_raw = pd.DataFrame({
        "query": ["q1", "q1", "q2", "q2", "q3", "q3"],
        "context": ["filler " * 50 + raw_seed.context[0] + " tail"] * 2 + ["pre " * 40 + raw_seed.context[2]] * 2 + ["unrelated text " * 30] * 2,
        "answer": ["41%", "41% due to Asia growth.", "$5 million", "$9m, a different hallucination.", "x", "y"],
        "label": [0, 1, 0, 1, 0, 1],
    })
    monkeypatch.setattr(phantom, "read_raw", lambda cell: long_raw.copy())
    out = phantom.load_long(phantom.Cell("def14a", 2000, "middle"), seed, max_unjoined=0.5)
    by_q = out.groupby("question").agg(pair=("pair_id", "first"), variant=("variant", "first"))
    assert by_q.loc["q1", "pair"] == pid["q1"] and by_q.loc["q1", "variant"] == "original"
    assert by_q.loc["q2", "pair"] == pid["q2"] and by_q.loc["q2", "variant"] == "reanswered"
    assert by_q.loc["q3", "variant"] == "long_only" and by_q.loc["q3", "pair"] not in pid.values()
    with pytest.raises(SchemaError):
        phantom.load_long(phantom.Cell("def14a", 2000, "middle"), seed, max_unjoined=0.1)


def test_splits_are_leak_free():
    hal = _rows("halueval", 200)
    rt_train, rt_test = _rows("ragtruth", 60, ctx="train"), _rows("ragtruth", 20, ctx="test")
    rt_test["pair_id"] = [p + "t" for p in rt_test["pair_id"]]
    seeds = {f: _rows("phantom", 30, filing=f, position="seed", target=500) for f in phantom.FILINGS}
    cell_pairs = {f: [set(seeds[f]["pair_id"][:40]), set(seeds[f]["pair_id"][10:])] for f in phantom.TEST_FILINGS}
    doc = splits.build_splits(hal, rt_train, rt_test, seeds, cell_pairs, grid_sub_pairs=5)

    ph = doc["splits"]["phantom"]
    for f in phantom.TEST_FILINGS:
        assert all(ph[p] == "test" for p in seeds[f]["pair_id"])
    for f in phantom.TRAIN_FILINGS:
        assert all(ph[p] in ("train", "val") for p in seeds[f]["pair_id"])
    assert set(doc["splits"]["halueval"].values()) == {"train", "val", "test"}
    assert all(doc["splits"]["ragtruth"][p] == "test" for p in rt_test["pair_id"])
    for f in phantom.TEST_FILINGS:
        common = set(seeds[f]["pair_id"]).intersection(*cell_pairs[f])
        assert len(doc["grid_sub"][f]) == 5 and set(doc["grid_sub"][f]) <= common
    # every row of a pair gets the same split
    hal["split"] = splits.assign(hal, doc)
    assert (hal.groupby("pair_id")["split"].nunique() == 1).all()
    # deterministic
    assert splits.build_splits(hal, rt_train, rt_test, seeds, cell_pairs, grid_sub_pairs=5)["splits"] == doc["splits"]
