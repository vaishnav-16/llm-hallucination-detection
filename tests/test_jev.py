"""Offline tests for the Jev adapter: request shape, grouping, score direction, cache, budget guard, retries."""

import json

import numpy as np
import pytest

import src.detectors.jev as jev
from src.detectors.base import ScoringError


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "CACHE_PATH", tmp_path / "jev.jsonl")
    monkeypatch.setattr(jev, "SPEND_PATH", tmp_path / "jev_spend.json")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


class FakeAPI:
    """Answers 'supported' when the response text appears in the document, otherwise 'unsupported'."""

    def __init__(self, fail_first: list[int] | None = None):
        self.calls: list[dict] = []
        self.fail_first = list(fail_first or [])

    def __call__(self, path, body):
        if self.fail_first:
            return self.fail_first.pop(0), {"detail": "busy"}, {"retry-after": "0"}
        self.calls.append(body)
        doc = body["state"]["document"]
        answers = {}
        for qid, q in body["questions"].items():
            supported = q["instructions"]["response"] in doc
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.9 if supported else 0.2}
            else:
                p = 0.8 if supported else 0.1
                answers[qid] = {"type": "choice", "choice": "supported" if supported else "unsupported",
                                "probabilities": {"supported": p, "contradicted": 0.05, "unsupported": 0.95 - p}, "confidence": 0.9}
        return 200, {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1000, "output_tokens": 10}}, {}


def _judge(answer="noul", api=None, **kw):
    j = jev.JevJudge(answer, **kw)
    j.client._post = api or FakeAPI()
    return j


def test_request_shape_and_grouping():
    api = FakeAPI()
    j = _judge(api=api)
    doc = "The payout ratio was 41%."
    claims = [jev.JevJudge.pack_question("What was the payout ratio?", r, "QA") for r in ("41%", "73% due to Asia")]
    p = j.score_pairs([doc, doc], claims)
    assert len(api.calls) == 1  # both answers of the pair share one request
    body = api.calls[0]
    assert body["model"] == "jev-1.13.0" and body["state"] == {"document": doc}
    assert set(body["questions"]) == {"c0_noul", "c0_choice", "c1_noul", "c1_choice"}
    q = body["questions"]["c1_noul"]
    assert q["type"] == "noul" and q["instructions"]["question_asked"] == "What was the payout ratio?"
    assert q["instructions"]["response"] == "73% due to Asia" and set(q["criteria"]) == {"true", "false"}
    assert list(body["questions"]["c0_choice"]["criteria"]) == ["supported", "contradicted", "unsupported"]
    # direction: supported -> low p_halluc
    assert p[0] == pytest.approx(0.1) and p[1] == pytest.approx(0.8)


def test_choice_direction_and_shared_cache():
    api = FakeAPI()
    doc = "Revenue was 5 million."
    claims = [jev.JevJudge.pack_question("Revenue?", "5 million", "QA"), jev.JevJudge.pack_question("Revenue?", "9 million", "QA")]
    noul = _judge("noul", api=api).score_pairs([doc, doc], claims)
    choice = _judge("choice", api=api).score_pairs([doc, doc], claims)  # new judge, same cache file
    assert len(api.calls) == 1, "jev_choice must be served from the cache written by jev_noul"
    assert choice[0] == pytest.approx(0.2) and choice[1] == pytest.approx(0.9)
    assert noul[0] < noul[1]


def test_summary_tasks_have_no_question():
    api = FakeAPI()
    _judge(api=api).score_pairs(["doc text"], [jev.JevJudge.pack_question("Summarize:", "doc text", "Summary")])
    ins = api.calls[0]["questions"]["c0_noul"]["instructions"]
    assert ins["question_asked"].startswith("(none") and "summary" in ins["question"]


def test_groups_split_by_premise_and_size():
    j = _judge(max_claims_per_request=2)
    groups = j._groups(["a", "a", "a", "b", "a"], ["x", "y", "z", "x", "x"])
    assert [(p, [c for c, _ in g]) for p, g in groups] == [("a", ["x", "y"]), ("a", ["z"]), ("b", ["x"])]
    assert groups[0][1][0][1] == [0, 4]  # duplicate (premise, claim) pairs are scored once


def test_budget_guard_and_spend_ledger(monkeypatch):
    api = FakeAPI()
    j = _judge(api=api, max_usd=1e-9)
    j.score_pairs(["doc one"], ["q\x1eone"])  # first live call is allowed, then the ledger exceeds the budget
    spend = json.loads(jev.SPEND_PATH.read_text())
    assert spend["input_tokens"] == 1000 and spend["requests"] == 1
    with pytest.raises(ScoringError, match="budget"):
        j.score_pairs(["doc two"], ["q\x1etwo"])
    # cached requests are still served after the budget is reached
    assert j.score_pairs(["doc one"], ["q\x1eone"]).shape == (1,)


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)
    api = FakeAPI(fail_first=[429, 529])
    p = _judge(api=api).score_pairs(["doc"], ["q\x1edoc"])
    assert np.isfinite(p).all() and len(api.calls) == 1


def test_client_errors_raise():
    j = _judge(api=lambda path, body: (401, {"detail": "bad key"}, {}))
    with pytest.raises(ScoringError, match="401"):
        j.score_pairs(["doc"], ["q\x1ea"])


@pytest.mark.parametrize("status,payload", [
    (422, {"detail": "input exceeds the 32k token context length"}),
    (400, {"detail": {"error_type": "max_tokens_exceeded"}}),  # what the live API returns
])
def test_too_long_request_shrinks_document(status, payload):
    calls = []

    def api(path, body):
        calls.append(len(body["state"]["document"]))
        if len(calls) == 1:
            return status, payload, {}
        return FakeAPI()(path, body)

    j = _judge(api=api)
    j.score_pairs(["x" * 1000], ["q\x1ex"])
    assert calls[1] == 750 and j.n_shrunk == 1


def test_other_400_still_raises():
    j = _judge(api=lambda path, body: (400, {"detail": {"error_type": "invalid_question"}}, {}))
    with pytest.raises(ScoringError, match="400"):
        j.score_pairs(["doc"], ["q\x1ea"])


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev, "ROOT", jev.CACHE_PATH.parent)  # no .env there
    j = jev.JevJudge("noul")
    with pytest.raises(ScoringError, match="TYPESAFE_API_KEY"):
        j.score_pairs(["doc"], ["q\x1ea"])


def test_piece_spans_and_estimate():
    text = "abcdefghij  kl"
    spans = jev.piece_spans(text)
    assert [text[a:b] for a, b in spans] == ["abcd", "efgh", "ij", "kl"]
    j = _judge()
    est = j.estimate(["doc"] * 2, ["q\x1ea", "q\x1eb"])
    assert est["requests"] == 1 and est["uncached_requests"] == 1 and est["approx_usd"] > 0


def test_runner_integration_with_chunking():
    import pandas as pd

    from src.verify.chunking import ChunkSpec
    from src.verify.runner import score_frame
    from src.verify.slices import META_COLUMNS

    filler = "Unrelated filing text about leases and segments. " * 400
    rows = []
    for k, (resp, lab) in enumerate((("payout ratio was 41%", 0), ("payout ratio was 73%", 1))):
        rows.append({"example_id": f"e{k}", "pair_id": "p0", "dataset": "phantom", "variant": "original", "filing": "10k",
                     "target_tokens": 2000, "position": "end", "task_type": "QA", "split": "test", "label": lab,
                     "faithful_chars": 20, "question": "What was the payout ratio?", "response": resp,
                     "context": filler + " The payout ratio was 41% this year."})
    df = pd.DataFrame(rows)
    api = FakeAPI()
    j = _judge(api=api)
    out = score_frame(df, j, ChunkSpec(strategy="chunk", window=2000, overlap=200))
    assert list(out.columns[: len(META_COLUMNS)]) == META_COLUMNS
    assert out["n_chunks"].min() > 1  # long context was chunked
    assert len(api.calls) == out["n_chunks"].iloc[0]  # one request per window, both answers share it
    # the evidence sits in the last window; min aggregation finds it for the faithful answer only
    assert out.loc[0, "p_halluc"] < 0.5 < out.loc[1, "p_halluc"]
    trunc = score_frame(df, j, ChunkSpec(strategy="truncate", window=2000))
    assert len(api.calls) == out["n_chunks"].iloc[0], "truncate must be served from the chunk cache"
    assert (trunc["p_halluc"] > 0.5).all()  # truncation cuts the evidence off: both look unsupported
