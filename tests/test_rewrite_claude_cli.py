"""Offline tests for the HaluEval rewrite through headless ``claude -p`` (subprocess stubbed)."""

import json
import re
import subprocess
import threading

import pandas as pd
import pytest

from src.data import claude_cli, normalize_answers
from src.data.claude_cli import ClaudeCliClient, ClaudeCliError
from src.data.normalize_answers import PROMPT, STRICT_SUFFIX

MODEL = "claude-haiku-test"
VERSION = "9.9.9 (Claude Code)"
REC_KEYS = {"example_id", "pair_id", "rewrite", "ok", "provider", "model", "cli_version"}


def _df(n_pairs=3):
    rows = []
    for i in range(n_pairs):
        rows.append({"example_id": f"e{i}_0", "pair_id": f"p{i}", "question": f"Who built bridge {i}?",
                     "context": "ctx", "response": f"Builder{i}", "label": 0})
        rows.append({"example_id": f"e{i}_1", "pair_id": f"p{i}", "question": f"Who built bridge {i}?",
                     "context": "ctx", "response": f"Bridge {i} was built by Impostor{i} in 1900.", "label": 1})
    return pd.DataFrame(rows)


def _result(text, model=MODEL, **kw):
    d = {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None,
         "result": text, "modelUsage": {model: {}}, **kw}
    return json.dumps(d)


def _answer(prompt):
    return re.search(r"^Answer: (.*)$", prompt, re.M).group(1)


class FakeRun:
    """Stands in for subprocess.run; ``reply(prompt) -> CompletedProcess | str`` decides each answer."""

    def __init__(self, reply=None):
        self.calls = []
        self.lock = threading.Lock()
        self.reply = reply or (lambda prompt: _result(f"The answer is {_answer(prompt)}."))

    def __call__(self, cmd, input=None, **kw):
        if cmd == ["claude", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, VERSION + "\n", "")
        with self.lock:
            self.calls.append({"cmd": cmd, "input": input, **kw})
        out = self.reply(input)
        if isinstance(out, BaseException):
            raise out
        return out if isinstance(out, subprocess.CompletedProcess) else subprocess.CompletedProcess(cmd, 0, out, "")


@pytest.fixture
def env(tmp_path, monkeypatch):
    df = _df()
    monkeypatch.setattr("src.data.halueval.load_halueval", lambda variant="original": df)
    monkeypatch.setattr(normalize_answers, "_select_pairs", lambda d, n: set(d["pair_id"]))
    monkeypatch.setattr("src.detectors.api_judge.load_providers",
                        lambda: {"claude_cli": {"model": MODEL, "timeout_s": 7, "max_retries": 2, "backoff_s": 1}})
    sleeps = []
    monkeypatch.setattr(claude_cli.time, "sleep", sleeps.append)
    fake = FakeRun()
    monkeypatch.setattr(claude_cli.subprocess, "run", fake)
    return {"df": df, "fake": fake, "out": tmp_path / "rw.jsonl", "sleeps": sleeps}


def _run(env, **kw):
    normalize_answers.rewrite_halueval(provider="claude_cli", out_path=env["out"], **kw)
    return [json.loads(l) for l in env["out"].read_text(encoding="utf-8").splitlines()]


def test_command_disables_tools_mcp_and_sessions():
    cmd = claude_cli.build_command(MODEL)
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--model") + 1] == MODEL
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--system-prompt") + 1] == claude_cli.SYSTEM_PROMPT
    assert "--bare" not in cmd


def test_one_answer_per_process_with_identical_prompt(env):
    recs = _run(env, workers=3)
    calls = env["fake"].calls
    assert len(calls) == len(env["df"]) == len(recs)
    for c in calls:
        assert c["cmd"] == claude_cli.build_command(MODEL)
        assert c["input"].count("Answer:") == 1 and c["input"].count("Question:") == 1
        assert c["timeout"] == 7
        assert not any(p.name == "CLAUDE.md" for p in __import__("pathlib").Path(c["cwd"]).iterdir())
    by_answer = {_answer(c["input"]) for c in calls}
    assert by_answer == set(env["df"]["response"])
    # The prompt is the shared PROMPT with only question/answer filled in, whatever the label.
    r = env["df"].iloc[1]
    assert PROMPT.format(question=r.question, answer=r.response) in [c["input"] for c in calls]


def test_calls_are_issued_in_the_seeded_shuffled_order(env):
    _run(env, workers=1)
    order = [_answer(c["input"]) for c in env["fake"].calls]
    expected = env["df"].sample(frac=1.0, random_state=normalize_answers.SEED)["response"].tolist()
    assert order == expected


def test_record_format(env):
    recs = _run(env)
    for rec in recs:
        assert set(rec) == REC_KEYS
        assert rec["provider"] == "claude_cli" and rec["model"] == MODEL and rec["cli_version"] == VERSION
        assert rec["ok"] is True and rec["rewrite"].startswith("The answer is ")
    assert env["out"].with_suffix(".audit.md").exists()


def test_failed_meaning_check_retries_once_with_strict_suffix(env):
    def reply(prompt):
        ans = _answer(prompt)
        if ans == "Builder0" and not prompt.endswith(STRICT_SUFFIX):
            return _result("Somebody else did it.")
        if ans == "Builder1":
            return _result("No idea.")  # fails twice
        return _result(f"The answer is {ans}.")

    env["fake"].reply = reply
    recs = {r["example_id"]: r for r in _run(env)}
    p0 = [c["input"] for c in env["fake"].calls if _answer(c["input"]) == "Builder0"]
    assert len(p0) == 2 and p0[1] == p0[0] + STRICT_SUFFIX
    assert recs["e0_0"]["ok"] is True and recs["e0_0"]["rewrite"] == "The answer is Builder0."
    assert recs["e1_0"]["ok"] is False and len([c for c in env["fake"].calls if _answer(c["input"]) == "Builder1"]) == 2
    others = [c for c in env["fake"].calls if _answer(c["input"]) not in ("Builder0", "Builder1")]
    assert len(others) == 4  # answers that pass first time are called once


def test_failed_pair_is_dropped(env, monkeypatch):
    env["fake"].reply = lambda p: _result("No idea.") if _answer(p) == "Builder1" else _result(f"The answer is {_answer(p)}.")
    _run(env)
    monkeypatch.setattr(normalize_answers, "REWRITES_PATH", env["out"])
    out = normalize_answers.apply_rewrites(env["df"].assign(variant="original"))
    assert set(out["pair_id"]) == {"p0", "p2"} and len(out) == 4
    assert out["response"].str.startswith("The answer is ").all()


def test_resume_skips_answers_on_file(env):
    existing = [{"example_id": "e0_0", "pair_id": "p0", "rewrite": "Builder0 built it.", "ok": True,
                 "provider": "claude_cli", "model": MODEL, "cli_version": VERSION}]
    env["out"].write_text("".join(json.dumps(r) + "\n" for r in existing), encoding="utf-8")
    recs = _run(env)
    assert "Builder0" not in {_answer(c["input"]) for c in env["fake"].calls}
    assert len(env["fake"].calls) == 5 and len(recs) == 6
    assert recs[0] == existing[0]
    env["fake"].calls.clear()
    _run(env)  # everything on file: no calls at all
    assert env["fake"].calls == []


def test_max_new_stops_early_then_resumes(env):
    assert len(_run(env, max_new=2)) == 2
    assert len(_run(env)) == 6 and len(env["fake"].calls) == 6


def test_refuses_to_mix_models_in_one_file(env):
    env["out"].write_text(json.dumps({"example_id": "e0_0", "pair_id": "p0", "rewrite": "x", "ok": True,
                                      "provider": "claude_cli", "model": "other-model", "cli_version": VERSION}) + "\n")
    with pytest.raises(RuntimeError, match="refusing"):
        _run(env)
    assert env["fake"].calls == []


@pytest.mark.parametrize("bad", [
    subprocess.CompletedProcess([], 1, "", "boom"),
    _result(""),
    _result("Claude AI usage limit reached|1700000000"),
    _result("x", is_error=True, subtype="error_during_execution"),
    "not json",
    subprocess.TimeoutExpired("claude", 7),
])
def test_failures_are_retried_with_backoff_then_raise(env, bad):
    env["fake"].reply = lambda p: bad
    with pytest.raises(ClaudeCliError, match="after 3 attempts"):
        _run(env, workers=1)
    assert env["sleeps"][:2] == [1, 2]
    assert not env["out"].exists() or env["out"].read_text() == ""


def test_transient_failure_then_success(env):
    state = {"n": 0}

    def reply(prompt):
        state["n"] += 1
        if state["n"] == 1:
            return _result("You've hit your limit. Rate limited, try again later.")
        return _result(f"The answer is {_answer(prompt)}.")

    env["fake"].reply = reply
    recs = _run(env, workers=1)
    assert len(recs) == 6 and all(r["ok"] for r in recs)
    assert env["sleeps"] == [1]


def test_wrong_served_model_raises_without_retry(env):
    env["fake"].reply = lambda p: _result(f"The answer is {_answer(p)}.", model="claude-sonnet-fallback")
    with pytest.raises(ClaudeCliError, match="refusing to record"):
        _run(env, workers=1)
    # No retry; at most the one call the single worker had already started is still in flight.
    assert len(env["fake"].calls) <= 2 and env["sleeps"] == []
    assert env["out"].read_text() == ""


def test_unpinned_model_raises(monkeypatch):
    with pytest.raises(ClaudeCliError, match="no model pinned"):
        ClaudeCliClient(None, run=FakeRun())
