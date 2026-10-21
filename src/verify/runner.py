"""Scoring runner: dataset slices x detectors x chunking strategies -> prediction files.

A YAML config lists runs. Each run has a detector, one or more chunking strategies and one or more
slices. Work is done one *part* (a PHANTOM cell, or a whole small slice) at a time; each finished part
is written to ``results/runs/<run_id>/<part>.parquet``, so an interrupted run resumes where it stopped.
When every part of a run is done, the parts are merged into ``artifacts/predictions/<run_id>.parquet``
and recorded in ``artifacts/predictions/manifest/<run_id>.json`` with a hash of the run's config. One file per run
means the laptop (GPU runs) and the cloud session (API runs) can push in parallel without git conflicts. A changed
config invalidates the old result.

Example config::

    name: zeroshot_grid
    runs:
      - name: hhem
        detector: {type: hhem, batch_size: 8, max_length: 4096}
        strategies:
          - {strategy: truncate, window: 512}
          - {strategy: chunk, window: 2048, overlap: 1536}
        slices: [phantom_grid_sub, phantom_val, halueval_test, ragtruth_test]
        hypothesis: qa              # qa | answer
        ablate_context: false       # true: score with an empty context (shortcut control)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.config import ARTIFACTS_DIR, RESULTS_DIR, ROOT
from src.data.store import read_json, read_table, write_json, write_table
from src.detectors.base import Detector
from src.verify.chunking import ChunkSpec, aggregate, build_premises, hypothesis_text
from src.verify.slices import META_COLUMNS, get_slice

logger = logging.getLogger(__name__)

PRED_DIR = ARTIFACTS_DIR / "predictions"


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg or "runs" not in cfg:
        raise ValueError(f"{path}: config needs a 'runs' list")
    return cfg


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def expand_runs(cfg: dict) -> list[dict]:
    """One job per (run, strategy, slice), with a readable run_id and a config hash."""
    jobs = []
    for run in cfg["runs"]:
        for strat in run.get("strategies", [{"strategy": "truncate", "window": 512}]):
            spec = ChunkSpec(**strat)
            for entry in run["slices"]:
                sl = entry["name"] if isinstance(entry, dict) else entry
                max_pairs = entry.get("max_pairs", run.get("max_pairs")) if isinstance(entry, dict) else run.get("max_pairs")
                ablate = bool(run.get("ablate_context", False))
                run_id = f"{run['name']}__{spec.tag()}__{sl}" + ("__ablate" if ablate else "")
                job = {
                    "run_id": run_id, "run": run["name"], "detector": run["detector"], "spec": strat, "slice": sl,
                    "hypothesis": run.get("hypothesis", "qa"), "ablate": ablate, "max_pairs": max_pairs,
                }
                job["hash"] = _hash({k: job[k] for k in ("detector", "spec", "slice", "hypothesis", "ablate", "max_pairs")})
                jobs.append(job)
    ids = [j["run_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate run ids in config (two runs share a name?)")
    return jobs


def output_dir(cfg: dict, limit: int | None = None) -> Path:
    """Committed predictions go to artifacts/predictions; smoke configs and --limit runs go to scratch."""
    if cfg.get("scratch") or limit is not None:
        return RESULTS_DIR / "scratch_predictions"
    return PRED_DIR


def _manifest(pred_dir: Path = PRED_DIR) -> dict:
    """All run records: one JSON file per run under manifest/ (plus a legacy single manifest.json, if present)."""
    man = read_json(pred_dir / "manifest.json") if (pred_dir / "manifest.json").exists() else {}
    for p in sorted((pred_dir / "manifest").glob("*.json")):
        man[p.stem] = read_json(p)
    return man


def _is_done(job: dict, man: dict, pred_dir: Path) -> bool:
    return man.get(job["run_id"], {}).get("hash") == job["hash"] and (pred_dir / f"{job['run_id']}.parquet").exists()


def config_runs_complete(path: str | Path) -> bool:
    try:
        cfg = load_config(path)
    except FileNotFoundError:
        return False
    pred_dir = output_dir(cfg)
    man = _manifest(pred_dir)
    return all(_is_done(j, man, pred_dir) for j in expand_runs(cfg))


def build_detector(dcfg: dict, device: str) -> Detector:
    """Construct a detector from its config block."""
    kind = dcfg["type"]
    kw = {k: v for k, v in dcfg.items() if k != "type"}
    if kind == "hhem":
        from src.detectors.hf_models import HHEMDetector

        return HHEMDetector(device=device, **kw)
    if kind == "seqcls":
        from src.detectors.hf_models import SequenceClassifierDetector

        return SequenceClassifierDetector(device=device, **kw)
    if kind == "lettuce":
        from src.detectors.hf_models import LettuceDetector

        return LettuceDetector(device=device, **kw)
    if kind == "api_judge":
        from src.detectors.api_judge import ApiJudge

        return ApiJudge(**kw)
    if kind == "jev":
        from src.detectors.jev import JevJudge

        return JevJudge(**kw)
    if kind == "local_llm":
        from src.detectors.llm_judge import LocalLLMJudge

        return LocalLLMJudge(device=device, **kw)
    if kind == "stub":
        from src.detectors.stub import StubDetector

        return StubDetector(**kw)
    raise ValueError(f"unknown detector type {kind!r}")


def build_inputs(df: pd.DataFrame, det: Detector, spec: ChunkSpec, hyp_fmt: str = "qa", ablate: bool = False) -> tuple[list[str], list[str], list[int]]:
    """Flatten a frame into (premise chunk, hypothesis) pairs plus the row index that owns each pair.

    Detectors that need the question separately from the response (LettuceDetect, Jev) define
    ``pack_question``; everything else gets ``hypothesis_text``.
    """
    span_fn = det.span_fn()
    pack = getattr(det, "pack_question", None)
    premises: list[str] = []
    hyps: list[str] = []
    owner: list[int] = []
    for i, r in enumerate(df.itertuples(index=False)):
        hyp = pack(r.question, r.response, r.task_type) if pack else hypothesis_text(r.question, r.response, r.task_type, hyp_fmt)
        context = "" if ablate else r.context
        chunks = build_premises(context, f"{r.question} {r.response}", spec, span_fn) if context else [""]
        premises += chunks
        hyps += [hyp] * len(chunks)
        owner += [i] * len(chunks)
    return premises, hyps, owner


def score_frame(df: pd.DataFrame, det: Detector, spec: ChunkSpec, hyp_fmt: str = "qa", ablate: bool = False) -> pd.DataFrame:
    """Score every row of a canonical frame. Returns META_COLUMNS + p_halluc, n_chunks, latency_ms."""
    t0 = time.time()
    premises, hyps, owner = build_inputs(df, det, spec, hyp_fmt, ablate)
    t_chunk = time.time() - t0
    window_scores = det.score_pairs(premises, hyps)
    owner_arr = np.asarray(owner)
    p = np.empty(len(df))
    n_chunks = np.bincount(owner_arr, minlength=len(df))
    for i in range(len(df)):
        p[i] = aggregate(window_scores[owner_arr == i], spec.aggregation, spec.top_k)
    elapsed = time.time() - t0
    out = df[META_COLUMNS].copy()
    out["p_halluc"] = p
    out["n_chunks"] = n_chunks
    out["latency_ms"] = 1000.0 * elapsed / max(1, len(df))
    logger.info("  scored %d rows / %d windows in %.1fs (chunking %.1fs)", len(df), len(premises), elapsed, t_chunk)
    return out


def run_job(job: dict, det: Detector, limit: int | None = None, pred_dir: Path = PRED_DIR) -> Path:
    """Run one (run, strategy, slice) job part by part, resuming finished parts."""
    spec = ChunkSpec(**job["spec"])
    shard_dir = RESULTS_DIR / ("runs" if pred_dir == PRED_DIR else "scratch_runs") / job["run_id"]
    meta_path = shard_dir / "_job.json"
    if meta_path.exists() and read_json(meta_path).get("hash") != job["hash"]:
        logger.info("%s: config changed, discarding old shards", job["run_id"])
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_json({"hash": job["hash"], "job": job}, meta_path)

    parts = get_slice(job["slice"], max_pairs=job.get("max_pairs") or limit)
    t0 = time.time()
    for k, part in enumerate(parts, 1):
        out = shard_dir / f"{part.part}.parquet"
        if out.exists():
            continue
        df = part.loader()
        if df.empty:
            logger.warning("%s: part %s is empty, skipping", job["run_id"], part.part)
            continue
        logger.info("%s [%d/%d] %s: %d rows", job["run_id"], k, len(parts), part.part, len(df))
        pred = score_frame(df, det, spec, job["hypothesis"], job["ablate"])
        pred["part"] = part.part
        write_table(pred, out)

    shards = sorted(p for p in shard_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"{job['run_id']}: no predictions produced")
    pred = pd.concat([read_table(p) for p in shards], ignore_index=True)
    pred["detector"] = job["run"]
    pred["strategy"] = spec.tag() + ("__ablate" if job["ablate"] else "")
    pred["run_id"] = job["run_id"]
    final = write_table(pred, pred_dir / f"{job['run_id']}.parquet")
    record = {
        "hash": job["hash"], "run": job["run"], "slice": job["slice"], "strategy": spec.tag(), "ablate": job["ablate"],
        "detector": det.describe(), "spec": job["spec"], "rows": int(len(pred)), "limit": limit or job.get("max_pairs"),
        "elapsed_s": round(time.time() - t0, 1), "git_commit": _git_commit(), "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(record, pred_dir / "manifest" / f"{job['run_id']}.json")
    return final


def estimate_config(path: str | Path, limit: int | None = None, only: list[str] | None = None) -> pd.DataFrame:
    """Dry run: requests / tokens / USD per unfinished job for detectors that implement ``estimate`` (API judges).

    Nothing is sent to any API. Cached requests count as free.
    """
    cfg = load_config(path)
    jobs = [j for j in expand_runs(cfg) if not only or j["run"] in only]
    pred_dir = output_dir(cfg, limit)
    man = _manifest(pred_dir)
    rows, dets = [], {}
    for j in jobs:
        if _is_done(j, man, pred_dir):
            continue
        if j["run"] not in dets:
            dets[j["run"]] = build_detector(j["detector"], "cpu")
        det = dets[j["run"]]
        if not hasattr(det, "estimate"):
            rows.append({"run_id": j["run_id"], "note": "no estimate (local model)"})
            continue
        spec = ChunkSpec(**j["spec"])
        tot = {"rows": 0, "requests": 0, "uncached_requests": 0, "approx_tokens": 0, "approx_usd": 0.0}
        for part in get_slice(j["slice"], max_pairs=j.get("max_pairs") or limit):
            df = part.loader()
            if df.empty:
                continue
            premises, hyps, _ = build_inputs(df, det, spec, j["hypothesis"], j["ablate"])
            est = det.estimate(premises, hyps)
            tot["rows"] += len(df)
            for k in ("requests", "uncached_requests", "approx_tokens", "approx_usd"):
                tot[k] += est[k]
        rows.append({"run_id": j["run_id"], **tot})
    out = pd.DataFrame(rows)
    if not out.empty and "approx_usd" in out:
        # Runs that send byte-identical requests (e.g. jev_noul / jev_choice) share the cache: count them once.
        logger.info("Estimated spend if every listed job ran: $%.2f (identical requests across runs are paid once, "
                    "so the real total is lower when runs share questions)", out["approx_usd"].sum())
    return out


def run_config(path: str | Path, device: str = "auto", limit: int | None = None, only: list[str] | None = None) -> None:
    """Run every job in a config, building each detector once and skipping finished jobs."""
    from src.config import get_device, seed_everything

    seed_everything()
    cfg = load_config(path)
    jobs = expand_runs(cfg)
    if only:
        jobs = [j for j in jobs if j["run"] in only]
    pred_dir = output_dir(cfg, limit)
    man = _manifest(pred_dir)
    todo = [j for j in jobs if not _is_done(j, man, pred_dir)]
    logger.info("%s: %d jobs, %d to run -> %s", cfg.get("name", path), len(jobs), len(todo), pred_dir)
    if not todo:
        return
    dev = get_device(device)
    by_run: dict[str, list[dict]] = {}
    for j in todo:
        by_run.setdefault(j["run"], []).append(j)
    for run_name, run_jobs in by_run.items():
        logger.info("Loading detector for run %r: %s", run_name, run_jobs[0]["detector"])
        det = build_detector(run_jobs[0]["detector"], dev)
        for j in run_jobs:
            run_job(j, det, limit, pred_dir)
        del det
