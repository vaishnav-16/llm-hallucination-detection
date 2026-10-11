"""Unified command-line interface: ``python -m src.cli <command> [options]``.

Commands:
  prepare            download + normalize datasets, build splits and the data card     (cpu)
  baselines          answer-only / lexical / ablation shortcut baselines              (cpu)
  rewrite-halueval   label-blind answer rewrite for the normalized HaluEval variant    (api)
  probe-api          check context limit, logprobs and latency of configured APIs     (api)
  probe-jev          tiny live check of TypeSafe Jev (models, grouping, long context) (api)
  score              run detectors x strategies from a YAML config                    (cpu / gpu / api)
  finetune           fine-tune an encoder from a YAML config                          (gpu)
  evaluate           metrics with bootstrap CIs, McNemar tests, calibration          (cpu)
  report             regenerate report/final_report.md and figures                    (cpu)
  status             show which stages are done and where the rest must run
  smoke              tiny end-to-end run of every stage on CPU
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

from src.config import setup_logging

logger = logging.getLogger("src.cli")


def cmd_prepare(args: argparse.Namespace) -> None:
    from src.data.prepare import prepare

    prepare(rebuild_splits=args.rebuild_splits, measure_tokens=not args.no_tokens)


def cmd_status(args: argparse.Namespace) -> None:
    from src.pipeline import status_rows

    rows = status_rows()
    print(f"\n{'stage':<18} {'needs':<8} {'done':<5} command")
    print("-" * 100)
    for r in rows:
        mark = "yes" if r["done"] else "-"
        print(f"{r['stage']:<18} {r['requires']:<8} {mark:<5} {r['command']}")
        if r["note"] and not r["done"]:
            print(f"{'':<33} ({r['note']})")
    todo = [r for r in rows if not r["done"]]
    gpu = [r["stage"] for r in todo if "gpu" in r["requires"]]
    print(f"\n{len(todo)} stage(s) left." + (f" GPU-only (run on the RTX 4070 laptop): {', '.join(gpu)}" if gpu else ""))


def cmd_baselines(args: argparse.Namespace) -> None:
    from src.detectors.baselines import run_shortcut_audit

    run_shortcut_audit()


def cmd_rewrite(args: argparse.Namespace) -> None:
    from src.data.normalize_answers import rewrite_halueval

    rewrite_halueval(provider=args.provider, limit=args.limit)


def cmd_probe(args: argparse.Namespace) -> None:
    from src.detectors.api_judge import probe_providers

    probe_providers(args.providers)


def cmd_score(args: argparse.Namespace) -> None:
    from src.verify.runner import estimate_config, run_config

    if args.dry_run:
        import pandas as pd

        with pd.option_context("display.max_rows", 500, "display.width", 200, "display.max_colwidth", 80):
            print(estimate_config(args.config, limit=args.limit, only=args.only).to_string(index=False))
        return
    run_config(args.config, device=args.device, limit=args.limit, only=args.only)


def cmd_probe_jev(args: argparse.Namespace) -> None:
    from src.detectors.jev import probe

    probe()


def cmd_finetune(args: argparse.Namespace) -> None:
    from src.train.finetune import finetune_from_config

    finetune_from_config(args.config, device=args.device, seeds=args.seeds, max_steps=args.max_steps)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.eval.evaluate import evaluate_all

    evaluate_all()


def cmd_report(args: argparse.Namespace) -> None:
    from src.display.report import generate_report

    generate_report()


def cmd_smoke(args: argparse.Namespace) -> None:
    from src.smoke import run_smoke

    run_smoke(device=args.device)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="download + normalize datasets, splits, data card")
    s.add_argument("--rebuild-splits", action="store_true", help="regenerate artifacts/splits.json (changes committed splits!)")
    s.add_argument("--no-tokens", action="store_true", help="skip tokenizer measurements")
    s.set_defaults(func=cmd_prepare)

    sub.add_parser("status", help="show stage status").set_defaults(func=cmd_status)
    sub.add_parser("baselines", help="shortcut baselines").set_defaults(func=cmd_baselines)

    s = sub.add_parser("rewrite-halueval", help="label-blind HaluEval answer rewrite")
    s.add_argument("--provider", default="dartmouth")
    s.add_argument("--limit", type=int, default=None, help="only rewrite the first N pairs (testing)")
    s.set_defaults(func=cmd_rewrite)

    s = sub.add_parser("probe-api", help="probe configured API providers")
    s.add_argument("--providers", nargs="*", default=None)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("score", help="run detectors from a config")
    s.add_argument("--config", required=True)
    s.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    s.add_argument("--limit", type=int, default=None, help="cap examples per run (testing)")
    s.add_argument("--only", nargs="*", default=None, help="only these run names from the config")
    s.add_argument("--dry-run", action="store_true", help="estimate API requests/tokens/cost without calling anything")
    s.set_defaults(func=cmd_score)

    sub.add_parser("probe-jev", help="tiny live check of the TypeSafe Jev API (< $0.01)").set_defaults(func=cmd_probe_jev)

    s = sub.add_parser("finetune", help="fine-tune an encoder")
    s.add_argument("--config", required=True)
    s.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    s.add_argument("--max-steps", type=int, default=None, help="cap training steps (smoke tests)")
    s.set_defaults(func=cmd_finetune)

    sub.add_parser("evaluate", help="compute metrics").set_defaults(func=cmd_evaluate)
    sub.add_parser("report", help="regenerate report").set_defaults(func=cmd_report)

    s = sub.add_parser("smoke", help="tiny end-to-end run")
    s.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    s.set_defaults(func=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception:
        traceback.print_exc()
        logger.error("Command %r failed", args.command)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
