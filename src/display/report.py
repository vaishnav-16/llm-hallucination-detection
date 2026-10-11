"""`report` stage: figures + ``report/final_report.md`` assembled only from committed artifacts.

Every number in the report is read from ``artifacts/`` (metrics, data card, manifest, training logs),
so the report can be regenerated on any machine without the datasets or model weights. Sections whose
inputs do not exist yet are marked as pending instead of being filled with placeholder numbers.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DATASETS, MODELS, REPORT_DIR
from src.data.store import read_json, read_table

logger = logging.getLogger(__name__)

FIG_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

# Reference palette (dataviz skill), light mode. Slots 1-3 validate all-pairs; aqua needs relief
# (direct labels + markers + tables), which the figures provide.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK_2, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BASELINE = "#8a8984"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "text.color": INK,
    })
    return plt


def _strategy_kind(s: str) -> str:
    return s.split("-")[0]


def plot_length_curves(long: pd.DataFrame, slice_name: str, group_suffix: str, baseline: pd.DataFrame | None, out: Path) -> Path | None:
    """Small multiples: one panel per detector, AUROC vs context length, one line per chunking strategy."""
    d = long[(long["slice"] == slice_name) & (long["metric"] == "auroc") & long["group"].str.match(rf"^len=\d+{group_suffix}$")].copy()
    d = d[~d["detector"].str.startswith("answer_") & (d["detector"] != "lexical_support")]
    if d.empty:
        return None
    d["len"] = d["group"].str.extract(r"len=(\d+)")[0].astype(int)
    dets = sorted(d["detector"].unique())
    plt = _mpl()
    ncol = min(3, len(dets))
    nrow = int(np.ceil(len(dets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
    lens = sorted(d["len"].unique())
    xpos = {n: i for i, n in enumerate(lens)}
    for ax, det in zip(axes.flat, dets):
        dd = d[d["detector"] == det]
        kinds = sorted(dd["strategy"].unique(), key=lambda s: ["truncate", "chunk", "retrieve"].index(_strategy_kind(s)) if _strategy_kind(s) in ("truncate", "chunk", "retrieve") else 9)
        if baseline is not None and not baseline.empty:
            ax.axhline(baseline["value"].mean(), color=BASELINE, lw=1.5, ls=(0, (4, 3)), zorder=1)
            ax.text(len(lens) - 1, baseline["value"].mean(), " answer-only", color=INK_MUTED, va="bottom", ha="right", fontsize=8)
        ends = []
        for k, strat in enumerate(kinds[:3]):
            s = dd[dd["strategy"] == strat].sort_values("len")
            x = [xpos[n] for n in s["len"]]
            ax.fill_between(x, s["ci_lo"], s["ci_hi"], color=SERIES[k], alpha=0.12, lw=0, zorder=2)
            ax.plot(x, s["value"], color=SERIES[k], lw=2, marker=MARKERS[k], ms=6, mec=SURFACE, mew=1.5, zorder=3,
                    label=_strategy_kind(strat))
            ends.append([float(s["value"].iloc[-1]), x[-1], _strategy_kind(strat)])
        ends.sort()
        for i in range(1, len(ends)):  # keep direct labels at least 0.035 AUROC apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
        for y, x, text in ends:
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", color=INK_2, fontsize=8, va="center")
        ax.set_title(det, loc="left")
        ax.set_xticks(range(len(lens)), ["seed" if n == 500 else f"{n // 1000}K" for n in lens])
        ax.set_xlim(-0.3, len(lens) - 0.3)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color=GRID, lw=1)
    for ax in axes.flat[len(dets):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")
    for ax in axes[-1, :]:
        ax.set_xlabel("PHANTOM context length (nominal tokens)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_position_heatmap(long: pd.DataFrame, slice_name: str, det: str, strat: str, out: Path) -> Path | None:
    d = long[(long["slice"] == slice_name) & (long["detector"] == det) & (long["strategy"] == strat) & (long["metric"] == "auroc") & long["group"].str.startswith("cell=")].copy()
    if d.empty:
        return None
    parts = d["group"].str.removeprefix("cell=").str.split("|", expand=True)
    d["len"], d["pos"] = parts[1].astype(int), parts[2]
    t = d.groupby(["pos", "len"])["value"].mean().unstack("len").reindex(["beginning", "middle", "end"])
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(1.1 * t.shape[1] + 1.8, 2.6))
    ax.grid(False)
    im = ax.imshow(t.to_numpy(), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=SURFACE if v > 0.78 else INK)
    ax.set_xticks(range(t.shape[1]), [f"{n // 1000}K" for n in t.columns])
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.set_title(f"{det} ({_strategy_kind(strat)}): AUROC by evidence position", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _rel(p: Path) -> str:
    return "../" + p.relative_to(ARTIFACTS_DIR.parent).as_posix()


def generate_report() -> Path:
    from src.pipeline import status_rows

    sections: list[str] = []
    status = status_rows()
    done = [r["stage"] for r in status if r["done"]]
    pending = [r["stage"] for r in status if not r["done"]]
    sections.append(
        "# Length-robust hallucination detection on PHANTOM, RAGTruth and HaluEval\n\n"
        f"_Generated {_dt.date.today().isoformat()} by `python -m src.cli report` from the committed `artifacts/`._\n\n"
        f"**Pipeline status.** Done: {', '.join(done) or 'none'}. Pending: {', '.join(pending) or 'none'}. "
        "Sections that depend on pending stages say so explicitly.\n"
    )

    sections.append(
        "## Research question\n\nHow do hallucination detectors degrade as the context grows from a ~500-token passage to 30K "
        "nominal tokens and as the evidence moves to the beginning, middle or end, and does chunked or retrieval-based "
        "verification recover the loss? Every detector is evaluated next to answer-only shortcut baselines, because all "
        "three benchmarks can be partly solved without reading the context.\n"
    )

    sb_path = METRICS_DIR / "shortcut_baselines.csv"
    if sb_path.exists():
        sb = read_table(sb_path)
        sb = sb[sb["eval"].str.endswith(("test", "grid_full"))]
        piv = sb.pivot_table(index="eval", columns="detector", values="auroc").reset_index()
        sections.append(
            "## Shortcut audit\n\nAUROC of baselines trained on each benchmark's training split. `answer_length` and "
            "`answer_tfidf` never see the context; `lexical_support` is 1 minus the share of the answer's content words "
            "found in the context. PHANTOM baselines are trained on 8-K/497K and tested on 10-K/DEF 14A at every length.\n\n"
            + _md_table(piv) + "\n\nA detector only demonstrates that it uses the context where it beats the answer-only "
            "line, or on the length-matched hard subset where the answer-only signal is weaker.\n"
        )

    card_path = ARTIFACTS_DIR / "data_card.json"
    if card_path.exists():
        card = read_json(card_path)
        sl = card["slices"]
        joins = {k: v["join"] for k, v in sl.items() if "join" in v}
        reans = sum(v["reanswered"] for v in joins.values())
        total = sum(sum(v.values()) for v in joins.values())
        by_filing: dict[str, list[int]] = {}
        for k, v in joins.items():
            f = k.split("_")[1]
            by_filing.setdefault(f, [0, 0])
            by_filing[f][0] += v["reanswered"]
            by_filing[f][1] += sum(v.values())
        per_filing = "; ".join(f"{f}: {a} of {b} ({a / max(1, b):.0%})" for f, (a, b) in sorted(by_filing.items()))
        sections.append(
            "## Data\n\n"
            f"- HaluEval QA: {sl['halueval']['pairs']} questions ({sl['halueval']['rows']} rows). RAGTruth: "
            f"{sl['ragtruth_train']['rows']} train / {sl['ragtruth_test']['rows']} test responses.\n"
            f"- PHANTOM: 4 seed files plus 30 long-context files (10-K and DEF 14A at 2K-30K tokens x 3 positions). "
            f"Long rows are joined to their seed pair; {reans} of {total} long-file pairs have answers regenerated "
            f"relative to the seed (`variant=reanswered`; {per_filing}).\n"
            "- Splits are grouped by pair: PHANTOM 10-K and DEF 14A are test-only at every length; 8-K/497K are train/val; "
            "HaluEval 70/15/15; RAGTruth official test, train split 85/15. `grid_sub` = 200 pairs per test filing present in every cell.\n"
            "- Full statistics and tokenizer measurements: [`artifacts/data_card.md`](../artifacts/data_card.md).\n"
        )

    long_path = METRICS_DIR / "metrics_long.csv"
    if not long_path.exists():
        sections.append("## Results\n\n_Pending: run the scoring configs, then `python -m src.cli evaluate`._\n")
    else:
        long = read_table(long_path)
        head = read_table(METRICS_DIR / "headline.csv")
        sections.append("## Results\n\nAUROC with 95% pair-clustered bootstrap CIs. Thresholds for F1 and balanced accuracy are chosen on validation slices only.\n\n" + _md_table(head) + "\n")
        base = long[(long["detector"] == "answer_tfidf") & (long["slice"] == "phantom_grid_full") & (long["metric"] == "auroc") & long["group"].str.match(r"^len=\d+$")]
        for sl_name in ("phantom_grid_sub", "phantom_grid_full"):
            for suffix, label in (("", "all pairs"), (r"\|hard", "hard subset")):
                f = plot_length_curves(long, sl_name, suffix, base, FIG_DIR / f"length_curves_{sl_name}{'_hard' if suffix else ''}.png")
                if f:
                    sections.append(f"\n### {sl_name}: AUROC vs context length ({label})\n\n![length curves]({_rel(f)})\n")
        sub = long[(long["slice"] == "phantom_grid_sub") & (long["metric"] == "auroc") & (long["group"] == "all")]
        for (det, strat), _ in sub.groupby(["detector", "strategy"]):
            if strat.startswith("chunk") and not det.startswith("answer_"):
                f = plot_position_heatmap(long, "phantom_grid_sub", det, strat, FIG_DIR / f"position_{det}_{strat}.png")
                if f:
                    sections.append(f"![position heatmap]({_rel(f)})\n")
        comp_path = METRICS_DIR / "comparisons.csv"
        if comp_path.exists() and comp_path.stat().st_size > 1:
            comp = read_table(comp_path)
            if not comp.empty:
                c = comp[comp["group"] == "all"][["detector", "slice", "a", "b", "n", "diff", "ci_lo", "ci_hi", "p_value"]]
                sections.append("\n### Paired comparisons (AUROC difference a - b, same rows)\n\n" + _md_table(c) + "\n")

    seeds_path = METRICS_DIR / "seed_summary.csv"
    if seeds_path.exists() and seeds_path.stat().st_size > 1:
        ss = read_table(seeds_path)
        if not ss.empty:
            ss = ss[ss["metric"] == "auroc"][["base", "strategy", "slice", "group", "n_seeds", "mean", "sd"]]
            sections.append("\n### Seed variation (AUROC mean and sd over training seeds)\n\n" + _md_table(ss) + "\n")

    train_logs = sorted((ARTIFACTS_DIR / "training").glob("*.json"))
    if train_logs:
        rows = []
        for p in train_logs:
            m = read_json(p)
            rows.append({"run": p.stem, "base": m["base"]["repo_id"], "train_rows": m["train_rows"], "val_auroc": m["best_val"].get("eval_auroc"), "minutes": m["train_minutes"]})
        sections.append("## Fine-tuning\n\n" + _md_table(pd.DataFrame(rows)) + "\n")
    else:
        sections.append("## Fine-tuning\n\n_Pending: GPU stage `finetune` (see HANDOFF.md)._\n")

    sections.append(
        "## Discussion\n\n_To be written once the GPU stages are complete. It must discuss each result against the "
        "answer-only line and the hard subset, not in isolation._\n\n"
        "## Limitations\n\n"
        "- All three benchmarks contain answer-only signal (see the shortcut audit); absolute scores overstate grounding.\n"
        "- PHANTOM's long variants are built around a single 500-token evidence passage, so `min_phalluc` (any window "
        "supports the answer) suits it; claims needing evidence from several windows are not tested.\n"
        "- 512-token models (DeBERTa) cannot hold most seed passages in one window; their results are partial-evidence results.\n"
        "- `grid_sub` uses 200 pairs per filing per cell; CIs reflect this.\n"
        "- API judges (Jev, Dartmouth Chat, Gemini) can change behind the same name; each run records the model id that "
        "answered, and Jev is pinned to `jev-1.13.0` rather than the moving `jev-latest` alias.\n"
        "- Jev results use a 60-pair-per-cell PHANTOM sample (cost); its CIs are wider than the encoders' 200-pair grid.\n"
    )

    revs = "\n".join(f"- `{v['repo_id']}` @ `{v['revision'][:12]}`" for v in {**DATASETS, **MODELS}.values() if v.get("revision"))
    spend_path = ARTIFACTS_DIR / "api_cache" / "jev_spend.json"
    if spend_path.exists():
        sp = read_json(spend_path)
        revs += (f"\n- TypeSafe Jev API, model pinned to `jev-1.13.0` (served model ids are recorded in "
                 f"`artifacts/predictions/manifest.json`); {sp['requests']} live requests, {sp['input_tokens']:,} input tokens, ${sp['usd']:.2f}")
    sections.append("## Reproducibility\n\nPinned revisions:\n\n" + revs + "\n\nRun `python -m src.cli status` to see how each artifact is produced.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "final_report.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Wrote %s", out)
    return out
