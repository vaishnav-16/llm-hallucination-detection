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


# ---------------------------------------------------------------------------------------------------------------
# Discussion: every number and every comparative word ("above", "below chance") is computed from artifacts/metrics.
# ---------------------------------------------------------------------------------------------------------------

ANSWER_ONLY_CONTROL = "modernbert_answeronly_s42"
LENS = [500, 2000, 5000, 10000, 20000, 30000]


def _row(long: pd.DataFrame, det: str, sl: str, group: str, kind: str | None = None) -> pd.Series | None:
    """The AUROC row for one detector / slice / group; ``kind`` picks truncate|chunk|retrieve (bidf retrieve first)."""
    d = long[(long["detector"] == det) & (long["slice"] == sl) & (long["group"] == group) & (long["metric"] == "auroc")]
    if kind:
        d = d[d["strategy"].map(_strategy_kind) == kind]
    if d.empty:
        return None
    return d.sort_values("strategy", key=lambda s: ~s.str.endswith("-bidf")).iloc[0]


def _v(r: pd.Series | None) -> str:
    return "n/a" if r is None else f"{r['value']:.3f}"


def _vci(r: pd.Series | None) -> str:
    if r is None:
        return "n/a"
    return f"{r['value']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if np.isfinite(r["ci_lo"]) else f"{r['value']:.3f}"


def _diff(c: pd.Series) -> str:
    sig = "" if c["ci_lo"] <= 0 <= c["ci_hi"] else "*"
    return f"{c['diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]{sig}"


def _range(xs: list[float]) -> str:
    return "n/a" if not xs else f"{min(xs):+.3f} to {max(xs):+.3f}"


def _detectors(long: pd.DataFrame, kind: str) -> list[str]:
    dets = sorted(long.loc[long["slice"] == "phantom_grid_sub", "detector"].unique())
    dets = [d for d in dets if not d.endswith(("_ablate", "_full")) and not d.startswith("answer_")]
    if kind == "zeroshot":
        return [d for d in dets if "_ft_" not in d and "answeronly" not in d]
    return [d for d in dets if "_ft_" in d]


def discussion(long: pd.DataFrame, comp: pd.DataFrame, seeds: pd.DataFrame) -> str:
    out = ["## Discussion\n"]
    tf = lambda sl, g="all": _row(long, "answer_tfidf", sl, g)  # noqa: E731
    ctl = lambda sl, g="all": _row(long, ANSWER_ONLY_CONTROL, sl, g)  # noqa: E731
    tf_hard = tf("phantom_grid_full", "hard")

    # 1. The answer-only lines.
    out.append(
        "### 1. The answer-only line\n\n"
        f"Answer-only TF-IDF reaches {_vci(tf('phantom_grid_full'))} AUROC on PHANTOM test filings "
        f"({_vci(tf_hard)} on the hard subset), {_vci(tf('ragtruth_test'))} on RAGTruth and {_vci(tf('halueval_test'))} on "
        f"original HaluEval, where answer length alone gives {_vci(_row(long, 'answer_length', 'halueval_test', 'all'))}. "
        f"Rewriting the HaluEval answers removes most of that: answer length falls to "
        f"{_vci(_row(long, 'answer_length', 'halueval_norm_test', 'all'))} and TF-IDF to {_vci(tf('halueval_norm_test'))}.\n\n"
        "TF-IDF is not the strongest answer-only line. `modernbert_answeronly_s42` is ModernBERT fine-tuned with the "
        "same recipe as `modernbert_ft_s42` but with every context blanked, so it only ever sees question and answer. It "
        f"reaches {_vci(ctl('phantom_grid_sub'))} on PHANTOM ({_vci(ctl('phantom_grid_sub', 'hard'))} hard, identical at every "
        f"length because it never reads the document), {_vci(ctl('ragtruth_test'))} on RAGTruth and "
        f"{_vci(ctl('halueval_norm_test'))} on rewritten HaluEval. The hard subset was built against the answer-length cue; "
        "a fine-tuned encoder finds other answer-side cues that it does not remove. Fine-tuned detectors are therefore "
        "judged against this control (section 5), zero-shot detectors against TF-IDF.\n"
    )

    # 2-3. Zero-shot detectors and the length curves.
    rows, beats = [], []
    for det in _detectors(long, "zeroshot"):
        r = {"detector": det, "truncate seed": _v(_row(long, det, "phantom_grid_sub", "len=500", "truncate"))}
        for k in ("truncate", "chunk", "retrieve"):
            r[f"{k} 30K"] = _v(_row(long, det, "phantom_grid_sub", "len=30000", k))
        hard = [x for x in (_row(long, det, "phantom_grid_sub", "hard", k) for k in ("truncate", "chunk", "retrieve")) if x is not None]
        best = max(hard, key=lambda x: x["value"]) if hard else None
        r["best hard (strategy)"] = "n/a" if best is None else f"{_vci(best)} ({_strategy_kind(best['strategy'])})"
        rows.append(r)
        if best is not None and tf_hard is not None and best["ci_lo"] > tf_hard["value"]:
            beats.append(det)
    if rows:
        out.append(
            "### 2. Zero-shot detectors on PHANTOM\n\n" + _md_table(pd.DataFrame(rows)) +
            f"\n\nAnswer-only TF-IDF on the hard subset: {_vci(tf_hard)}. Detectors whose best hard-subset CI lies entirely "
            f"above it: {', '.join(f'`{d}`' for d in beats) or 'none'}. The others do not show, on PHANTOM, that they use "
            "the evidence better than a bag of answer words.\n"
        )
        drops, rec, chunk_shift = [], [], []
        for det in _detectors(long, "zeroshot") + _detectors(long, "finetuned"):
            t0, t30 = _row(long, det, "phantom_grid_sub", "len=500", "truncate"), _row(long, det, "phantom_grid_sub", "len=30000", "truncate")
            if t0 is not None and t30 is not None:
                drops.append((det, t0["value"] - t30["value"]))
            c0, c30 = _row(long, det, "phantom_grid_sub", "len=500", "chunk"), _row(long, det, "phantom_grid_sub", "len=30000", "chunk")
            if c0 is not None and c30 is not None:
                chunk_shift.append(c30["value"] - c0["value"])
            c = comp[(comp["detector"] == det) & (comp["slice"] == "phantom_grid_sub") & (comp["group"] == "len=30000")
                     & comp["b"].str.startswith(f"{det}:truncate")]
            for x in c.itertuples():
                rec.append(x)
        sig = [x for x in rec if x.ci_lo > 0]
        out.append(
            "### 3. Truncate, chunk and retrieve as the context grows\n\n"
            "Truncation loses the evidence once it falls outside the window. From the seed passage to 30K nominal tokens "
            "truncate loses " + ", ".join(f"`{d}` {x:+.3f}".replace("+-", "-") for d, x in sorted(drops, key=lambda t: -t[1])) +
            " AUROC (positive = drop). Chunking and retrieval reread the whole document: at 30K, "
            f"{len(sig)} of {len(rec)} chunk/retrieve strategies beat their own truncate on the same rows (paired CI above 0). "
            f"Between the seed passage and 30K, chunk changes by {_range(chunk_shift)} AUROC across detectors. Recovering the "
            "seed-length level is the ceiling here, and section 2 compares that level with the answer-only line.\n"
        )

    # 4. Position.
    pos_rows = []
    for det in _detectors(long, "zeroshot") + _detectors(long, "finetuned"):
        r = {"detector": det}
        for p in ("beginning", "middle", "end"):
            r[p] = _v(_row(long, det, "phantom_grid_sub", f"pos={p}", "chunk"))
        if any(r[p] != "n/a" for p in ("beginning", "middle", "end")):
            vals = [float(r[p]) for p in ("beginning", "middle", "end") if r[p] != "n/a"]
            r["spread"] = f"{max(vals) - min(vals):.3f}"
            pos_rows.append(r)
    if pos_rows:
        out.append(
            "### 4. Evidence position (chunk strategy, PHANTOM long cells)\n\n" + _md_table(pd.DataFrame(pos_rows)) +
            "\n\nWith chunking every window is read, so position matters only through which window holds the evidence and "
            "how much filing text surrounds it. The spread column is the largest gap between positions. The per-cell "
            "heatmaps above show position by length.\n"
        )

    # 5. Fine-tuned vs zero-shot, against the answer-only control.
    c = comp[comp["b"].str.startswith(f"{ANSWER_ONLY_CONTROL}:")]
    keep = c["detector"].str.contains("_ft_") | c["detector"].str.startswith("jev_")
    c = c[keep & c["group"].isin(["all", "hard", "len=30000|hard"])]
    if not c.empty:
        t = c.assign(diff_ci=c.apply(_diff, axis=1))[["detector", "a", "slice", "group", "n", "diff_ci"]]
        t = t.rename(columns={"a": "strategy", "diff_ci": "AUROC minus answer-only control [95% CI]"})
        ph = c[(c["slice"] == "phantom_grid_sub") & c["detector"].str.contains("_ft_")]
        sh = c[c["slice"].isin(["ragtruth_test", "halueval_norm_test"]) & c["detector"].str.contains("_ft_")]
        ft = _detectors(long, "finetuned")
        out.append(
            "### 5. Fine-tuned vs zero-shot, against the answer-only control\n\n"
            "Fine-tuned detectors score " + ", ".join(
                f"`{d}` {_v(_row(long, d, 'phantom_grid_sub', 'all', 'chunk'))} on PHANTOM and "
                f"{_v(_row(long, d, 'ragtruth_test', 'all', 'chunk'))} on RAGTruth (chunk)" for d in ft) +
            ", against at most " + _v(max((r for r in (_row(long, d, 'phantom_grid_sub', 'all', k) for d in _detectors(long, 'zeroshot')
                                              for k in ('truncate', 'chunk', 'retrieve')) if r is not None), key=lambda r: r['value'], default=None)) +
            " for any zero-shot detector on PHANTOM. The paired differences against the answer-only control "
            "(\\* = CI excludes 0) say how much of that comes from reading the evidence:\n\n" + _md_table(t) + "\n\n"
            f"On PHANTOM the fine-tuned detectors differ from the control by {ph['diff'].min():+.3f} to {ph['diff'].max():+.3f} "
            f"AUROC (all rows, hard subset and 30K hard). On RAGTruth and rewritten HaluEval they are "
            f"{sh['diff'].min():+.3f} to {sh['diff'].max():+.3f} relative to it. "
            + _control_verdict(ph, sh) + "\n" if not ph.empty and not sh.empty else "")

    # 6. Ablations and evidence-outside-window rows.
    ab = []
    for det in sorted(long.loc[long["detector"].str.endswith("_ablate") & ~long["detector"].str.endswith("_full"), "detector"].unique()):
        r = {"detector": det}
        for sl, g, name in (("phantom_seed_test", "all", "PHANTOM seed"), ("phantom_seed_test", "hard", "PHANTOM seed hard"),
                            ("ragtruth_test", "all", "RAGTruth"), ("halueval_test", "all", "HaluEval"), ("halueval_norm_test", "all", "HaluEval-norm")):
            r[name] = _v(_row(long, det, sl, g))
        ab.append(r)
    inverted = [r["detector"] for r in ab for k, v in r.items() if k != "detector" and v != "n/a" and float(v) < 0.5]
    ev = []
    for det in _detectors(long, "zeroshot") + _detectors(long, "finetuned"):
        t = _row(long, det, "phantom_grid_sub", "all", "truncate")
        if t is None:
            continue
        w = int(t["strategy"].split("-w")[1].split("_")[0])
        a, h = _row(long, det, "phantom_grid_sub", f"evid>={w}", "truncate"), _row(long, det, "phantom_grid_sub", f"evid>={w}|hard", "truncate")
        if a is not None:
            ev.append({"detector": det, "window": w, "n": int(a["n"]), "AUROC": _vci(a), "hard": _vci(h),
                       "answer-only control, all evid>=W rows": _v(_row(long, ANSWER_ONLY_CONTROL, "phantom_grid_sub", f"evid>={w}"))})
    if ab:
        out.append(
            "### 6. Ablations: what is left without the evidence\n\n"
            "Empty-context runs score each answer with a blank context:\n\n" + _md_table(pd.DataFrame(ab)) + "\n\n"
            + (f"Below 0.5 (inverted, so the score still carries answer information, in reverse): "
               f"{', '.join(sorted(set(f'`{d}`' for d in inverted)))}. " if inverted else "")
            + _ablation_verdict(ab) + "\n\n"
            "An empty context is out of distribution for the fine-tuned models, so it understates what they can do "
            "without evidence. The `evid>=W` rows avoid that: documents whose evidence passage first appears after W words, so "
            "a `truncate-wW` window provably cannot contain it (`artifacts/phantom_evidence_offsets.csv`).\n\n"
            + (_md_table(pd.DataFrame(ev)) + "\n\n" + _evidence_verdict(ev) + "\n" if ev else "")
        )

    # 7. Seed variation.
    if seeds is not None and not seeds.empty:
        s = seeds[(seeds["metric"] == "auroc")]
        s = s.assign(value=s.apply(lambda r: f"{r['mean']:.3f} +/- {r['sd']:.3f} (n={int(r['n_seeds'])})", axis=1))
        s = s[["base", "strategy", "slice", "group", "value"]]
        # Seed-mean gap between the fine-tuned detector and the answer-only control on the slices both were seeded on.
        ft, ctl = (seeds[(seeds["metric"] == "auroc") & (seeds["base"] == b)].set_index(["slice", "group"])
                   for b in ("modernbert_ft", ANSWER_ONLY_CONTROL.rsplit("_s", 1)[0]))
        both = ft.index.intersection(ctl.index)
        gaps = [(f"{sl} {g}", ft.loc[(sl, g), "mean"] - ctl.loc[(sl, g), "mean"],
                 max(ft.loc[(sl, g), "sd"], ctl.loc[(sl, g), "sd"])) for sl, g in both]
        gap_txt = ("Seed-mean gap, fine-tuned ModernBERT minus the answer-only control: "
                   + ", ".join(f"{n} {d:+.3f} (larger seed sd {sd:.3f})" for n, d, sd in gaps) + ". " if gaps else "")
        out.append("### 7. Seed variation\n\nMean +/- sd over training seeds:\n\n" + _md_table(s) + "\n\n"
                   f"The largest seed sd is {seeds.loc[seeds['metric'] == 'auroc', 'sd'].max():.3f} AUROC. " + gap_txt
                   + "With 3 seeds the sd is a rough estimate, so it only rules out seed noise as the cause of gaps several times "
                   "larger than it. The long-context PHANTOM grid has one fine-tuned seed (42), so the fine-tuned results in sections 3-6 rest on that seed.\n")
    else:
        out.append("### 7. Seed variation\n\nOnly one training seed (42) is available, so there is no seed-variation estimate. "
                   "The CIs above cover test-sample variation, not training variation.\n")

    summ = ["### Summary\n"]
    if rows:
        summ.append(f"- Truncation costs the detectors up to {max(x for _, x in drops):.3f} AUROC between the seed passage and "
                    f"30K tokens; {len(sig)} of {len(rec)} chunk/retrieve strategies beat their own truncate at 30K on the same rows.")
        summ.append(f"- Zero-shot detectors above answer-only TF-IDF on the PHANTOM hard subset: {', '.join(f'`{d}`' for d in beats) or 'none'}.")
    if not c.empty:
        ph = c[(c["slice"] == "phantom_grid_sub") & c["detector"].str.contains("_ft_")]
        sh = c[c["slice"].isin(["ragtruth_test", "halueval_norm_test"]) & c["detector"].str.contains("_ft_")]
        if not ph.empty and not sh.empty:
            summ.append(f"- Fine-tuned detectors minus the answer-only fine-tuned control: {ph['diff'].min():+.3f} to "
                        f"{ph['diff'].max():+.3f} AUROC on PHANTOM, {sh['diff'].min():+.3f} to {sh['diff'].max():+.3f} on RAGTruth "
                        "and rewritten HaluEval.")
    out.append("\n".join(summ) + "\n")
    return "\n".join(out)


def _control_verdict(ph: pd.DataFrame, sh: pd.DataFrame) -> str:
    """Sentences worded from the paired CIs, never from expectations."""
    pos_ph, neg_ph, pos_sh = ph[ph["ci_lo"] > 0], ph[ph["ci_hi"] < 0], sh[sh["ci_lo"] > 0]
    parts = []
    if len(pos_sh) == len(sh):
        parts.append("Every RAGTruth and HaluEval comparison is above the control (CI above 0), so on those datasets the "
                     "fine-tuned models use the evidence.")
    else:
        parts.append(f"{len(pos_sh)} of {len(sh)} RAGTruth/HaluEval comparisons are above the control.")
    parts.append(f"On PHANTOM, {len(pos_ph)} of {len(ph)} comparisons are above the control (largest gain "
                 f"{ph['diff'].max():+.3f}) and {len(neg_ph)} are below it (largest loss {ph['diff'].min():+.3f}).")
    worst = ph.loc[ph["diff"].idxmin()]
    if worst["ci_hi"] < 0:
        parts.append(f"The largest loss is `{worst['detector']}` {_strategy_kind(worst['a'])} ({worst['group']}): reading a "
                     "window that does not contain the evidence scores worse than not reading the context at all.")
    return " ".join(parts)


def _ablation_verdict(ab: list[dict]) -> str:
    def f(r: dict, k: str) -> float | None:
        return float(r[k]) if r.get(k, "n/a") != "n/a" else None
    lines = []
    for r in ab:
        if "_ft_" not in r["detector"]:
            continue
        ph, rt = f(r, "PHANTOM seed"), f(r, "RAGTruth")
        if ph is not None and rt is not None:
            lines.append(f"With no context, `{r['detector']}` scores {ph:.3f} on PHANTOM seed passages and {rt:.3f} on RAGTruth.")
    return " ".join(lines)


def _evidence_verdict(ev: list[dict]) -> str:
    def val(s: str) -> float:
        return float(s.split(" ")[0])
    zs = [r for r in ev if "_ft_" not in r["detector"]]
    ft = [r for r in ev if "_ft_" in r["detector"]]
    out = []
    if zs:
        out.append(f"With the evidence out of view, zero-shot detectors score {min(val(r['AUROC']) for r in zs):.3f} to "
                   f"{max(val(r['AUROC']) for r in zs):.3f}.")
    if ft:
        out.append("The fine-tuned detectors score " + ", ".join(
            f"`{r['detector']}` {val(r['AUROC']):.3f} (answer-only control on all evid>={r['window']} rows: {r['answer-only control, all evid>=W rows']})"
            for r in ft) + ".")
    return " ".join(out)


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

    if long_path.exists() and (METRICS_DIR / "comparisons.csv").exists():
        seeds = read_table(seeds_path) if seeds_path.exists() and seeds_path.stat().st_size > 1 else None
        sections.append(discussion(long, read_table(METRICS_DIR / "comparisons.csv"), seeds))
    else:
        sections.append("## Discussion\n\n_Pending: run `python -m src.cli evaluate`._\n")
    sections.append(
        "## Limitations\n\n"
        "- The answer-only control is one architecture (ModernBERT) and, for the PHANTOM grid, one seed; another encoder "
        "could find more or fewer answer-side cues.\n"
        "- `evid>=W` locates the evidence by exact match on the seed passage's first 200 characters. A fact restated "
        "elsewhere in the filing in other words is not detected, so a few of those rows may still hold usable evidence.\n"
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
                 f"`artifacts/predictions/manifest/`); {sp['requests']} live requests, {sp['input_tokens']:,} input tokens, ${sp['usd']:.2f}")
    sections.append("## Reproducibility\n\nPinned revisions:\n\n" + revs + "\n\nRun `python -m src.cli status` to see how each artifact is produced.\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "final_report.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Wrote %s", out)
    return out
