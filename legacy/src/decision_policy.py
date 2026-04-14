"""
decision_policy.py — 3-tier confidence-based decision policy for hallucination detection.

Implements ANSWER / ANSWER WITH CAVEATS / ABSTAIN framework at three tolerance levels.
Generates calibration and coverage-precision trade-off figures.
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def compute_policy_metrics(y_true, probs, low_thresh, high_thresh):
    """
    Compute metrics for a 3-tier policy.

    probs: hallucination probability (high = more hallucinated)
    ANSWER: prob <= low_thresh (confident faithful)
    CAVEAT: low_thresh < prob <= high_thresh (uncertain)
    ABSTAIN: prob > high_thresh (confident hallucinated)
    """
    n = len(y_true)
    answer_mask = probs <= low_thresh
    abstain_mask = probs > high_thresh
    caveat_mask = ~answer_mask & ~abstain_mask

    # Among ANSWER examples: how many are actually hallucinated?
    served = answer_mask
    if served.sum() == 0:
        halluc_rate_served = 0.0
    else:
        halluc_rate_served = y_true[served].mean()

    # Coverage = fraction of examples where we give an answer (ANSWER + CAVEAT)
    coverage = (answer_mask | caveat_mask).mean()

    # Precision on served = 1 - halluc_rate_served
    precision_served = 1.0 - halluc_rate_served

    # False abstention rate: faithful examples we abstain from
    faithful_mask = y_true == 0
    if faithful_mask.sum() == 0:
        false_abstain_rate = 0.0
    else:
        false_abstain_rate = abstain_mask[faithful_mask].mean()

    return {
        "coverage": float(coverage),
        "pct_answer": float(answer_mask.mean()),
        "pct_caveat": float(caveat_mask.mean()),
        "pct_abstain": float(abstain_mask.mean()),
        "halluc_rate_in_answers": float(halluc_rate_served),
        "precision_of_answers": float(precision_served),
        "false_abstention_rate": float(false_abstain_rate),
        "n_answer": int(answer_mask.sum()),
        "n_caveat": int(caveat_mask.sum()),
        "n_abstain": int(abstain_mask.sum()),
    }


def find_thresholds_for_tolerance(y_true, probs, max_halluc_rate: float):
    """
    Sweep for low_thresh such that halluc_rate_in_answers <= max_halluc_rate,
    maximising coverage. Then set high_thresh to create a caveat band.

    probs: hallucination probability (high = hallucinated).
    ANSWER zone: prob <= low_thresh  (confident faithful).
    """
    best_low = 0.05
    best_coverage = 0.0
    # Sweep from high to low: the highest low_thresh that still satisfies
    # the hallucination-rate constraint gives the best coverage.
    for low in np.arange(0.95, 0.04, -0.01):
        answer_mask = probs <= low
        if answer_mask.sum() == 0:
            continue
        rate = y_true[answer_mask].mean()
        coverage = answer_mask.mean()
        if rate <= max_halluc_rate and coverage > best_coverage:
            best_low = float(low)
            best_coverage = coverage
    # Caveat band: from best_low up to best_low + 0.15 (tunable)
    high = min(0.95, best_low + 0.15)
    return float(best_low), float(high)


def reliability_diagram(y_true, probs, n_bins=10, ax=None, title="Calibration"):
    """Plot reliability diagram (calibration curve)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_pred = []
    mean_actual = []
    sizes = []

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        mean_pred.append(probs[mask].mean())
        mean_actual.append(y_true[mask].mean())
        sizes.append(mask.sum())

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    scatter = ax.scatter(mean_pred, mean_actual, c=sizes, cmap="Blues", s=80, zorder=5)
    ax.plot(mean_pred, mean_actual, color="darkorange", linewidth=2, label="Model calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of hallucinated", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hhem_preds", default="results/hhem_predictions.csv")
    parser.add_argument("--ft_preds", default="results/finetuned_predictions.csv")
    parser.add_argument("--output_metrics", default="results/decision_policy_metrics.json")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # Load predictions
    hhem_df = pd.read_csv(args.hhem_preds)
    y_true = hhem_df["label"].values

    # HHEM: convert consistency score to hallucination probability
    hhem_probs = 1.0 - hhem_df["hhem_score"].values

    # Fine-tuned: load if available, else fall back to HHEM
    ft_probs = hhem_probs
    ft_available = False
    if os.path.exists(args.ft_preds):
        try:
            ft_df = pd.read_csv(args.ft_preds)
            if "predicted_prob" in ft_df.columns:
                ft_probs = ft_df["predicted_prob"].values
                ft_available = True
                logger.info("Using fine-tuned DeBERTa probabilities for policy.")
        except Exception:
            pass

    if not ft_available:
        logger.info("Fine-tuned predictions not available. Using HHEM scores for policy.")

    # The FT model is extremely bimodal (probs near 0 or 1) — no graduated policy
    # is possible with it alone. HHEM has continuous scores but weak discrimination.
    # Solution: use ENSEMBLE AVERAGE which combines HHEM's continuous distribution
    # with FT's strong discrimination, giving a well-spread, well-calibrated score.
    if ft_available:
        mid_range = np.sum((ft_probs >= 0.1) & (ft_probs <= 0.9))
        logger.info(f"Fine-tuned model mid-range predictions (0.1-0.9): {mid_range}/{len(ft_probs)}")
        # Ensemble average for policy
        ensemble_probs = (hhem_probs + ft_probs) / 2
        probs = ensemble_probs
        logger.info("Using ensemble average (HHEM + FT) / 2 for decision policy — "
                     "combines HHEM's continuous scores with FT's accuracy.")
    else:
        probs = hhem_probs

    tolerances = {
        "STRICT (≤3% hallucinated)": 0.03,
        "MODERATE (≤5% hallucinated)": 0.05,
        "RELAXED (≤10% hallucinated)": 0.10,
    }

    results = {}
    logger.info("\n=== 3-TIER DECISION POLICY ===")
    logger.info(f"{'Tolerance':<30} {'LowT':>6} {'HighT':>6} {'Cov%':>6} {'Ans%':>6} {'Cav%':>6} {'Abs%':>6} {'HallRate':>10} {'FalseAbs':>10}")
    logger.info("-" * 90)

    for tol_name, max_rate in tolerances.items():
        low_t, high_t = find_thresholds_for_tolerance(y_true, probs, max_rate)
        m = compute_policy_metrics(y_true, probs, low_t, high_t)
        results[tol_name] = {
            "low_threshold": low_t,
            "high_threshold": high_t,
            "max_halluc_rate": max_rate,
            **m,
        }
        logger.info(f"{tol_name:<30} {low_t:>6.2f} {high_t:>6.2f} "
                    f"{m['coverage']:>6.1%} {m['pct_answer']:>6.1%} {m['pct_caveat']:>6.1%} "
                    f"{m['pct_abstain']:>6.1%} {m['halluc_rate_in_answers']:>10.3%} "
                    f"{m['false_abstention_rate']:>10.1%}")

    # Save
    with open(args.output_metrics, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {args.output_metrics}")

    # ---- FIGURES ----

    # 1. Stacked bar chart: answer/caveats/abstain across tolerances
    fig, ax = plt.subplots(figsize=(10, 6))
    tol_labels = list(results.keys())
    pct_answer = [results[t]["pct_answer"] for t in tol_labels]
    pct_caveat = [results[t]["pct_caveat"] for t in tol_labels]
    pct_abstain = [results[t]["pct_abstain"] for t in tol_labels]
    x = np.arange(len(tol_labels))

    bars1 = ax.bar(x, pct_answer, label="ANSWER", color="seagreen", alpha=0.85)
    bars2 = ax.bar(x, pct_caveat, bottom=pct_answer, label="CAVEAT", color="gold", alpha=0.85)
    bars3 = ax.bar(x, pct_abstain, bottom=np.array(pct_answer)+np.array(pct_caveat), label="ABSTAIN", color="tomato", alpha=0.85)

    for bar, val in zip(bars1, pct_answer):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.0%}", ha="center", va="center", fontsize=10, fontweight="bold")
    for bar, bot, val in zip(bars2, pct_answer, pct_caveat):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f"{val:.0%}", ha="center", va="center", fontsize=10)
    for bar, bot, val in zip(bars3, np.array(pct_answer)+np.array(pct_caveat), pct_abstain):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f"{val:.0%}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.split("(")[0].strip() for t in tol_labels], fontsize=10)
    ax.set_ylabel("Fraction of Examples", fontsize=12)
    ax.set_title("3-Tier Decision Policy: Answer / Caveat / Abstain", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "decision_policy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    reliability_diagram(y_true, hhem_probs, ax=axes[0], title="HHEM Calibration (1−score)")
    reliability_diagram(y_true, ft_probs, ax=axes[1], title="Fine-tuned DeBERTa Calibration")
    plt.suptitle("Reliability Diagrams: Predicted vs Actual Hallucination Rate", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "confidence_calibration.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Coverage-precision trade-off
    thresholds_sweep = np.arange(0.05, 0.96, 0.02)
    coverages, precisions = [], []
    for t in thresholds_sweep:
        served = probs <= t
        if served.sum() == 0:
            continue
        cov = served.mean()
        prec = 1.0 - y_true[served].mean()
        coverages.append(cov)
        precisions.append(prec)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, precisions, color="steelblue", linewidth=2)
    ax.fill_between(coverages, precisions, alpha=0.15, color="steelblue")

    for tol_name, m in results.items():
        ax.scatter([m["coverage"]], [m["precision_of_answers"]], s=100, zorder=5,
                   label=f"{tol_name.split('(')[0].strip()}: cov={m['coverage']:.0%}")

    ax.set_xlabel("Coverage (fraction served)", fontsize=12)
    ax.set_ylabel("Precision (1 - halluc rate)", fontsize=12)
    ax.set_title("Coverage-Precision Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "coverage_precision_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved: decision_policy.png, confidence_calibration.png, coverage_precision_tradeoff.png")
    logger.info("decision_policy.py complete.")


if __name__ == "__main__":
    main()
