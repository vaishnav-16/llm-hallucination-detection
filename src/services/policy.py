"""Decision policy — 3-tier confidence-based ANSWER/CAVEAT/ABSTAIN framework."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_policy_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> dict:
    """Compute metrics for a 3-tier decision policy.

    Args:
        y_true: True labels (0=faithful, 1=hallucinated).
        probs: Hallucination probability (high = more hallucinated).
        low_thresh: Below this → ANSWER (confident faithful).
        high_thresh: Above this → ABSTAIN (confident hallucinated).
    """
    answer_mask = probs <= low_thresh
    abstain_mask = probs > high_thresh
    caveat_mask = ~answer_mask & ~abstain_mask

    served = answer_mask
    halluc_rate_served = y_true[served].mean() if served.sum() > 0 else 0.0
    coverage = (answer_mask | caveat_mask).mean()
    precision_served = 1.0 - halluc_rate_served

    faithful_mask = y_true == 0
    false_abstain_rate = abstain_mask[faithful_mask].mean() if faithful_mask.sum() > 0 else 0.0

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


def find_thresholds_for_tolerance(
    y_true: np.ndarray,
    probs: np.ndarray,
    max_halluc_rate: float,
) -> tuple:
    """Find optimal thresholds for a given hallucination tolerance.

    Sweeps for the highest low_thresh where halluc_rate_in_answers <= max_halluc_rate.
    Returns (low_thresh, high_thresh).
    """
    best_low = 0.05
    best_coverage = 0.0

    for low in np.arange(0.95, 0.04, -0.01):
        answer_mask = probs <= low
        if answer_mask.sum() == 0:
            continue
        rate = y_true[answer_mask].mean()
        coverage = answer_mask.mean()
        if rate <= max_halluc_rate and coverage > best_coverage:
            best_low = float(low)
            best_coverage = coverage

    high = min(0.95, best_low + 0.15)
    return float(best_low), float(high)


DEFAULT_TOLERANCES = {
    "STRICT (<=3% hallucinated)": 0.03,
    "MODERATE (<=5% hallucinated)": 0.05,
    "RELAXED (<=10% hallucinated)": 0.10,
}


def evaluate_policies(
    y_true: np.ndarray,
    probs: np.ndarray,
    tolerances: dict = None,
) -> dict:
    """Evaluate 3-tier policy at multiple tolerance levels."""
    if tolerances is None:
        tolerances = DEFAULT_TOLERANCES

    results = {}
    for name, max_rate in tolerances.items():
        low_t, high_t = find_thresholds_for_tolerance(y_true, probs, max_rate)
        m = compute_policy_metrics(y_true, probs, low_t, high_t)
        results[name] = {
            "low_threshold": low_t,
            "high_threshold": high_t,
            "max_halluc_rate": max_rate,
            **m,
        }
        logger.info(f"{name:<30} low={low_t:.2f} high={high_t:.2f} "
                     f"coverage={m['coverage']:.1%} halluc_rate={m['halluc_rate_in_answers']:.3%}")

    return results
