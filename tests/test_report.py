"""The Discussion is built from metrics only: it renders with partial inputs and quotes the numbers it is given."""

import pandas as pd

from src.display.report import discussion


def _m(det, strat, sl, group, value, lo=None, hi=None, n=400):
    return {"detector": det, "strategy": strat, "slice": sl, "group": group, "metric": "auroc", "value": value,
            "ci_lo": value - 0.01 if lo is None else lo, "ci_hi": value + 0.01 if hi is None else hi, "n": n}


def _comp(det, strat, sl, group, diff, lo, hi, b="modernbert_answeronly_s42:truncate-w7168__ablate"):
    return {"detector": det, "slice": sl, "a": strat, "b": b, "group": group, "n": 400, "metric": "auroc",
            "diff": diff, "ci_lo": lo, "ci_hi": hi, "p_value": 0.01}


def test_discussion_renders_from_metrics_and_quotes_them():
    rows = [
        _m("answer_tfidf", "none", "phantom_grid_full", "all", 0.957), _m("answer_tfidf", "none", "phantom_grid_full", "hard", 0.888),
        _m("modernbert_answeronly_s42", "truncate-w7168__ablate", "phantom_grid_sub", "all", 0.993),
        _m("modernbert_answeronly_s42", "truncate-w7168__ablate", "phantom_grid_sub", "evid>=1536", 0.991),
        _m("modernbert_nli_ablate", "truncate-w1536__ablate", "phantom_seed_test", "all", 0.244),
    ]
    for det, t0, t30, c30 in (("modernbert_nli", 0.741, 0.481, 0.747), ("modernbert_ft_s42", 1.0, 0.956, 0.982)):
        w = "1536" if det == "modernbert_nli" else "7168"
        rows += [_m(det, f"truncate-w{w}", "phantom_grid_sub", "len=500", t0), _m(det, f"truncate-w{w}", "phantom_grid_sub", "len=30000", t30),
                 _m(det, f"chunk-w{w}-o1024-min_phalluc", "phantom_grid_sub", "len=500", t0),
                 _m(det, f"chunk-w{w}-o1024-min_phalluc", "phantom_grid_sub", "len=30000", c30),
                 _m(det, f"truncate-w{w}", "phantom_grid_sub", "all", (t0 + t30) / 2),
                 _m(det, f"chunk-w{w}-o1024-min_phalluc", "phantom_grid_sub", "hard", 0.70 if det == "modernbert_nli" else 0.97),
                 _m(det, f"truncate-w{w}", "phantom_grid_sub", f"evid>={w}", 0.43 if det == "modernbert_nli" else 0.95)]
    comp = pd.DataFrame([
        _comp("modernbert_ft_s42", "chunk-w7168-o1024-min_phalluc", "phantom_grid_sub", "all", -0.002, -0.006, 0.002),
        _comp("modernbert_ft_s42", "truncate-w7168", "phantom_grid_sub", "len=30000|hard", -0.129, -0.157, -0.103),
        _comp("modernbert_ft_s42", "chunk-w7168-o1024-min_phalluc", "ragtruth_test", "all", 0.085, 0.070, 0.102),
        _comp("modernbert_nli", "chunk-w1536-o1024-min_phalluc", "phantom_grid_sub", "len=30000", 0.266, 0.24, 0.29,
              b="modernbert_nli:truncate-w1536"),
    ])
    text = discussion(pd.DataFrame(rows), comp, None)
    assert "0.957" in text and "0.993" in text and "0.244" in text  # quoted from the inputs
    assert "+0.085" in text and "-0.129" in text
    assert "1 of 1 chunk/retrieve strategies beat their own truncate" in text
    assert "Only one training seed" in text
    assert "inverted" in text  # 0.244 < 0.5 is called out, not hidden


def test_discussion_with_no_comparisons_or_seeds_still_renders():
    long = pd.DataFrame([_m("answer_tfidf", "none", "phantom_grid_full", "all", 0.957)])
    empty = pd.DataFrame(columns=["detector", "slice", "a", "b", "group", "n", "metric", "diff", "ci_lo", "ci_hi", "p_value"])
    text = discussion(long, empty, pd.DataFrame())
    assert text.startswith("## Discussion") and "n/a" in text
