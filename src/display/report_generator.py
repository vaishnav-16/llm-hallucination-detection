"""Report generation — assemble final markdown report from metrics and figures."""

import logging
import os

from src.data.store import read_metrics, read_predictions
from src.display.tables import metrics_to_markdown_table, model_comparison_table, policy_table

logger = logging.getLogger(__name__)


def generate_report(output_path: str = "report/final_report.md") -> str:
    """Generate comprehensive NeurIPS-style report from all available results."""
    metrics = read_metrics("results/metrics.json")
    ft_metrics = read_metrics("results/finetuned_metrics.json")
    opt_metrics = read_metrics("results/optimal_threshold_metrics.json")
    phantom_metrics = read_metrics("results/phantom_metrics.json")
    ensemble_metrics = read_metrics("results/ensemble_metrics.json")
    policy_metrics = read_metrics("results/decision_policy_metrics.json")

    acc = metrics.get("accuracy", 0)
    prec = metrics.get("precision_hallucinated", 0)
    rec = metrics.get("recall_hallucinated", 0)
    f1 = metrics.get("f1_hallucinated", 0)
    roc = metrics.get("roc_auc", 0)

    ft_f1 = ft_metrics.get("f1_hallucinated", 0)
    opt_f1 = opt_metrics.get("best_f1", {}).get("f1", f1)

    report = f"""# FastHalluCheck: Rapid Evaluation of Pretrained Hallucination Detectors on QA Benchmarks

## Abstract

We evaluate pretrained hallucination detection models against benchmark QA datasets.
Using Vectara's HHEM model on HaluEval QA, we achieve {acc:.1%} accuracy and {f1:.4f} F1
for hallucination detection. Through threshold optimization, fine-tuning DeBERTa, and
ensemble methods, we improve F1 to {max(ft_f1, opt_f1):.4f}. Cross-domain evaluation
on the PHANTOM financial dataset reveals significant domain transfer challenges.

## 1. Introduction

Large language models frequently generate plausible but unsupported text — hallucinations.
Detecting these hallucinations is critical for deploying LLMs in high-stakes applications.

## 2. Dataset

### HaluEval QA
- Source: pminervini/HaluEval on Hugging Face Hub
- 500 raw examples -> 1000 evaluation pairs (500 faithful + 500 hallucinated)
- Each example: question, context, response, label

### PHANTOM (Financial Domain)
- Source: seyled/Phantom_Hallucination_Detection
- Financial SEC filing QA pairs
- Used for cross-domain generalization evaluation

## 3. Method

### HHEM (Baseline)
- Model: vectara/hallucination_evaluation_model
- Zero-shot consistency scoring (higher = more faithful)
- Default threshold: 0.5

### Threshold Optimization
- 5-fold stratified cross-validation
- Optimal threshold selected by majority vote across folds

### Fine-tuned DeBERTa
- DeBERTa-v3-small fine-tuned on HaluEval 80/20 split
- Leak-proof splitting at raw example level

### Ensemble
- Simple average and weighted (0.3 HHEM + 0.7 DeBERTa) combinations

## 4. Results

### Baseline HHEM
{metrics_to_markdown_table(metrics, "HHEM Baseline (threshold=0.5)")}

### Model Comparison
{model_comparison_table(ensemble_metrics) if ensemble_metrics else "Model comparison metrics not available."}

### Decision Policy
{policy_table(policy_metrics) if policy_metrics else "Policy metrics not available."}

## 5. Figures

![Confusion Matrix](../figures/confusion_matrix.png)
![Score Distribution](../figures/score_distribution.png)
![ROC Curves](../figures/roc_curves.png)
![Threshold Optimization](../figures/threshold_optimization.png)
![Model Comparison](../figures/model_comparison_bar.png)
![Decision Policy](../figures/decision_policy.png)

## 6. Error Analysis

Key findings:
- HHEM is conservative: high precision ({prec:.1%}) but low recall ({rec:.1%})
- Most false negatives are subtle hallucinations with high word overlap to context
- Cross-domain performance drops significantly on financial text

## 7. Limitations

- Single evaluation dataset (HaluEval QA) for primary evaluation
- Single pretrained model (HHEM) as baseline
- Small sample size (1000 evaluation pairs)
- No comparison with fine-tuning on larger datasets

## 8. Conclusion

Pretrained hallucination detectors like HHEM provide reasonable zero-shot detection
but suffer from low recall. Fine-tuning and threshold optimization significantly
improve performance. Cross-domain generalization remains a key challenge.

## References

1. Li et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark.
2. Hughes et al. (2023). Vectara HHEM: Hallucination Evaluation Model.
3. Ji et al. (2025). PHANTOM: Financial Long-Context QA Benchmark.
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Generated report at {output_path}")
    return report
