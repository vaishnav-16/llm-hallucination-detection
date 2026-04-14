"""
run_phantom_eval.py — Evaluate HHEM and fine-tuned DeBERTa on PHANTOM dataset.

Reads data/phantom_normalized.csv, runs both models, generates cross-domain comparison.
"""

import argparse
import json
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42


def get_device():
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    return "cpu"


def run_hhem_on_phantom(df: pd.DataFrame, device: str, optimal_threshold: float) -> pd.DataFrame:
    """Run HHEM on PHANTOM data using direct snapshot loading."""
    import glob
    snapshot_patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/models--vectara--hallucination_evaluation_model/snapshots/*/modeling_hhem_v2.py"),
    ]
    snapshot_dir = None
    for pat in snapshot_patterns:
        files = glob.glob(pat)
        if files:
            snapshot_dir = os.path.dirname(files[0])
            break

    if snapshot_dir is None:
        logger.error("HHEM snapshot not found. Skipping HHEM on PHANTOM.")
        df["hhem_score"] = 0.5
        df["hhem_predicted"] = 0
        return df

    import tempfile
    import shutil
    from safetensors.torch import load_file

    tmp_pkg_dir = tempfile.mkdtemp(prefix="hhem_pkg_phantom_")
    pkg_name = "hhem_pkg_phantom"
    pkg_dir = os.path.join(tmp_pkg_dir, pkg_name)
    os.makedirs(pkg_dir)

    for fname in ["modeling_hhem_v2.py", "configuration_hhem_v2.py"]:
        src = os.path.join(snapshot_dir, fname)
        with open(src) as f:
            content = f.read()
        content = content.replace("from .configuration_hhem_v2", f"from {pkg_name}.configuration_hhem_v2")
        with open(os.path.join(pkg_dir, fname), "w") as f:
            f.write(content)
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write("")

    if tmp_pkg_dir not in sys.path:
        sys.path.insert(0, tmp_pkg_dir)

    from hhem_pkg_phantom.modeling_hhem_v2 import HHEMv2ForSequenceClassification
    from hhem_pkg_phantom.configuration_hhem_v2 import HHEMv2Config

    config = HHEMv2Config()
    model = HHEMv2ForSequenceClassification(config)
    weights_path = os.path.join(snapshot_dir, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.t5 = model.t5.to(device)

    pairs = [(str(row["question"]) + " " + str(row["context"]), str(row["response"])) for _, row in df.iterrows()]
    batch_size = 32 if device == "cuda" else 8
    scores = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(pairs), batch_size), desc="HHEM on PHANTOM"):
        batch = pairs[i:i+batch_size]
        try:
            s = model.predict(batch)
            scores.extend(s.cpu().numpy().tolist())
        except Exception as e:
            logger.warning(f"Batch {i} error: {e}")
            scores.extend([0.5] * len(batch))

    df = df.copy()
    df["hhem_score"] = scores
    df["hhem_predicted"] = (np.array(scores) <= optimal_threshold).astype(int)
    return df


def run_finetuned_on_phantom(df: pd.DataFrame, model_dir: str, device: str) -> pd.DataFrame:
    """Run fine-tuned DeBERTa on PHANTOM data."""
    if not os.path.exists(model_dir):
        logger.warning(f"Model dir {model_dir} not found. Skipping fine-tuned eval.")
        df["finetuned_prob"] = 0.5
        df["finetuned_predicted"] = 0
        return df

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader

    logger.info(f"Loading fine-tuned model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model = model.to(device)

    texts_a = (df["question"].astype(str) + " " + df["context"].astype(str)).tolist()
    texts_b = df["response"].astype(str).tolist()

    all_probs = []
    batch_size = 16
    for i in range(0, len(texts_a), batch_size):
        batch_a = texts_a[i:i+batch_size]
        batch_b = texts_b[i:i+batch_size]
        try:
            enc = tokenizer(batch_a, batch_b, truncation=True, max_length=512, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
                probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
        except Exception as e:
            logger.warning(f"Batch {i} error: {e}")
            all_probs.extend([0.5] * len(batch_a))

    df = df.copy()
    df["finetuned_prob"] = all_probs
    df["finetuned_predicted"] = (np.array(all_probs) >= 0.5).astype(int)
    return df


def compute_metrics(y_true, y_pred, y_prob=None):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["roc_auc"] = None
    return m


def plot_cross_domain_comparison(metrics_dict: dict, figures_dir: str):
    """Grouped bar chart: models x datasets x metrics."""
    models = ["HHEM (τ=opt)", "Fine-tuned DeBERTa"]
    datasets = ["HaluEval", "PHANTOM"]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    x = np.arange(len(datasets))
    width = 0.35
    colors = ["steelblue", "darkorange"]

    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[i]
        for j, model in enumerate(models):
            vals = [metrics_dict.get(f"{model}_{ds}", {}).get(metric, 0) for ds in datasets]
            bars = ax.bar(x + j*width - width/2, vals, width, label=model, color=colors[j], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.1)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label if i == 0 else "")
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Cross-Domain Evaluation: HaluEval (in-domain) vs PHANTOM (financial)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cross_domain_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved cross_domain_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/phantom_normalized.csv")
    parser.add_argument("--output_hhem", default="results/phantom_hhem_predictions.csv")
    parser.add_argument("--output_finetuned", default="results/phantom_finetuned_predictions.csv")
    parser.add_argument("--output_metrics", default="results/phantom_metrics.json")
    parser.add_argument("--model_dir", default="models/deberta-hallucination-detector")
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    device = get_device()

    logger.info(f"Loading PHANTOM data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows, label dist: {df['label'].value_counts().to_dict()}")

    # Load optimal threshold from Phase 2
    optimal_threshold = 0.5  # fallback
    try:
        with open("results/optimal_threshold_metrics.json") as f:
            thresh_data = json.load(f)
        optimal_threshold = thresh_data["best_f1"]["threshold"]
        logger.info(f"Using optimal threshold: {optimal_threshold}")
    except Exception:
        logger.info("Using default threshold=0.5")

    # HHEM on PHANTOM
    logger.info("\n--- Running HHEM on PHANTOM ---")
    df_hhem = run_hhem_on_phantom(df, device, optimal_threshold)
    df_hhem.to_csv(args.output_hhem, index=False)
    hhem_phantom_metrics = compute_metrics(df_hhem["label"].values, df_hhem["hhem_predicted"].values,
                                           1.0 - df_hhem["hhem_score"].values)
    logger.info(f"HHEM on PHANTOM: {hhem_phantom_metrics}")

    # Fine-tuned on PHANTOM
    logger.info("\n--- Running Fine-tuned DeBERTa on PHANTOM ---")
    df_ft = run_finetuned_on_phantom(df, args.model_dir, device)
    df_ft.to_csv(args.output_finetuned, index=False)
    ft_phantom_metrics = compute_metrics(df_ft["label"].values, df_ft["finetuned_predicted"].values,
                                         df_ft["finetuned_prob"].values)
    logger.info(f"Fine-tuned on PHANTOM: {ft_phantom_metrics}")

    # Load HaluEval metrics for comparison
    halueval_hhem = {"accuracy": 0.732, "precision": 0.8816, "recall": 0.536, "f1": 0.6667}
    halueval_ft = {}
    try:
        with open("results/finetuned_metrics.json") as f:
            ft_data = json.load(f)
        halueval_ft = {
            "accuracy": ft_data["accuracy"],
            "precision": ft_data["precision_hallucinated"],
            "recall": ft_data["recall_hallucinated"],
            "f1": ft_data["f1_hallucinated"],
        }
    except Exception:
        halueval_ft = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Load optimal HHEM metrics
    try:
        with open("results/optimal_threshold_metrics.json") as f:
            opt_data = json.load(f)
        halueval_hhem_opt = {
            "accuracy": opt_data["best_f1"]["accuracy"],
            "precision": opt_data["best_f1"]["precision"],
            "recall": opt_data["best_f1"]["recall"],
            "f1": opt_data["best_f1"]["f1"],
        }
    except Exception:
        halueval_hhem_opt = halueval_hhem

    # Save all metrics
    phantom_metrics = {
        "hhem_phantom": hhem_phantom_metrics,
        "finetuned_phantom": ft_phantom_metrics,
        "hhem_halueval_optimized": halueval_hhem_opt,
        "finetuned_halueval": halueval_ft,
        "domain_drop_hhem_f1": float(halueval_hhem_opt.get("f1", 0)) - hhem_phantom_metrics.get("f1", 0),
        "domain_drop_ft_f1": float(halueval_ft.get("f1", 0)) - ft_phantom_metrics.get("f1", 0),
    }
    with open(args.output_metrics, "w") as f:
        json.dump(phantom_metrics, f, indent=2)
    logger.info(f"Saved to {args.output_metrics}")

    # Cross-domain comparison figure
    metrics_for_plot = {
        "HHEM (τ=opt)_HaluEval": halueval_hhem_opt,
        "HHEM (τ=opt)_PHANTOM": hhem_phantom_metrics,
        "Fine-tuned DeBERTa_HaluEval": halueval_ft,
        "Fine-tuned DeBERTa_PHANTOM": ft_phantom_metrics,
    }
    plot_cross_domain_comparison(metrics_for_plot, args.figures_dir)

    logger.info("\n=== CROSS-DOMAIN COMPARISON ===")
    logger.info(f"{'Model':<25} {'Dataset':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    logger.info("-" * 60)
    for model, halueval_m, phantom_m in [
        ("HHEM (τ=opt)", halueval_hhem_opt, hhem_phantom_metrics),
        ("Fine-tuned DeBERTa", halueval_ft, ft_phantom_metrics),
    ]:
        for ds, m in [("HaluEval", halueval_m), ("PHANTOM", phantom_m)]:
            logger.info(f"{model:<25} {ds:<12} "
                        f"{m.get('accuracy',0):>6.3f} {m.get('precision',0):>6.3f} "
                        f"{m.get('recall',0):>6.3f} {m.get('f1',0):>6.3f}")

    drop_hhem = phantom_metrics["domain_drop_hhem_f1"]
    drop_ft = phantom_metrics["domain_drop_ft_f1"]
    logger.info(f"\nF1 drop (HaluEval→PHANTOM): HHEM={drop_hhem:.3f}, Fine-tuned={drop_ft:.3f}")
    logger.info("run_phantom_eval.py complete.")


if __name__ == "__main__":
    main()
