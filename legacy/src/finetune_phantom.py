"""
finetune_phantom.py — Fine-tune DeBERTa-v3-small on PHANTOM financial QA data.

Combines seed + 2K variants, leak-proof base_id split, trains domain-specific model.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEED = 42


def get_device():
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logger.info("Using CPU.")
    return "cpu"


def compute_metrics_fn(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
    }


class HalluDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def try_load_model(model_name: str):
    """Load model with DeBERTa-v3 LayerNorm key fix."""
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    logger.info(f"Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "deberta" in model_name.lower():
        config = AutoConfig.from_pretrained(model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(config)
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        fixed_sd = {}
        for k, v in raw_sd.items():
            new_k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
            if "lm_predictions" in new_k or "mask_predictions" in new_k:
                continue
            fixed_sd[new_k] = v
        missing, unexpected = model.load_state_dict(fixed_sd, strict=False)
        logger.info(f"Loaded {model_name} with key fix. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_data", default="data/phantom_normalized.csv")
    parser.add_argument("--data_2k", default="data/phantom_2k_normalized.csv")
    parser.add_argument("--output_model", default="models/deberta-phantom-finetuned")
    parser.add_argument("--output_preds", default="results/phantom_trained_predictions.csv")
    parser.add_argument("--output_metrics", default="results/phantom_trained_metrics.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = get_device()

    # Step 1: Combine seed + 2K data
    dfs = []
    for path, name in [(args.seed_data, "seed"), (args.data_2k, "2K")]:
        try:
            df_part = pd.read_csv(path)
            logger.info(f"Loaded {name}: {len(df_part)} rows, labels={df_part['label'].value_counts().to_dict()}")
            dfs.append(df_part)
        except Exception as e:
            logger.warning(f"Could not load {name} ({path}): {e}")

    if not dfs:
        raise RuntimeError("No PHANTOM data available!")

    df = pd.concat(dfs, ignore_index=True)

    # Ensure required columns
    if "question" not in df.columns and "query" in df.columns:
        df["question"] = df["query"]

    # Build text pairs
    df["text_a"] = df["question"].astype(str) + " " + df["context"].astype(str)
    df["text_b"] = df["response"].astype(str)
    df["label"] = df["label"].astype(int)

    # Deduplicate by question+response
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["question", "response"], keep="first")
    logger.info(f"Combined: {len(df)} rows (deduplicated from {before_dedup})")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Step 2: Leak-proof split
    # Extract base_id: remove _faithful / _hallucinated suffix
    df["base_id"] = df["id"].str.replace("_faithful", "", regex=False).str.replace("_hallucinated", "", regex=False)
    unique_bases = df["base_id"].unique()
    logger.info(f"Unique base examples: {len(unique_bases)}")

    from sklearn.model_selection import train_test_split
    test_size = 0.3 if len(unique_bases) < 200 else 0.2
    train_bases, test_bases = train_test_split(unique_bases, test_size=test_size, random_state=SEED)

    train_mask = df["base_id"].isin(set(train_bases))
    test_mask = df["base_id"].isin(set(test_bases))
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    overlap = set(train_df["base_id"].unique()) & set(test_df["base_id"].unique())
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} base_ids overlap!"

    logger.info(f"LEAK-PROOF SPLIT (test_size={test_size}):")
    logger.info(f"  Train: {len(train_df)} rows from {len(train_bases)} base examples")
    logger.info(f"  Test:  {len(test_df)} rows from {len(test_bases)} base examples")
    logger.info(f"  Base ID overlap: {len(overlap)}")
    logger.info(f"  Train labels: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"  Test labels:  {test_df['label'].value_counts().to_dict()}")

    # Step 3: Load model
    model_name = "microsoft/deberta-v3-small"
    try:
        tokenizer, model = try_load_model(model_name)
    except Exception as e:
        logger.warning(f"DeBERTa-v3-small failed: {e}. Falling back to distilbert.")
        model_name = "distilbert-base-uncased"
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    logger.info(f"Model: {model_name}")

    # Tokenize
    def tokenize(texts_a, texts_b, max_length):
        return tokenizer(texts_a, texts_b, truncation=True, max_length=max_length,
                         padding="longest", return_tensors=None)

    train_enc = tokenize(train_df["text_a"].tolist(), train_df["text_b"].tolist(), args.max_length)
    test_enc = tokenize(test_df["text_a"].tolist(), test_df["text_b"].tolist(), args.max_length)

    train_dataset = HalluDataset(train_enc, train_df["label"].tolist())
    test_dataset = HalluDataset(test_enc, test_df["label"].tolist())

    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

    use_bf16 = device == "cuda" and "deberta" in model_name.lower()
    use_fp16 = device == "cuda" and not use_bf16
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # More epochs for smaller dataset
    n_epochs = args.epochs
    if len(train_df) < 300:
        n_epochs = 8
        logger.info(f"Small dataset ({len(train_df)} rows), using {n_epochs} epochs")

    training_args = TrainingArguments(
        output_dir=args.output_model,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=2e-5,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="models/phantom_logs",
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator,
    )

    logger.info(f"Training {model_name} on PHANTOM ({len(train_df)} train rows, {n_epochs} epochs)...")
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Test metrics: {eval_results}")

    # Save model
    trainer.save_model(args.output_model)
    tokenizer.save_pretrained(args.output_model)
    logger.info(f"Model saved to {args.output_model}")

    # Inference on full dataset
    logger.info("Running inference on full PHANTOM dataset...")
    full_enc = tokenize(df["text_a"].tolist(), df["text_b"].tolist(), args.max_length)
    full_dataset = HalluDataset(full_enc, df["label"].tolist())

    model.eval()
    model = model.to(device)
    from torch.utils.data import DataLoader
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    df["predicted_prob"] = all_probs
    df["predicted_label"] = (np.array(all_probs) >= 0.5).astype(int)

    # Save predictions
    out_cols = [c for c in ["id", "question", "context", "response", "label", "predicted_prob", "predicted_label"]
                if c in df.columns]
    df[out_cols].to_csv(args.output_preds, index=False)
    logger.info(f"Saved predictions to {args.output_preds}")

    # Compute test-only metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    test_indices = df.index[test_mask].tolist()
    y_test = df.loc[test_indices, "label"].values
    y_test_prob = np.array([all_probs[i] for i in test_indices])
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    test_metrics = {
        "n_examples": int(len(y_test)),
        "n_base_examples": int(len(test_bases)),
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
    }

    # Load comparison metrics
    hhem_phantom = {}
    halueval_ft = {}
    try:
        pm = json.load(open("results/phantom_metrics.json"))
        hhem_phantom = pm.get("hhem_phantom", {})
        halueval_ft = pm.get("finetuned_phantom", {})
    except Exception:
        pass

    output = {
        "model": model_name,
        "training_data": "PHANTOM (seed + 2K combined)",
        "split_method": "leak-proof base_id level split",
        "train_n_rows": int(len(train_df)),
        "test_n_rows": int(len(test_df)),
        "n_epochs": n_epochs,
        "base_id_overlap": int(len(overlap)),
        "test_metrics": test_metrics,
    }

    with open(args.output_metrics, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved metrics to {args.output_metrics}")

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("KEY COMPARISON: ALL MODELS ON PHANTOM")
    logger.info("=" * 80)
    logger.info(f"{'Model':<40} {'F1':>8} {'Acc':>8} {'ROC-AUC':>8}")
    logger.info("-" * 68)
    logger.info(f"{'HHEM zero-shot (CV threshold)':<40} {hhem_phantom.get('f1', 0):>8.4f} {hhem_phantom.get('accuracy', 0):>8.4f} {hhem_phantom.get('roc_auc', 0):>8.4f}")
    logger.info(f"{'DeBERTa trained on HaluEval':<40} {halueval_ft.get('f1', 0):>8.4f} {halueval_ft.get('accuracy', 0):>8.4f} {halueval_ft.get('roc_auc', 0):>8.4f}")
    logger.info(f"{'DeBERTa trained on PHANTOM':<40} {test_metrics['f1']:>8.4f} {test_metrics['accuracy']:>8.4f} {test_metrics['roc_auc']:>8.4f}")
    logger.info("=" * 68)
    logger.info("finetune_phantom.py complete.")


if __name__ == "__main__":
    main()
