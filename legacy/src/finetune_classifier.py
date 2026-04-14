"""
finetune_classifier.py — Fine-tune DeBERTa-v3-small as a binary hallucination classifier.

Reads data/halueval_enriched.csv, does LEAK-PROOF 80/20 split at the raw-example level,
fine-tunes, evaluates on test set, runs inference on full 1000 examples.

CRITICAL: The split is done by base_id (raw example), NOT by row. Each raw example
produces two rows (faithful + hallucinated) sharing the same (question, context). Both
rows go into the same split to prevent the model from memorizing context representations.
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


def try_load_model(model_name: str, device: str):
    """Try loading model, return (tokenizer, model) or raise."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    logger.info(f"Attempting to load: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config)

    # DeBERTa-v3 checkpoints use gamma/beta instead of weight/bias for LayerNorm.
    # Load raw state dict and rename keys before loading into model.
    if "deberta" in model_name.lower():
        from huggingface_hub import hf_hub_download
        import torch as _torch
        ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        raw_sd = _torch.load(ckpt_path, map_location="cpu", weights_only=True)
        fixed_sd = {}
        for k, v in raw_sd.items():
            new_k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
            # Skip LM head keys not present in classification model
            if "lm_predictions" in new_k or "mask_predictions" in new_k:
                continue
            fixed_sd[new_k] = v
        missing, unexpected = model.load_state_dict(fixed_sd, strict=False)
        logger.info(f"Loaded {model_name} with key fix. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            logger.info(f"  Missing keys (expected for new classifier head): {missing}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/halueval_enriched.csv")
    parser.add_argument("--output_model", default="models/deberta-hallucination-detector")
    parser.add_argument("--output_preds", default="results/finetuned_predictions.csv")
    parser.add_argument("--output_metrics", default="results/finetuned_metrics.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = get_device()

    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows.")

    # Build text pairs: (question + context, response)
    df["text_a"] = df["question"].astype(str) + " " + df["context"].astype(str)
    df["text_b"] = df["response"].astype(str)
    df["label"] = df["label"].astype(int)

    # ===== LEAK-PROOF SPLIT: split at the raw-example (base_id) level =====
    # Each raw example produces two rows: {idx}_faithful and {idx}_hallucinated.
    # Both share the same (question, context). Both MUST go into the same split.
    df["base_id"] = df["id"].str.replace("_faithful", "", regex=False).str.replace("_hallucinated", "", regex=False)
    unique_bases = df["base_id"].unique()
    logger.info(f"Unique base examples: {len(unique_bases)}")

    from sklearn.model_selection import train_test_split
    train_bases, test_bases = train_test_split(
        unique_bases, test_size=0.2, random_state=SEED
    )

    train_mask = df["base_id"].isin(set(train_bases))
    test_mask = df["base_id"].isin(set(test_bases))
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # Verify zero overlap
    train_base_set = set(train_df["base_id"].unique())
    test_base_set = set(test_df["base_id"].unique())
    overlap = train_base_set & test_base_set
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} base_ids in both splits!"

    logger.info(f"LEAK-PROOF SPLIT:")
    logger.info(f"  Train: {len(train_df)} rows from {len(train_bases)} base examples")
    logger.info(f"  Test:  {len(test_df)} rows from {len(test_bases)} base examples")
    logger.info(f"  Base ID overlap: {len(overlap)} (must be 0)")
    logger.info(f"  Train label dist: {train_df['label'].value_counts().to_dict()}")
    logger.info(f"  Test label dist:  {test_df['label'].value_counts().to_dict()}")

    # Model candidates: DeBERTa-v3-small first (same family as HHEM)
    model_candidates = [
        ("microsoft/deberta-v3-small", 4, 2, 512),    # bs=4, ga=2, max_len=512
        ("microsoft/deberta-v3-small", 2, 4, 384),    # OOM fallback
        ("distilbert-base-uncased", 8, 1, 512),       # final fallback
    ]

    tokenizer = None
    model = None
    chosen_model = None
    chosen_bs = args.batch_size
    chosen_ga = 1
    chosen_max_length = args.max_length

    for model_name, bs, ga, ml in model_candidates:
        try:
            tokenizer, model = try_load_model(model_name, device)
            chosen_model = model_name
            chosen_bs = bs
            chosen_ga = ga
            chosen_max_length = ml
            logger.info(f"Loaded model: {model_name} (bs={bs}, ga={ga}, max_len={ml})")
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")

    if model is None:
        raise RuntimeError("All model candidates failed to load.")

    logger.info(f"Fine-tuning {chosen_model}...")

    # Tokenize — use padding="longest" for efficiency (Fix 5)
    def tokenize(texts_a, texts_b, max_length):
        return tokenizer(
            texts_a, texts_b,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors=None,
        )

    logger.info("Tokenizing train set...")
    train_enc = tokenize(
        train_df["text_a"].tolist(),
        train_df["text_b"].tolist(),
        chosen_max_length,
    )
    logger.info("Tokenizing test set...")
    test_enc = tokenize(
        test_df["text_a"].tolist(),
        test_df["text_b"].tolist(),
        chosen_max_length,
    )

    train_dataset = HalluDataset(train_enc, train_df["label"].tolist())
    test_dataset = HalluDataset(test_enc, test_df["label"].tolist())

    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

    # DeBERTa-v3 is incompatible with fp16 gradient scaling; use bf16 instead
    use_bf16 = device == "cuda" and "deberta" in chosen_model.lower()
    use_fp16 = device == "cuda" and not use_bf16
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def try_train(bs, ga):
        training_args = TrainingArguments(
            output_dir=args.output_model,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs * 2,
            gradient_accumulation_steps=ga,
            warmup_ratio=0.1,
            weight_decay=0.01,
            learning_rate=2e-5,
            fp16=use_fp16,
            bf16=use_bf16,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir="models/logs",
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
        trainer.train()
        return trainer

    trainer = None
    try:
        logger.info(f"Training with batch_size={chosen_bs}, grad_accum={chosen_ga}, max_length={chosen_max_length}...")
        trainer = try_train(chosen_bs, chosen_ga)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            # Fall back to smaller config
            chosen_bs = max(2, chosen_bs // 2)
            chosen_ga = chosen_ga * 2
            logger.warning(f"OOM! Retrying with batch_size={chosen_bs}, grad_accum={chosen_ga}...")
            try:
                trainer = try_train(chosen_bs, chosen_ga)
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    torch.cuda.empty_cache()
                    chosen_bs = 2
                    chosen_ga = 4
                    chosen_max_length = 256
                    logger.warning("OOM again! Retrying with batch_size=2, max_length=256...")
                    train_enc = tokenize(train_df["text_a"].tolist(), train_df["text_b"].tolist(), 256)
                    test_enc = tokenize(test_df["text_a"].tolist(), test_df["text_b"].tolist(), 256)
                    train_dataset = HalluDataset(train_enc, train_df["label"].tolist())
                    test_dataset = HalluDataset(test_enc, test_df["label"].tolist())
                    trainer = try_train(chosen_bs, chosen_ga)
                else:
                    raise
        else:
            raise

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    eval_results = trainer.evaluate()
    logger.info(f"Test metrics: {eval_results}")

    # Save model
    trainer.save_model(args.output_model)
    tokenizer.save_pretrained(args.output_model)
    logger.info(f"Model saved to {args.output_model}")

    # Inference on FULL 1000 examples (for ensemble/downstream use)
    logger.info("Running inference on full dataset...")
    full_enc = tokenize(df["text_a"].tolist(), df["text_b"].tolist(), chosen_max_length)
    full_dataset = HalluDataset(full_enc, df["label"].tolist())

    all_probs = []
    model.eval()
    model = model.to(device)
    from torch.utils.data import DataLoader
    data_collator_infer = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False, collate_fn=data_collator_infer)
    with torch.no_grad():
        for batch in loader:
            labels_tmp = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    df["predicted_prob"] = all_probs
    df["predicted_label"] = (np.array(all_probs) >= 0.5).astype(int)

    # Save predictions
    out_cols = ["id", "question", "context", "response", "label", "predicted_prob", "predicted_label"]
    available_cols = [c for c in out_cols if c in df.columns]
    df[available_cols].to_csv(args.output_preds, index=False)
    logger.info(f"Saved predictions to {args.output_preds}")

    # Compute metrics: TEST ONLY (primary) and FULL (reference)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    # Test-only metrics (PRIMARY — no data leakage)
    test_indices = df.index[test_mask].tolist()
    y_test_true = df.loc[test_indices, "label"].values
    y_test_prob = np.array([all_probs[i] for i in test_indices])
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    test_metrics = {
        "n_examples": int(len(y_test_true)),
        "n_base_examples": int(len(test_bases)),
        "accuracy": float(accuracy_score(y_test_true, y_test_pred)),
        "precision_hallucinated": float(precision_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)),
        "recall_hallucinated": float(recall_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)),
        "f1_hallucinated": float(f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_true, y_test_prob)),
    }

    # Full-dataset metrics (for reference, includes training data)
    y_full_true = df["label"].values
    y_full_pred = df["predicted_label"].values
    y_full_prob = np.array(all_probs)

    full_metrics = {
        "n_examples": int(len(y_full_true)),
        "note": "INCLUDES TRAINING DATA — for downstream ensemble use only, not for reporting",
        "accuracy": float(accuracy_score(y_full_true, y_full_pred)),
        "precision_hallucinated": float(precision_score(y_full_true, y_full_pred, pos_label=1, zero_division=0)),
        "recall_hallucinated": float(recall_score(y_full_true, y_full_pred, pos_label=1, zero_division=0)),
        "f1_hallucinated": float(f1_score(y_full_true, y_full_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_full_true, y_full_prob)),
    }

    output_metrics = {
        "model": chosen_model,
        "split_method": "leak-proof base_id level split",
        "train_n_rows": int(len(train_df)),
        "train_n_bases": int(len(train_bases)),
        "test_n_rows": int(len(test_df)),
        "test_n_bases": int(len(test_bases)),
        "base_id_overlap": int(len(overlap)),
        "test_metrics": test_metrics,
        "full_metrics": full_metrics,
        "trainer_eval_results": {k: v for k, v in eval_results.items()},
        # Keep top-level keys for backward compatibility with downstream scripts
        "accuracy": test_metrics["accuracy"],
        "precision_hallucinated": test_metrics["precision_hallucinated"],
        "recall_hallucinated": test_metrics["recall_hallucinated"],
        "f1_hallucinated": test_metrics["f1_hallucinated"],
        "f1_macro": test_metrics["f1_macro"],
        "roc_auc": test_metrics["roc_auc"],
    }

    with open(args.output_metrics, "w") as f:
        json.dump(output_metrics, f, indent=2)
    logger.info(f"Saved metrics to {args.output_metrics}")

    logger.info("\n=== RESULTS ===")
    logger.info(f"Model: {chosen_model}")
    logger.info(f"TEST SET ONLY (primary, n={test_metrics['n_examples']}, {test_metrics['n_base_examples']} base examples):")
    logger.info(f"  accuracy={test_metrics['accuracy']:.4f}, precision={test_metrics['precision_hallucinated']:.4f}, "
                f"recall={test_metrics['recall_hallucinated']:.4f}, f1={test_metrics['f1_hallucinated']:.4f}, "
                f"roc_auc={test_metrics['roc_auc']:.4f}")
    logger.info(f"FULL DATASET (includes training data, n={full_metrics['n_examples']}):")
    logger.info(f"  accuracy={full_metrics['accuracy']:.4f}, precision={full_metrics['precision_hallucinated']:.4f}, "
                f"recall={full_metrics['recall_hallucinated']:.4f}, f1={full_metrics['f1_hallucinated']:.4f}")
    logger.info("finetune_classifier.py complete.")


if __name__ == "__main__":
    main()
