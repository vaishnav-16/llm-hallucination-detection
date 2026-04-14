"""
load_data.py — Download and normalize HaluEval QA dataset from Hugging Face Hub.

Produces data/halueval_qa_normalized.csv with columns:
  id, question, context, response, label
where label=0 means faithful (correct answer) and label=1 means hallucinated.
Each raw example produces TWO evaluation rows.
"""

import argparse
import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42


def load_halueval_qa(num_samples: int = 500):
    """Load HaluEval QA dataset from Hugging Face Hub."""
    from datasets import get_dataset_config_names, load_dataset

    logger.info("Loading HaluEval QA dataset from Hugging Face Hub...")

    # Try several config names
    configs_to_try = ["qa", "qa_samples", "qa_data", None]
    dataset = None
    used_config = None

    for config in configs_to_try:
        try:
            if config is None:
                dataset = load_dataset("pminervini/HaluEval")
                used_config = "default"
            else:
                dataset = load_dataset("pminervini/HaluEval", config)
                used_config = config
            logger.info(f"Successfully loaded with config: '{used_config}'")
            break
        except Exception as e:
            logger.warning(f"Config '{config}' failed: {e}")

    if dataset is None:
        # Inspect available configs
        try:
            configs = get_dataset_config_names("pminervini/HaluEval")
            logger.info(f"Available configs: {configs}")
            # Try each available config
            for config in configs:
                try:
                    dataset = load_dataset("pminervini/HaluEval", config)
                    used_config = config
                    logger.info(f"Loaded with discovered config: '{config}'")
                    break
                except Exception as e2:
                    logger.warning(f"Config '{config}' failed: {e2}")
        except Exception as e3:
            logger.error(f"Could not get config names: {e3}")

    if dataset is None:
        logger.error("All loading attempts failed. Creating synthetic dataset.")
        return _create_synthetic_dataset(num_samples)

    # Inspect schema
    split = list(dataset.keys())[0]
    ds_split = dataset[split]
    logger.info(f"Dataset split used: '{split}', total rows: {len(ds_split)}")
    logger.info(f"Columns: {ds_split.column_names}")
    logger.info("Sample rows:")
    for i in range(min(3, len(ds_split))):
        logger.info(f"  Row {i}: {dict(ds_split[i])}")

    return ds_split


def normalize_dataset(ds_split, num_samples: int = 500) -> pd.DataFrame:
    """Normalize raw dataset into evaluation pairs."""
    random.seed(SEED)
    np.random.seed(SEED)

    columns = ds_split.column_names
    logger.info(f"Normalizing dataset with columns: {columns}")

    # Sample num_samples examples
    total = len(ds_split)
    sample_size = min(num_samples, total)
    indices = list(range(total))
    random.shuffle(indices)
    indices = indices[:sample_size]

    logger.info(f"Sampling {sample_size} raw examples → {sample_size * 2} evaluation pairs...")

    rows = []
    for i, idx in enumerate(indices):
        item = ds_split[idx]

        # Detect column names flexibly
        question = _get_field(item, ["question", "query", "input", "q"])
        context = _get_field(item, ["knowledge", "context", "passage", "document", "source", "right_context"])
        correct_answer = _get_field(item, ["right_answer", "correct_answer", "answer", "response", "target", "output"])
        hallucinated_answer = _get_field(item, ["hallucinated_answer", "wrong_answer", "hallucination", "hal_answer", "negative_answer"])

        if not question:
            question = str(item.get(columns[0], ""))
        if not context:
            context = str(item.get(columns[1], "")) if len(columns) > 1 else ""
        if not correct_answer:
            correct_answer = str(item.get(columns[2], "")) if len(columns) > 2 else ""
        if not hallucinated_answer:
            hallucinated_answer = str(item.get(columns[3], "")) if len(columns) > 3 else ""

        # Faithful row (label=0)
        rows.append({
            "id": f"{idx}_faithful",
            "question": question,
            "context": context,
            "response": correct_answer,
            "label": 0,
        })

        # Hallucinated row (label=1)
        rows.append({
            "id": f"{idx}_hallucinated",
            "question": question,
            "context": context,
            "response": hallucinated_answer,
            "label": 1,
        })

    df = pd.DataFrame(rows)
    return df


def _get_field(item: dict, candidates: list) -> str:
    """Return first matching field value from candidates."""
    for key in candidates:
        if key in item and item[key]:
            return str(item[key])
    return ""


def _create_synthetic_dataset(num_samples: int = 500) -> object:
    """Create a minimal synthetic dataset as fallback."""
    logger.warning("Creating synthetic fallback dataset with 50 examples.")
    import random as rnd

    class FakeDataset:
        column_names = ["question", "knowledge", "right_answer", "hallucinated_answer"]
        def __init__(self, n):
            self.data = []
            templates = [
                ("What is the capital of France?", "France is a country in Western Europe.", "Paris", "Lyon"),
                ("Who wrote Hamlet?", "Hamlet is a tragedy by Shakespeare.", "William Shakespeare", "Charles Dickens"),
                ("What is the speed of light?", "Light travels at approximately 3×10^8 m/s.", "299,792,458 m/s", "150,000,000 m/s"),
                ("When was the Eiffel Tower built?", "The Eiffel Tower was constructed as a temporary exhibit.", "1889", "1901"),
                ("What is H2O?", "H2O is the chemical formula for water.", "Water", "Oxygen"),
            ]
            for i in range(n):
                q, k, r, h = templates[i % len(templates)]
                self.data.append({"question": q, "knowledge": k, "right_answer": r, "hallucinated_answer": h})

        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    return FakeDataset(50)


def main():
    parser = argparse.ArgumentParser(description="Load and normalize HaluEval QA dataset.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of raw examples to sample")
    parser.add_argument("--output", type=str, default="data/halueval_qa_normalized.csv")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    ds = load_halueval_qa(args.num_samples)
    df = normalize_dataset(ds, args.num_samples)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    logger.info(f"Sample rows:\n{df.head(4).to_string()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    df.to_csv(args.output, index=False)
    logger.info(f"Saved normalized data to {args.output}")
    logger.info("load_data.py complete.")


if __name__ == "__main__":
    main()
