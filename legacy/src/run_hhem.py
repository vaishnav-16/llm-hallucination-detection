"""
run_hhem.py — Run Vectara HHEM hallucination evaluation model on normalized QA data.

Reads data/halueval_qa_normalized.csv, runs cross-encoder inference,
and saves results to results/hhem_predictions.csv.

HHEM scores CONSISTENCY: score > 0.5 means faithful (label=0),
score <= 0.5 means hallucinated (label=1).
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
THRESHOLD = 0.5


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available. Using GPU: {name}")
        return "cuda"
    else:
        logger.info("CUDA not available. Using CPU.")
        return "cpu"


def _find_hhem_snapshot() -> str:
    """Find the HHEM snapshot directory in HF cache."""
    import glob
    home = os.path.expanduser("~")
    patterns = [
        os.path.join(home, ".cache/huggingface/hub/models--vectara--hallucination_evaluation_model/snapshots/*/modeling_hhem_v2.py"),
        "C:/Users/*/AppData/Local/huggingface/hub/models--vectara--hallucination_evaluation_model/snapshots/*/modeling_hhem_v2.py",
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return os.path.dirname(files[0])
    return None


def load_hhem_direct(device: str):
    """Load HHEM using its own predict() method, bypassing AutoModel pipeline issues.

    The custom HHEMv2ForSequenceClassification class has a predict(text_pairs) method
    that handles tokenization and inference internally using the flan-t5-base tokenizer.
    We create a temporary Python package from the snapshot to handle relative imports,
    then load weights manually via safetensors.
    """
    try:
        import sys
        import shutil
        import tempfile
        from safetensors.torch import load_file

        snapshot_dir = _find_hhem_snapshot()
        if snapshot_dir is None:
            from transformers import AutoConfig
            try:
                AutoConfig.from_pretrained("vectara/hallucination_evaluation_model", trust_remote_code=True)
            except Exception:
                pass
            snapshot_dir = _find_hhem_snapshot()

        if snapshot_dir is None:
            raise FileNotFoundError("HHEM snapshot directory not found")

        logger.info(f"Found HHEM snapshot at: {snapshot_dir}")

        # Create a temp package to allow relative imports in the custom code
        tmp_pkg_dir = tempfile.mkdtemp(prefix="hhem_pkg_")
        pkg_name = "hhem_pkg"
        pkg_dir = os.path.join(tmp_pkg_dir, pkg_name)
        os.makedirs(pkg_dir)

        # Copy files and fix the relative import
        for fname in ["modeling_hhem_v2.py", "configuration_hhem_v2.py"]:
            src = os.path.join(snapshot_dir, fname)
            dst = os.path.join(pkg_dir, fname)
            with open(src) as f:
                content = f.read()
            # Fix relative import → absolute import within our package
            content = content.replace("from .configuration_hhem_v2", f"from {pkg_name}.configuration_hhem_v2")
            with open(dst, "w") as f:
                f.write(content)

        # Write __init__.py
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")

        if tmp_pkg_dir not in sys.path:
            sys.path.insert(0, tmp_pkg_dir)

        # Re-import fresh copies
        for mod in list(sys.modules.keys()):
            if "hhem_pkg" in mod or "modeling_hhem" in mod or "configuration_hhem" in mod:
                del sys.modules[mod]

        from hhem_pkg.modeling_hhem_v2 import HHEMv2ForSequenceClassification
        from hhem_pkg.configuration_hhem_v2 import HHEMv2Config

        logger.info("Instantiating HHEMv2ForSequenceClassification...")
        config = HHEMv2Config()
        model = HHEMv2ForSequenceClassification(config)

        # Load weights from safetensors
        weights_path = os.path.join(snapshot_dir, "model.safetensors")
        logger.info(f"Loading weights from {weights_path}...")
        state_dict = load_file(weights_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"Missing keys (expected for T5 shared embeddings): {len(missing)}")

        model.eval()
        model.t5 = model.t5.to(device)
        logger.info("HHEM loaded successfully via direct snapshot loading!")

        # Cleanup temp dir on successful load (keep for import lifetime)
        return ("hhem_direct", model, None)
    except Exception as e:
        logger.warning(f"Direct HHEM load failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def load_nli_crossencoder(device: str):
    """Load NLI cross-encoder as HHEM fallback."""
    models_to_try = [
        "cross-encoder/nli-deberta-v3-base",
        "cross-encoder/nli-MiniLM2-L6-H768",
        "cross-encoder/nli-deberta-v3-small",
    ]
    for model_id in models_to_try:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading NLI cross-encoder: {model_id}...")
            model = CrossEncoder(model_id, device=device, max_length=512)
            logger.info(f"NLI cross-encoder loaded: {model_id}")
            return ("nli_crossencoder", model, None)
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")
    return None


def load_similarity_fallback(device: str):
    """Load cosine similarity baseline as last resort."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Falling back to all-MiniLM-L6-v2 cosine similarity...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return ("similarity", model, None)
    except Exception as e:
        logger.error(f"Similarity fallback failed: {e}")
        raise RuntimeError("Could not load any hallucination detection model.")


def load_model(device: str):
    """Try HHEM direct, then NLI cross-encoder, then similarity."""
    result = load_hhem_direct(device)
    if result is not None:
        return result

    result = load_nli_crossencoder(device)
    if result is not None:
        return result

    return load_similarity_fallback(device)


def run_hhem_direct_batched(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> List[float]:
    """Run HHEM model using its own predict() method in batches.

    HHEMv2ForSequenceClassification.predict() handles tokenization internally
    using the flan-t5-base tokenizer. It returns probability of class 1 (consistent).
    """
    from tqdm import tqdm
    scores = []
    total = len(pairs)
    current_batch_size = batch_size

    i = 0
    pbar = tqdm(total=total, desc="Running HHEM inference")
    while i < total:
        batch_pairs = pairs[i : i + current_batch_size]
        try:
            # Use the model's built-in predict method
            raw_scores = model.predict(batch_pairs)  # returns tensor of probs
            if hasattr(raw_scores, 'cpu'):
                raw_scores = raw_scores.cpu().numpy().tolist()
            elif hasattr(raw_scores, 'tolist'):
                raw_scores = raw_scores.tolist()
            scores.extend(raw_scores)
            pbar.update(len(batch_pairs))
            i += current_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"OOM! Reducing batch size to {current_batch_size}")
            else:
                logger.warning(f"Batch {i} error: {e}. Using neutral score.")
                scores.extend([0.5] * len(batch_pairs))
                pbar.update(len(batch_pairs))
                i += current_batch_size
        except Exception as e:
            logger.warning(f"Batch {i} error: {e}. Using neutral score.")
            scores.extend([0.5] * len(batch_pairs))
            pbar.update(len(batch_pairs))
            i += current_batch_size

    pbar.close()
    return scores


def run_nli_crossencoder_batched(
    model,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    device: str,
) -> List[float]:
    """Run NLI cross-encoder. Labels: [contradiction, entailment, neutral].
    We use entailment probability as consistency score.
    """
    from tqdm import tqdm
    scores = []
    total = len(pairs)
    current_batch_size = batch_size

    i = 0
    pbar = tqdm(total=total, desc="Running NLI cross-encoder")
    while i < total:
        batch = pairs[i : i + current_batch_size]
        try:
            # predict() returns scores per class: [contradiction, entailment, neutral]
            raw = model.predict(batch, show_progress_bar=False, apply_softmax=True)
            if hasattr(raw, 'tolist'):
                raw = raw.tolist()
            for row in raw:
                if isinstance(row, (list, np.ndarray)):
                    # Label order for deberta-v3: [contradiction, entailment, neutral]
                    # Use entailment score as consistency score
                    entail_score = float(row[1]) if len(row) > 1 else float(row[0])
                    scores.append(entail_score)
                else:
                    scores.append(float(row))
            pbar.update(len(batch))
            i += current_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"OOM! Reducing batch size to {current_batch_size}")
            else:
                logger.warning(f"Batch {i} RuntimeError: {e}")
                scores.extend([0.5] * len(batch))
                pbar.update(len(batch))
                i += current_batch_size
        except Exception as e:
            logger.warning(f"Batch {i} error: {e}")
            scores.extend([0.5] * len(batch))
            pbar.update(len(batch))
            i += current_batch_size

    pbar.close()
    return scores


def run_similarity_batched(model, pairs: List[Tuple[str, str]], batch_size: int) -> List[float]:
    """Run cosine similarity as fallback. Returns 1 - similarity as hallucination proxy.
    Note: For this fallback, we INVERT the score so that higher = more consistent.
    The cosine similarity is already in [0,1] range after normalization.
    We use raw similarity as the consistency score.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from tqdm import tqdm
    scores = []

    premises = [p for p, _ in pairs]
    hypotheses = [h for _, h in pairs]

    for i in tqdm(range(0, len(pairs), batch_size), desc="Running similarity baseline"):
        p_batch = premises[i : i + batch_size]
        h_batch = hypotheses[i : i + batch_size]
        try:
            p_embs = model.encode(p_batch, convert_to_numpy=True)
            h_embs = model.encode(h_batch, convert_to_numpy=True)
            for p_emb, h_emb in zip(p_embs, h_embs):
                sim = cos_sim([p_emb], [h_emb])[0][0]
                score = (float(sim) + 1.0) / 2.0  # Map to [0,1]
                scores.append(score)
        except Exception as e:
            logger.warning(f"Similarity batch failed: {e}")
            scores.extend([0.5] * len(p_batch))

    return scores


def validate_scores(scores: List[float], true_labels: np.ndarray, model_type: str) -> List[float]:
    """Validate and potentially invert scores based on model type and score-label correlation."""
    scores_arr = np.array(scores)
    mean_faithful = scores_arr[true_labels == 0].mean()
    mean_halluc = scores_arr[true_labels == 1].mean()

    logger.info(f"Pre-validation — Mean score: faithful={mean_faithful:.4f}, hallucinated={mean_halluc:.4f}")

    # If hallucinated has HIGHER score than faithful, we need to invert
    # (since HHEM convention: higher score = more consistent = faithful)
    if mean_halluc > mean_faithful:
        logger.warning(f"Score polarity is inverted (hallucinated > faithful). Inverting scores.")
        scores = [1.0 - s for s in scores]
        scores_arr = np.array(scores)
        mean_faithful = scores_arr[true_labels == 0].mean()
        mean_halluc = scores_arr[true_labels == 1].mean()
        logger.info(f"After inversion — Mean score: faithful={mean_faithful:.4f}, hallucinated={mean_halluc:.4f}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Run HHEM on normalized QA data.")
    parser.add_argument("--input", type=str, default="data/halueval_qa_normalized.csv")
    parser.add_argument("--output", type=str, default="results/hhem_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("results", exist_ok=True)

    # Load data
    logger.info(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # Detect device
    device = get_device()

    # Set batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 32 if device == "cuda" else 8
    logger.info(f"Batch size: {batch_size}")

    # Load model
    model_type, model, tokenizer = load_model(device)
    logger.info(f"Using model type: {model_type}")

    # Build input pairs: (question + " " + context, response)
    logger.info("Building input pairs...")
    pairs = []
    for _, row in df.iterrows():
        premise = str(row["question"]) + " " + str(row["context"])
        hypothesis = str(row["response"])
        pairs.append((premise, hypothesis))

    logger.info(f"Running inference on {len(pairs)} examples...")
    start_time = time.time()

    if model_type == "hhem_direct":
        scores = run_hhem_direct_batched(model, tokenizer, pairs, batch_size, device)
    elif model_type == "nli_crossencoder":
        scores = run_nli_crossencoder_batched(model, pairs, batch_size, device)
    else:  # similarity
        scores = run_similarity_batched(model, pairs, batch_size)

    elapsed = time.time() - start_time
    samples_per_sec = len(pairs) / elapsed if elapsed > 0 else 0

    logger.info(f"Inference complete. Total time: {elapsed:.2f}s ({samples_per_sec:.1f} samples/sec)")
    logger.info(f"Device used: {device}, Model type: {model_type}")

    # Validate and potentially fix score polarity
    true_labels = df["label"].values
    scores = validate_scores(scores, true_labels, model_type)

    # Apply threshold: score > 0.5 → faithful (label=0), score <= 0.5 → hallucinated (label=1)
    predicted_labels = [0 if s > args.threshold else 1 for s in scores]

    # Build results dataframe
    df["hhem_score"] = scores
    df["predicted_label"] = predicted_labels

    # Statistics
    score_arr = np.array(scores)
    logger.info(f"Final mean score for faithful (label=0): {score_arr[true_labels == 0].mean():.4f}")
    logger.info(f"Final mean score for hallucinated (label=1): {score_arr[true_labels == 1].mean():.4f}")
    logger.info(f"Predicted label distribution: {pd.Series(predicted_labels).value_counts().to_dict()}")

    # Quick accuracy check
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(true_labels, predicted_labels)
    logger.info(f"Accuracy: {acc:.4f}")

    # Save
    df.to_csv(args.output, index=False)
    logger.info(f"Saved predictions to {args.output}")
    logger.info("run_hhem.py complete.")


if __name__ == "__main__":
    main()
