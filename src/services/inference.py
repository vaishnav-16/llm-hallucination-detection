"""Model inference — HHEM, fine-tuned DeBERTa, and fallback models."""

import glob
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SEED = 42
DEFAULT_THRESHOLD = 0.5


class InferenceError(Exception):
    """Raised when model inference fails irrecoverably."""
    pass


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available. Using GPU: {name}")
        return "cuda"
    logger.info("CUDA not available. Using CPU.")
    return "cpu"


def _find_hhem_snapshot() -> Optional[str]:
    """Find the HHEM snapshot directory in HF cache."""
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


def load_hhem(device: str = "auto"):
    """Load HHEM model with fallback chain.

    Returns (model_type, model, tokenizer) tuple.
    Tries: HHEM direct -> NLI cross-encoder -> similarity baseline.
    """
    if device == "auto":
        device = get_device()

    result = _load_hhem_direct(device)
    if result is not None:
        return result

    result = _load_nli_crossencoder(device)
    if result is not None:
        return result

    return _load_similarity_fallback(device)


def _load_hhem_direct(device: str):
    """Load HHEM using direct snapshot loading."""
    try:
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

        tmp_pkg_dir = tempfile.mkdtemp(prefix="hhem_pkg_")
        pkg_name = "hhem_pkg"
        pkg_dir = os.path.join(tmp_pkg_dir, pkg_name)
        os.makedirs(pkg_dir)

        for fname in ["modeling_hhem_v2.py", "configuration_hhem_v2.py"]:
            src = os.path.join(snapshot_dir, fname)
            dst = os.path.join(pkg_dir, fname)
            with open(src) as f:
                content = f.read()
            content = content.replace("from .configuration_hhem_v2", f"from {pkg_name}.configuration_hhem_v2")
            with open(dst, "w") as f:
                f.write(content)

        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")

        if tmp_pkg_dir not in sys.path:
            sys.path.insert(0, tmp_pkg_dir)

        for mod in list(sys.modules.keys()):
            if "hhem_pkg" in mod or "modeling_hhem" in mod or "configuration_hhem" in mod:
                del sys.modules[mod]

        from hhem_pkg.modeling_hhem_v2 import HHEMv2ForSequenceClassification
        from hhem_pkg.configuration_hhem_v2 import HHEMv2Config

        config = HHEMv2Config()
        model = HHEMv2ForSequenceClassification(config)

        weights_path = os.path.join(snapshot_dir, "model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.t5 = model.t5.to(device)
        logger.info("HHEM loaded successfully via direct snapshot loading!")
        return ("hhem_direct", model, None)
    except Exception as e:
        logger.warning(f"Direct HHEM load failed: {e}")
        return None


def _load_nli_crossencoder(device: str):
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


def _load_similarity_fallback(device: str):
    """Load cosine similarity baseline as last resort."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Falling back to all-MiniLM-L6-v2 cosine similarity...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return ("similarity", model, None)
    except Exception as e:
        raise InferenceError(f"Could not load any hallucination detection model: {e}")


def predict_hhem(
    pairs: List[Tuple[str, str]],
    device: str = "auto",
    batch_size: int = 32,
    model_type: str = None,
    model=None,
    tokenizer=None,
) -> np.ndarray:
    """Run HHEM inference on text pairs.

    Args:
        pairs: List of (premise, hypothesis) tuples.
        device: Device to use ('cuda', 'cpu', or 'auto').
        batch_size: Batch size for inference.
        model_type/model/tokenizer: Pre-loaded model (optional).

    Returns:
        Array of consistency scores (higher = more faithful).
    """
    if device == "auto":
        device = get_device()

    if model is None:
        model_type, model, tokenizer = load_hhem(device)

    if batch_size is None:
        batch_size = 32 if device == "cuda" else 8

    logger.info(f"Running {model_type} inference on {len(pairs)} pairs (batch_size={batch_size})...")
    start_time = time.time()

    if model_type == "hhem_direct":
        scores = _run_hhem_direct(model, pairs, batch_size, device)
    elif model_type == "nli_crossencoder":
        scores = _run_nli_crossencoder(model, pairs, batch_size, device)
    else:
        scores = _run_similarity(model, pairs, batch_size)

    elapsed = time.time() - start_time
    logger.info(f"Inference complete. {elapsed:.2f}s ({len(pairs)/elapsed:.1f} samples/sec)")

    return np.array(scores)


def predict_finetuned(
    pairs: List[Tuple[str, str]],
    model_path: str = "models/deberta-hallucination-detector",
    device: str = "auto",
    batch_size: int = 16,
) -> np.ndarray:
    """Run fine-tuned DeBERTa inference.

    Returns hallucination probabilities (higher = more hallucinated).
    """
    if device == "auto":
        device = get_device()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading fine-tuned model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    probs = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        texts = [f"{p} [SEP] {h}" for p, h in batch]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())

    return np.array(probs)


def validate_scores(scores: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Validate and potentially invert scores based on label correlation."""
    mean_faithful = scores[true_labels == 0].mean()
    mean_halluc = scores[true_labels == 1].mean()

    logger.info(f"Score validation — faithful={mean_faithful:.4f}, hallucinated={mean_halluc:.4f}")

    if mean_halluc > mean_faithful:
        logger.warning("Score polarity inverted. Flipping scores.")
        scores = 1.0 - scores
        logger.info(f"After inversion — faithful={scores[true_labels == 0].mean():.4f}, "
                     f"hallucinated={scores[true_labels == 1].mean():.4f}")

    return scores


def build_pairs(df) -> List[Tuple[str, str]]:
    """Build (premise, hypothesis) pairs from a DataFrame."""
    pairs = []
    for _, row in df.iterrows():
        premise = str(row["question"]) + " " + str(row["context"])
        hypothesis = str(row["response"])
        pairs.append((premise, hypothesis))
    return pairs


def _run_hhem_direct(model, pairs, batch_size, device):
    """Run HHEM direct model with OOM recovery."""
    from tqdm import tqdm
    scores = []
    current_bs = batch_size
    i = 0
    pbar = tqdm(total=len(pairs), desc="Running HHEM inference")

    while i < len(pairs):
        batch = pairs[i:i + current_bs]
        try:
            raw = model.predict(batch)
            if hasattr(raw, 'cpu'):
                raw = raw.cpu().numpy().tolist()
            elif hasattr(raw, 'tolist'):
                raw = raw.tolist()
            scores.extend(raw)
            pbar.update(len(batch))
            i += current_bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                logger.warning(f"OOM! Reducing batch size to {current_bs}")
            else:
                scores.extend([0.5] * len(batch))
                pbar.update(len(batch))
                i += current_bs
        except Exception:
            scores.extend([0.5] * len(batch))
            pbar.update(len(batch))
            i += current_bs
    pbar.close()
    return scores


def _run_nli_crossencoder(model, pairs, batch_size, device):
    """Run NLI cross-encoder with OOM recovery."""
    from tqdm import tqdm
    scores = []
    current_bs = batch_size
    i = 0
    pbar = tqdm(total=len(pairs), desc="Running NLI cross-encoder")

    while i < len(pairs):
        batch = pairs[i:i + current_bs]
        try:
            raw = model.predict(batch, show_progress_bar=False, apply_softmax=True)
            if hasattr(raw, 'tolist'):
                raw = raw.tolist()
            for row in raw:
                if isinstance(row, (list, np.ndarray)):
                    scores.append(float(row[1]) if len(row) > 1 else float(row[0]))
                else:
                    scores.append(float(row))
            pbar.update(len(batch))
            i += current_bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                logger.warning(f"OOM! Reducing batch size to {current_bs}")
            else:
                scores.extend([0.5] * len(batch))
                pbar.update(len(batch))
                i += current_bs
        except Exception:
            scores.extend([0.5] * len(batch))
            pbar.update(len(batch))
            i += current_bs
    pbar.close()
    return scores


def _run_similarity(model, pairs, batch_size):
    """Run cosine similarity baseline."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from tqdm import tqdm
    scores = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="Running similarity baseline"):
        batch = pairs[i:i + batch_size]
        try:
            p_embs = model.encode([p for p, _ in batch], convert_to_numpy=True)
            h_embs = model.encode([h for _, h in batch], convert_to_numpy=True)
            for p_emb, h_emb in zip(p_embs, h_embs):
                sim = cos_sim([p_emb], [h_emb])[0][0]
                scores.append((float(sim) + 1.0) / 2.0)
        except Exception:
            scores.extend([0.5] * len(batch))

    return scores
