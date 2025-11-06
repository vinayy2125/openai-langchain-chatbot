"""Embedding utilities using SentenceTransformers with local-model support."""
from typing import List
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

logger = logging.getLogger(__name__)

# Allow overriding the model path via environment for offline/local use
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"

MODEL_NAME = DEFAULT_MODEL_NAME

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        logger.exception("Failed Hugging Face login: %s", e)
        raise


try:
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Loaded SentenceTransformer model: %s", MODEL_NAME)
except Exception as e:
    logger.exception("Failed to load embedding model %s: %s", MODEL_NAME, e)
    # re-raise so calling code notices configuration issues early
    raise


def _to_float_list(emb) -> List[float]:
    try:
        if hasattr(emb, "tolist"):
            out = emb.tolist()
        else:
            out = list(emb)
    except Exception:
        out = list(np.asarray(emb).reshape(-1).astype(float).tolist())
    return [float(x) for x in out]


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a given text using the loaded model.

    Returns a plain Python list[float].
    """
    emb = model.encode_query(text)
    return _to_float_list(emb)
 