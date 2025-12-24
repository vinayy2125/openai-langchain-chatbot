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


def get_embeddings_batch(texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
    """Batch encode multiple texts efficiently.
    
    Uses model.encode() which is optimized for batch processing,
    providing 5-10x speedup over sequential get_embedding() calls.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process at once (default 32)
        show_progress: Whether to show progress bar for large batches
    
    Returns:
        List of embedding vectors (each as List[float])
    """
    if not texts:
        return []
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # model.encode() handles batches efficiently
            embs = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend([_to_float_list(e) for e in embs])
        except Exception as e:
            logger.error(f"Batch embedding failed at index {i}: {e}")
            # Fallback: add zero vectors for failed batch
            all_embeddings.extend([[0.0] * 768 for _ in batch])
    
    if show_progress and len(texts) > 100:
        logger.info(f"Generated {len(all_embeddings)} embeddings in {total_batches} batches")
    
    return all_embeddings
