"""Embedding utilities using SentenceTransformers with local-model support."""
from typing import List
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import torch

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

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
# Determine device ONCE at startup to avoid timing/import issues
if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    logger.info("⚠️ CUDA not available - using CPU")

try:
    model = SentenceTransformer(MODEL_NAME)
    
    # Move model to device configuration immediately
    model = model.to(DEVICE)
    
    logger.info(f"Loaded SentenceTransformer model: {MODEL_NAME} on {DEVICE}")
except Exception as e:
    logger.exception("Failed to load embedding model %s: %s", MODEL_NAME, e)
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


# =============================================================================
# QUERY EMBEDDING CACHE
# =============================================================================
# LRU cache for query embeddings to avoid recomputing identical queries.
# 
# DEPLOYMENT NOTES:
# - Cache is per-process. Multi-worker deployments (uvicorn workers > 1)
#   will have independent caches. This is expected and safe.
# - Memory: ~12MB per 2000 cached 768-dim embeddings per worker.
# - Thread-safe: GIL protects lru_cache operations.
#
# Cache key is raw query text - lru_cache handles argument hashing internally.
# =============================================================================
from functools import lru_cache

# Configurable cache size via environment (default 2000)
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "2000"))


@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def _get_embedding_cached(text: str) -> tuple:
    """
    Cached embedding computation. Returns tuple for hashability.
    
    Internal function - use get_embedding() for public API.
    """
    # Use global DEVICE constant
    emb = model.encode(text, device=DEVICE)
    # Return as tuple for LRU cache compatibility
    return tuple(_to_float_list(emb))


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a given text using the loaded model.

    Returns a plain Python list[float].
    
    NOTE: This function uses LRU caching. Identical queries will return
    cached embeddings. Different queries always compute fresh embeddings.
    """
    # Convert cached tuple back to list for API compatibility
    return list(_get_embedding_cached(text))


def get_embedding_cache_info():
    """Get cache statistics for monitoring. Returns lru_cache cache_info."""
    return _get_embedding_cached.cache_info()


def clear_embedding_cache():
    """Clear the embedding cache. Useful for testing or memory pressure."""
    _get_embedding_cached.cache_clear()
    logger.info("Embedding cache cleared")


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
    
    for batch_idx, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i + batch_size]
        try:
            # Log progress for visibility during long operations
            if show_progress and total_batches > 1:
                logger.info(f"🧠 Embedding batch {batch_idx}/{total_batches} ({len(batch)} texts) on {DEVICE}...")
            
            # Use global DEVICE constant
            embs = model.encode(batch, batch_size=batch_size, show_progress_bar=False, device=DEVICE)
            all_embeddings.extend([_to_float_list(e) for e in embs])
        except Exception as e:
            logger.error(f"Batch {batch_idx}/{total_batches} embedding failed: {e}")
            # Fallback: add zero vectors for failed batch
            all_embeddings.extend([[0.0] * 768 for _ in batch])
    
    logger.info(f"✅ Embedding complete: {len(all_embeddings)} vectors in {total_batches} batches")
    return all_embeddings
