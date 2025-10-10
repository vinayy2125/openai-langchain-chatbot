# core_services\generate_embeddings.py
# core_services\generate_embeddings.py

import os
import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.db import redis_operations as redis_crud

logger = logging.getLogger(__name__)

# Download from the 🤗 Hub
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if HF_API_KEY:
    # Use the supported login mechanism from huggingface_hub if available to avoid
    # the deprecated `use_auth_token` kwarg on SentenceTransformer.
    try:
        from huggingface_hub import login as hf_login

        hf_login(token=HF_API_KEY)
        logger.info("Logged in to Hugging Face hub using HUGGINGFACE_API_KEY")
    except Exception:
        logger.debug("Could not use huggingface_hub.login; falling back to passing token to model")

    model = SentenceTransformer("google/embeddinggemma-300m")
else:
    model = SentenceTransformer("google/embeddinggemma-300m")
# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
logger.info("Embedding shapes: %s %s", getattr(query_embeddings, 'shape', None), getattr(document_embeddings, 'shape', None))

# Compute similarities to determine a ranking
try:
    similarities = model.similarity(query_embeddings, document_embeddings)  # type: ignore
    logger.info("Similarity preview: %s", similarities)
except Exception:
    logger.debug("Model does not expose `similarity` helper; skipping similarity demo")


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a given text using the loaded model.

    Returns a plain Python list[float] no matter what the model returns (numpy/torch/etc.).
    """
    emb = model.encode_query(text)
    # Convert to a list of floats robustly
    try:
        # numpy arrays and torch tensors expose .tolist()
        if hasattr(emb, "tolist"):
            out = emb.tolist()
        else:
            out = list(emb)
    except Exception:
        try:
            out = list(np.asarray(emb).reshape(-1).astype(float).tolist())
        except Exception:
            # Last resort: coerce to string split (unlikely)
            out = [float(x) for x in str(emb).strip('[]()').split() if x.replace('.', '', 1).lstrip('-').isdigit()]

    # Ensure all elements are floats
    return [float(x) for x in out]

def generate_and_store_embedding(r, text: str, metadata: Optional[dict] = None) -> str:
    """Generate embedding for the given text and store it in Redis with optional metadata.

    Args:
        r: Redis client instance.
        text: The text to generate an embedding for.
        metadata: Optional dictionary of metadata to store alongside the embedding.

    Returns:
        The Redis key under which the document is stored.
    """
    embedding = get_embedding(text)
    doc_id = str(uuid.uuid4())
    key = f"{redis_crud.PREFIX}{doc_id}"
    doc = {
        "text": text,
        "embedding": embedding,
        "metadata": metadata or {}
    }
    r.json().set(key, '$', doc)
    return doc_id