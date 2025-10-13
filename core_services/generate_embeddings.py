# core_services\generate_embeddings.py


import uuid
from typing import Optional

from app.db import redis_operations as redis_crud
from core_services.embedding_utils import get_embedding

def generate_and_store_embedding(r, text: str, metadata: Optional[dict[str, object]] = None) -> str:
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