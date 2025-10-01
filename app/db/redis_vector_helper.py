# app\db\redis_vector_helper.py

import numpy as np
import logging
import json
from app.config import get_redis
from datetime import datetime
from core_services.generate_embeddings import get_embedding as vectorize_text
logger = logging.getLogger(__name__)


# Use the Redis client from the config
r = get_redis

def store_text(session_id: str, text: str) -> bool:
    """Store the text along with its vectorized representation in Redis under the session_id in a structured JSON format."""
    now = datetime.utcnow().isoformat()
    vector = vectorize_text(text)
    query_obj = {
        "query": text,
        "query_embedding": vector,
        "response": text,  # In a real scenario, response might be different
        "timestamp": now
    }
    key = f"session:{session_id}"
    try:
        stored = r.json().get(key)
        if stored is None:
            # Initialize with a new object
            new_obj = {
                "session_id": session_id,
                "created_at": now,
                "queries": [query_obj]
            }
            r.json().set(key, '$', new_obj)
        else:
            # Append the new query object
            r.json().arrappend(key, '$.queries', query_obj)
        return True
    except Exception as e:
        # Optionally log the exception
        return False


# ✅ Updated similarity_search with correct RediSearch query syntax
def similarity_search(session_id: str, query: str, top_n: int = 5) -> list:
    from redis.commands.search.query import Query
    r = get_redis  # Use as object, not callable
    query_embedding = vectorize_text(query)
    query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
    logging.info(f"Performing similarity search for session_id: {session_id} with query: {query}")
    # Use KNN query with dialect 2 if supported
    knn_query = f'*=>[KNN {top_n} @embedding $vec AS vector_score]'
    try:
        q = (
            Query(knn_query)
            .sort_by("vector_score", asc=True)
            .return_fields("text", "chunk_id", "timestamp", "embedding", "vector_score")
            .paging(0, top_n)
            .dialect(2)
        )
        results = r.ft("chunk_index").search(q, query_params={"vec": query_blob})
    except Exception as e:
        logger.error(f"❌ RediSearch query failed: {e}")
        return []

    formatted_results = []
    for doc in results.docs:
        distance = float(getattr(doc, "vector_score", 0.0))
        similarity = 1.0 - (distance / 2.0)  # Normalize cosine distance [0,2] → [0,1]

        embedding = getattr(doc, "embedding", "[]")
        try:
            embedding_list = json.loads(embedding) if isinstance(embedding, str) else []
        except Exception:
            embedding_list = []

        formatted_results.append({
            "query": getattr(doc, "chunk_id", "unknown"),
            "query_embedding": embedding_list,
            "response": getattr(doc, "text", ""),
            "timestamp": getattr(doc, "timestamp", ""),
            "similarity": similarity
        })

    return formatted_results