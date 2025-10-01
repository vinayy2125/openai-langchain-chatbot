# app\db\redis_vector_helper.py
from app.config import get_redis
from datetime import datetime
from core_services.generate_embeddings import get_embedding as vectorize_text

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


def similarity_search(session_id: str, query: str, top_n: int = 5) -> list:
    """Perform a dummy similarity search by comparing the vectorized query with stored queries using RedisJSON."""
    key = f"session:{session_id}"
    stored = r.json().get(key)
    if not stored:
        return []

    queries = stored.get("queries", [])
    query_vector = vectorize_text(query)

    def distance(vec1, vec2):
        return sum([(a - b) ** 2 for a, b in zip(vec1, vec2)]) ** 0.5

    results = []
    for q in queries:
        vec = q.get("query_embedding", [])
        if len(vec) != len(query_vector):
            continue
        score = distance(query_vector, vec)
        results.append((score, q))

    results.sort(key=lambda x: x[0])
    # Return top_n query objects with similarity score included
    return [
        {
            "query": q.get("query"),
            "query_embedding": q.get("query_embedding"),
            "response": q.get("response"),
            "timestamp": q.get("timestamp"),
            "similarity": score
        } for score, q in results[:top_n]
    ]
