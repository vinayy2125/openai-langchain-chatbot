from __future__ import annotations
from datetime import datetime
from app.config import settings
from core_services.embedding_utils import get_embedding
import logging
logger = logging.getLogger("redis_service")

# -------------------------
# Configuration (override via settings)
# -------------------------
PREFIX = getattr(settings, "redis_prefix", "dits_chatbot:")


def generate_and_store_embedding(r, session_id: int, query: str, response: str) -> str:
    """Generate embedding for the given text and store it in Redis.
    Groups queries and embeddings by session_id.

    Args:
        r: Redis client instance.
        session_id: The session ID to group queries under.
        query: The query text to generate an embedding for.
        response: The response text associated with the query.

    Returns:
        The Redis key under which the document is stored.
    """

    # Generate embedding for the new query
    query_embedding = get_embedding(query)
    
    # Create a session-based key
    session_key = f"{PREFIX}session:{session_id}"
    
    try:
        # Try to get existing session data
        existing_data = r.json().get(session_key)
        
        if existing_data:
            # Session exists, append new query data
            existing_data["queries"].append({
                "query": query,
                "query_embedding": query_embedding,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Update the existing session
            r.json().set(session_key, '$', existing_data)
        else:
            # Create new session entry
            new_session_data = {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "queries": [{
                    "query": query,
                    "query_embedding": query_embedding,
                    "response": response,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            r.json().set(session_key, '$', new_session_data)
    
    except Exception as e:
        logger.error(f"Error storing embeddings for session {session_id}: {str(e)}")
        raise
    
    return session_key