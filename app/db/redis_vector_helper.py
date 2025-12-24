# app\db\redis_vector_helper.py
import numpy as np
import json
from app.logger import get_logger
from app.config import get_redis
from datetime import datetime
logger = get_logger(__name__)
from redis.commands.search.query import Query
from core_services.embedding_utils import get_embedding as vectorize_text



def store_text(session_id: str, text: str) -> bool:
    """Store the text along with its vectorized representation in Redis under the session_id in a structured JSON format."""
    now = datetime.utcnow().isoformat()
    vector = vectorize_text(text)
    query_obj = {
        "query": text,
        "query_embedding": vector,
        "response": text,  
        "timestamp": now
    }
    key = f"session:{session_id}"
    try:
        stored = get_redis().json().get(key)
        if stored is None:

            new_obj = {
                "session_id": session_id,
                "created_at": now,
                "queries": [query_obj]
            }
            get_redis().json().set(key, '$', new_obj)
        else:
            get_redis().json().arrappend(key, '$.queries', query_obj)
        return True
    except Exception as e:
        return False

def similarity_search(session_id: str, query: str, top_n: int = 4) -> list:
    """Search chunk_index for similar KB content.
    
    Note: session_id is kept for API compatibility but not used for filtering
    because chunk_index contains global knowledge base data, not per-session data.
    """
    query_embedding = vectorize_text(query)
    query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
    knn_query = f'*=>[KNN {top_n} @embedding $vec AS vector_score]'
    try:
        q = (
            Query(knn_query)
            .sort_by("vector_score", asc=True)
            .return_fields("text", "chunk_id", "timestamp", "vector_score")
            .paging(0, top_n)
            .dialect(2)
        )
        results = get_redis().ft("chunk_index").search(q, query_params={"vec": query_blob})
    except Exception as e:
        logger.error(f"❌ RediSearch query failed: {e}")
        return []

    formatted_results = []
    for doc in results.docs:
        try:
            distance_raw = getattr(doc, "vector_score", 0.0)
            try:
                distance = float(distance_raw)
            except Exception:
                distance = 0.0
            similarity = 1.0 - (distance / 2.0)

            embedding_list = []
            # Embedding removed from response optimization
            # embedding = getattr(doc, "embedding", None)
            # if isinstance(embedding, str):
            #     try:
            #         embedding_list = json.loads(embedding)
            #     except Exception:
            #         embedding_list = []
            # elif isinstance(embedding, (list, tuple)):
            #     embedding_list = list(embedding)

            response_text = getattr(doc, "text", None) or getattr(doc, "response", None) or ""
            chunk_id = getattr(doc, "chunk_id", None) or getattr(doc, "id", None) or "unknown"
            timestamp = getattr(doc, "timestamp", "")

            formatted_results.append({
                "query": chunk_id,
                "query_embedding": embedding_list,
                "response": str(response_text) if response_text is not None else "",
                "timestamp": str(timestamp),
                "similarity": float(similarity)
            })
        except Exception as e:
            logger.debug(f"Skipping doc in similarity_search due to error: {e}")

    if formatted_results:
        # sample_types = [type(x).__name__ for x in formatted_results[:3]]
        # logger.info(f"[redis_vector_helper] similarity_search returning {len(formatted_results)} items (sample types: {sample_types})")
        pass
    return formatted_results


def similarity_search_chat_history(query: str, session_id: str = None, top_n: int = 4) -> list:
    """Perform KNN search against `chat_history_index` using the query embedding.
    
    Args:
        query: The search query text
        session_id: If provided, filter results to this session only (RECOMMENDED for privacy)
        top_n: Number of results to return

    Returns a list of session-level results with session id, messages_text, embedding, and similarity score.
    """
    query_embedding = vectorize_text(query)
    query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
    
    # Build query with session filter if session_id provided
    # This prevents cross-session data leakage
    if session_id:
        # Use Redis tag filter to scope to current session only
        # Format: @field:{value} for exact tag match
        filter_query = f'(@session_id:{{{session_id}}})=>[KNN {top_n} @embedding $vec AS vector_score]'
    else:
        # Fallback to unfiltered (not recommended - can leak data)
        logger.warning("similarity_search_chat_history called without session_id - may return cross-session data")
        filter_query = f'*=>[KNN {top_n} @embedding $vec AS vector_score]'
    
    try:
        q = (
            Query(filter_query)
            .sort_by("vector_score", asc=True)
            .return_fields("session_id", "messages_text", "vector_score")
            .paging(0, top_n)
            .dialect(2)
        )
        results = get_redis().ft("chat_history_index").search(q, query_params={"vec": query_blob})
    except Exception as e:
        logger.error(f"❌ RediSearch chat_history query failed: {e}")
        return []

    formatted_results = []
    for doc in results.docs:
        try:
            distance_raw = getattr(doc, "vector_score", 0.0)
            try:
                distance = float(distance_raw)
            except Exception:
                distance = 0.0
            similarity = 1.0 - (distance / 2.0)

            embedding_list = []
            # Embedding removed from response optimization
            # embedding = getattr(doc, "embedding", None)
            # if isinstance(embedding, str):
            #     try:
            #         embedding_list = json.loads(embedding)
            #     except Exception:
            #         embedding_list = []
            # elif isinstance(embedding, (list, tuple)):
            #     embedding_list = list(embedding)

            messages_text = getattr(doc, "messages_text", None) or ""
            session_id = getattr(doc, "session_id", None) or getattr(doc, "id", None) or "unknown"

            formatted_results.append({
                "session_id": session_id,
                "messages_text": str(messages_text),
                "embedding": embedding_list,
                "similarity": float(similarity)
            })
        except Exception as e:
            logger.debug(f"Skipping doc in similarity_search_chat_history due to error: {e}")

    return formatted_results