import logging
from typing import List, Dict, Any
from app.db.redis_vector_helper import similarity_search

logger = logging.getLogger("chatbot")

def get_redis_context_chunks(session_id: str, query: str, conversation_history: List[Dict[str, Any]], top_n: int = 6) -> List[str]:
    """
    Retrieve context chunks from Redis using the latest query and conversation history.
    Returns a list of text chunks (strings) for LLM context.
    """
    # Combine latest query and recent conversation for richer context
    # (You can tune this logic as needed)
    search_query = query
    if conversation_history:
        # Accept both list[dict] (with 'content') and list[tuple](role, content)
        last_msgs = []
        for item in conversation_history[-3:]:
            try:
                if isinstance(item, dict):
                    c = item.get("content")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # role, message_text
                    c = item[1]
                else:
                    c = None
                if c and isinstance(c, str) and c.strip():
                    last_msgs.append(c.strip())
            except Exception:
                continue
        search_query = " ".join(last_msgs + [query])
    
    results = similarity_search(session_id, search_query, top_n=top_n)

    if not results:
        logger.info("[RedisContext] No context retrieved from Redis.")
        return []

    # Expect standardized list[dict] from similarity_search with keys: response, query, query_embedding, timestamp, similarity
    normalized: List[str] = []
    previews: List[str] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            logger.debug(f"[RedisContext] Skipping non-dict item at index {idx}: {type(item)}")
            continue
        text = item.get("response") or item.get("text") or item.get("query") or ""
        if text and isinstance(text, str) and text.strip():
            normalized.append(text.strip())
            previews.append(text.strip()[:100])

    logger.info(f"[RedisContext] Retrieved {len(normalized)} context items (top_n={top_n})")
    if previews:
        logger.info(f"[RedisContext] Retrieved context preview (first 100 chars each): {' | '.join(previews)}")

    return normalized
