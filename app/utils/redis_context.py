# Utility to build LLM context from chat history
# build_llm_context_from_history moved to app.utils.chat_state
from app.logger import get_logger
from typing import List, Dict, Any, Optional
from app.db.redis_vector_helper import similarity_search_chat_history
from app.utils.llm_client import call_llm_summarize_chunks
import json
from app.db import base
from app.config import get_redis_client as config_get_redis_client, get_redis as config_redis_client
from app.ingestion.scrape_to_redis import create_index_from_yaml
from core_services.embedding_utils import get_embedding
from concurrent.futures import ThreadPoolExecutor
import threading

# Single-threaded executor for background embedding tasks (small default)
_EMBEDDING_EXECUTOR = ThreadPoolExecutor(max_workers=2)
from pathlib import Path

logger = get_logger("chatbot")

# ChromaDB for knowledge base search (lazy-loaded)
_chroma_manager = None


def _get_chroma_manager():
    """Get or create ChromaDB manager instance (lazy-loaded)."""
    global _chroma_manager
    if _chroma_manager is None:
        try:
            from app.db.chroma_manager import get_chroma_manager
            _chroma_manager = get_chroma_manager()
        except Exception as e:
            logger.warning(f"ChromaDB not available, falling back to Redis: {e}")
            return None
    return _chroma_manager


def _safe_json_set(client, key, path, value):
    """Safely set a RedisJSON value.

    If the key exists but has the wrong Redis type (e.g., plain string), delete
    the key and retry the JSON set. Return True on success, False otherwise.
    """
    try:
        client.json().set(key, path, value)
        return True
    except Exception as e:
        # Detect Redis "wrong type" response and recover by deleting the key
        msg = str(e)
        if "wrong Redis type" in msg or "WRONGTYPE" in msg or "Existing key has wrong Redis type" in msg:
            try:
                # Attempt to remove conflicting key and retry
                if hasattr(client, 'delete'):
                    try:
                        client.delete(key)
                    except Exception:
                        # Older redis clients might use 'del' or not expose delete; try raw command
                        try:
                            client.execute_command('DEL', key)
                        except Exception:
                            pass
                client.json().set(key, path, value)
                return True
            except Exception:
                logger.exception("Failed to recover from wrong Redis type for key %s", key)
                return False
        else:
            # Not a type error – re-raise for upstream handling
            raise


def get_redis_client():
    """Return the shared redis client from `app.config` when available,
    otherwise create a fresh client via the config helper. Avoids duplicating
    connection logic.
    """
    try:
        # config_redis_client is a function (get_redis), call it to get the client
        if callable(config_redis_client):
            return config_redis_client()
        # If it's already a client instance, return it directly
        if config_redis_client:
            return config_redis_client
    except Exception:
        pass
    # Fall back to factory
    return config_get_redis_client()

# Key generator for chat history chunk
def get_chat_history_chunk_key(session_id):
    return f"chat_history:{session_id}"

# Save chat history chunk to Redis, fallback to DB
def save_chat_history(session_id, chat_history):
    """Save chat history chunk to Redis; on failure persist to DB messages table."""
    try:
        r = get_redis_client()
        key = get_chat_history_chunk_key(session_id)
        # Use RedisJSON to store structured chat history so it can be indexed/searched
        try:
            # Ensure the chat_history_index exists (create on demand)
            yaml_path = Path(__file__).parent.parent / "db" / "chat_history_index.yaml"
            create_index_from_yaml(str(yaml_path))
        except Exception:
            # Non-fatal: if index creation/check fails, continue to store the data
            pass

        # Store as JSON document under the key (root path '$') if RedisJSON is available.
        # Also write a legacy string value when possible so tests and older clients that
        # expect `.set`/`.get` continue to work.
        # Build a text blob from messages for text indexing and embeddings
        try:
            messages_text = "\n".join([ (m.get("content") or "") for m in chat_history ])
        except Exception:
            messages_text = ""

        # Prepare payload and store immediately; compute embedding in background
        payload = {"session_id": session_id, "messages": chat_history, "messages_text": messages_text}

        try:
            _safe_json_set(r, key, '$', payload)
        except Exception:
            # Best-effort: if JSON storage fails for reasons other than type conflict,
            # continue and still write legacy string form below.
            pass

        # Also store string form for compatibility with tests/mocks
        try:
            if hasattr(r, 'set'):
                r.set(key, json.dumps(chat_history))
        except Exception:
            pass

        # Offload embedding computation to background thread to avoid blocking
        def _compute_and_set_embedding(k, text):
            """Background task to compute and store embedding. Errors are logged but never propagate."""
            try:
                if not text or not text.strip():
                    return
                vec = get_embedding(text)
                
                # Get Redis client with validation
                try:
                    client = get_redis_client()
                except Exception as client_err:
                    logger.warning("Failed to get Redis client for embedding: %s", client_err)
                    return
                
                # Validate client is usable
                if client is None or not hasattr(client, 'json'):
                    logger.warning("Redis client is invalid or missing json() method for key %s", k)
                    return
                
                # Try to update existing JSON root with embedding
                try:
                    # If key contains a valid JSON root, this will return the object
                    existing = None
                    try:
                        existing = client.json().get(k)
                    except Exception:
                        existing = None

                    if existing is None:
                        # Try to recover from legacy string storage
                        payload = None
                        try:
                            raw = client.get(k) if hasattr(client, 'get') else None
                            if raw:
                                if isinstance(raw, bytes):
                                    raw = raw.decode('utf-8')
                                payload = json.loads(raw)
                        except Exception:
                            payload = None

                        if not payload:
                            # create a minimal root payload
                            payload = {"session_id": session_id, "messages": [], "messages_text": text}
                        else:
                            # Some legacy clients stored the value as a list of messages (not a dict).
                            # If we get a list, convert it into the expected dict structure.
                            if isinstance(payload, list):
                                payload = {"session_id": session_id, "messages": payload, "messages_text": text}
                            # If payload is not a dict at this point, replace with minimal payload
                            if not isinstance(payload, dict):
                                payload = {"session_id": session_id, "messages": [], "messages_text": text}
                        # Safely set embedding on the payload dict
                        payload["embedding"] = vec
                        try:
                            _safe_json_set(client, k, '$', payload)
                            return
                        except Exception as e:
                            logger.warning("Failed to create JSON root for %s: %s (non-fatal, continuing)", k, e)
                            return

                    # existing JSON root present, set embedding path
                    try:
                        _safe_json_set(client, k, '$.embedding', vec)
                        return
                    except Exception:
                        # If setting a non-root object fails, attempt to replace root with added embedding
                        try:
                            existing["embedding"] = vec
                            _safe_json_set(client, k, '$', existing)
                            return
                        except Exception:
                            logger.warning("Failed to store embedding into Redis JSON for %s (non-fatal)", k)
                except Exception:
                    logger.warning("Background embedding write failed for session %s (non-fatal)", session_id)
            except Exception:
                logger.warning("Background embedding computation failed for session %s (non-fatal)", session_id)

        try:
            _EMBEDDING_EXECUTOR.submit(_compute_and_set_embedding, key, messages_text)
        except Exception:
            # best-effort; ignore if executor fails
            pass

        return True
    except Exception as e:
        logger.error(f"Redis save failed for chat_history: {e}, falling back to DB.")
        try:
            conn = base.get_db_conn()
            cur = conn.cursor()
            insert_sql = (
                "INSERT INTO messages (session_id, content, role, metadata) VALUES (%s, %s, %s, %s)"
            )
            for msg in chat_history:
                content = msg.get("content")
                role = msg.get("role") or msg.get("sender")
                metadata = msg.get("metadata") or {}
                cur.execute(insert_sql, (session_id, content, role, json.dumps(metadata)))
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as db_e:
            logger.error(f"DB save failed for chat_history: {db_e}")
            return False

# Retrieve chat history chunk from Redis, fallback to DB
def get_chat_history(session_id):
    """Retrieve chat history chunk from Redis; fallback to DB messages table."""
    try:
        r = get_redis_client()
        key = get_chat_history_chunk_key(session_id)
        # First try legacy string GET (tests/mocks often implement this).
        try:
            if hasattr(r, 'get'):
                data = r.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    return json.loads(data)
        except Exception:
            pass

        # Fall back to RedisJSON get
        try:
            data = r.json().get(key)
            if data and isinstance(data, dict):
                return data.get("messages", [])
        except Exception:
            pass

        return []
    except Exception as e:
        logger.warning(f"Redis retrieval failed for chat_history: {e}, falling back to DB.")
        try:
            conn = base.get_db_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT content, role, metadata, created_at FROM messages WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,)
            )
            rows = cur.fetchall()
            chat_history = []
            for row in rows:
                content, role, metadata, created_at = row
                chat_history.append({
                    "sender": role,
                    "content": content,
                    "metadata": metadata,
                    "timestamp": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
                })
            cur.close()
            conn.close()
            return chat_history
        except Exception as db_e:
            logger.error(f"DB retrieval failed for chat_history: {db_e}")
            return []

# Append a message to the chat history chunk, with Redis and DB fallback
def append_message_to_chat_history(session_id, message):
    history = get_chat_history(session_id)
    history.append(message)
    save_chat_history(session_id, history)
    return True


def get_redis_context_chunks(
    session_id: str,
    query: str,
    top_n: int = 4,
    key_terms: Optional[List[str]] = None,
    domain_prefix: str = "",
    fallback_keywords: str = "capabilities",
) -> List[str]:
    """
    Retrieve context chunks using ChromaDB (primary) and Redis chat history.

    Uses ChromaDB for knowledge base semantic similarity search and Redis
    for chat history search. Falls back to Redis-only if ChromaDB is unavailable.

    Args:
        session_id: Session identifier
        query: User query (semantic search handles intent inference)
        top_n: Number of top results to retrieve (calculated dynamically by caller)
        key_terms: Optional domain-specific terms (if provided, used for filtering)
        domain_prefix: Optional domain prefix for search query
        fallback_keywords: Fallback if query is empty

    Returns:
        List of relevant context chunks, ordered by semantic similarity
    """
    chat_history = get_chat_history(session_id)

    # Derive search query from chat history and user query
    search_query = derive_search_query(chat_history, query, domain_prefix, fallback_keywords)

    # Perform KB similarity search using ChromaDB (primary)
    kb_results = []
    chroma = _get_chroma_manager()
    
    if chroma:
        try:
            # Use ChromaDB for knowledge base search
            chroma_results = chroma.similarity_search(search_query, n_results=top_n)
            # Convert ChromaDB results to common format
            for result in chroma_results:
                kb_results.append({
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": result.get("similarity", 0),
                })
            logger.debug(f"ChromaDB returned {len(kb_results)} results for query")
        except Exception as e:
            logger.warning(f"ChromaDB similarity search failed: {e}")
    else:
        # Fallback to Redis if ChromaDB is not available
        try:
            from app.db.redis_vector_helper import similarity_search
            kb_results = similarity_search(session_id, search_query, top_n=top_n)
            logger.debug(f"Redis fallback returned {len(kb_results)} results")
        except Exception as e:
            logger.warning(f"Redis KB similarity search failed: {e}")

    # Perform chat-history similarity search (Redis - chat_history_index)
    # IMPORTANT: Pass session_id to filter to current session only (prevents data leakage)
    history_results = []
    try:
        history_results = similarity_search_chat_history(search_query, session_id=session_id, top_n=top_n)
    except Exception as e:
        logger.warning(f"Chat-history similarity search failed: {e}")

    # Normalize both result sets to simple text chunks and deduplicate (history first)
    processed_history = []
    for item in history_results:
        text = item.get("messages_text") or item.get("response") or item.get("messages_text", "")
        if text:
            processed_history.append(str(text))

    processed_kb = []
    for item in kb_results:
        text = item.get("text") or item.get("response") or item.get("query") or ""
        if text:
            processed_kb.append(str(text))

    # Merge while preserving uniqueness (history prioritized)
    seen = set()
    merged: List[str] = []
    for t in processed_history + processed_kb:
        key = t.strip().lower()
        if not key or key in seen:
            continue
        merged.append(t)
        seen.add(key)

    return merged[:top_n]


def derive_search_query(chat_history, query, domain_prefix: str, fallback_keywords: str):
    q = (query or "").strip()
    if not q:
        q = get_recent_user_content(chat_history) or ""

    if not q:
        logger.info("[RedisContext] Empty query; using fallback keywords for search.")
        q = fallback_keywords or "capabilities"

    search_query = f"{domain_prefix} {q}" if domain_prefix else q
    recent_short = get_recent_user_content(chat_history, limit_chars=300)
    if recent_short and recent_short != q:
        search_query = f"{search_query} {recent_short}"
    return search_query


def get_recent_user_content(history, limit_chars=300):
    if not history:
        return None
    for item in reversed(history):
        role = item.get("role") or item.get("sender")
        if role == "user":
            content = (item.get("content") or "").strip()
            if content:
                return content[:limit_chars]
    return None


def process_similarity_results(results) -> List[str]:
    if not results:
        logger.info("[RedisContext] No context retrieved from Redis.")
        return []
    seen_content = set()
    normalized: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        text = (item.get("response") or item.get("text") or item.get("query") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_content:
            continue
        normalized.append(text)
        seen_content.add(key)
    if normalized:
        max_chunks = min(len(normalized), 5)
        return normalized[:max_chunks]
    return []

    # (end of get_redis_context_chunks)


# Helper function to summarize chunks with LLM
def summarize_chunks_with_llm(chunks: List[str], query: str) -> str:
    """
    Use the LLM to summarize the provided chunks, extracting links and detailed info relevant to the query.
    Returns a single summary string.
    """
    if not chunks:
        return ""

    prompt = (
        f"Summarize the following context chunks for the query: '{query}'. "
        "Extract any relevant links and provide detailed information. "
        "Return a single, concise summary for use as AI context.\n\n"
        "Context Chunks:\n" + "\n---\n".join(chunks)
    )
    try:
        summary = call_llm_summarize_chunks(prompt)
        return summary.strip() if summary else ""
    except Exception as e:
        logger.error(f"[RedisContext] LLM summarization failed: {e}")
        return ""
