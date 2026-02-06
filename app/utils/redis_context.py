# Utility to build LLM context from chat history
# build_llm_context_from_history moved to app.utils.chat_state
from app.logger import get_logger
from typing import List, Dict, Any, Optional
# Note: similarity_search_chat_history removed - chat history now uses LangChain memory
from app.utils.llm_client import call_llm_summarize_chunks
import json
from app.db import base
from app.config import get_redis_client as config_get_redis_client, get_redis as config_redis_client
from core_services.embedding_utils import get_embedding
from concurrent.futures import ThreadPoolExecutor
import threading

# Single-threaded executor for background embedding tasks (small default)
_EMBEDDING_EXECUTOR = ThreadPoolExecutor(max_workers=2)
from pathlib import Path
import os

logger = get_logger("chatbot")

# ChromaDB for knowledge base search (lazy-loaded)
_chroma_manager = None

# Hybrid search manager (lazy-loaded)
_hybrid_search_manager = None

# ParentDocumentRetriever (lazy-loaded)
_parent_retriever = None

# Environment flag to enable/disable hybrid search (default: enabled)
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"

# Environment flag to enable/disable ParentDocumentRetriever (default: disabled for gradual rollout)
ENABLE_PARENT_RETRIEVER = os.getenv("ENABLE_PARENT_RETRIEVER", "false").lower() == "true"


def _get_chroma_manager():
    """Get or create ChromaDB manager instance (lazy-loaded)."""
    global _chroma_manager
    if _chroma_manager is None:
        try:
            from app.db.chroma_manager import get_chroma_manager
            _chroma_manager = get_chroma_manager()
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}")
            return None
    return _chroma_manager


def _get_hybrid_search_manager():
    """Get or create HybridSearchManager instance (lazy-loaded)."""
    global _hybrid_search_manager
    if _hybrid_search_manager is None:
        try:
            from core_services.hybrid_search import get_hybrid_search_manager
            _hybrid_search_manager = get_hybrid_search_manager()
        except Exception as e:
            logger.warning(f"HybridSearchManager not available: {e}")
            return None
    return _hybrid_search_manager


def _get_parent_retriever():
    """Get or create ParentDocumentRetriever instance (lazy-loaded)."""
    global _parent_retriever
    if _parent_retriever is None:
        try:
            from core_services.parent_document_retriever import get_parent_document_retriever
            _parent_retriever = get_parent_document_retriever()
        except Exception as e:
            logger.warning(f"ParentDocumentRetriever not available: {e}")
            return None
    return _parent_retriever


def _semantic_rerank(query: str, results: List[Dict], top_n: int) -> List[Dict]:
    """
    Re-rank results based on semantic relevance to query.
    Boosts results that contain query keywords for better accuracy.
    
    Args:
        query: The user's search query
        results: List of ChromaDB results with text and similarity
        top_n: Number of results to return after re-ranking
        
    Returns:
        Re-ranked list of results
    """
    if not results:
        return results
    
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 3]  # Skip short words
    
    scored_results = []
    for result in results:
        text = (result.get("text", "") or "").lower()
        base_score = result.get("similarity", 0)
        
        # Boost for exact query term presence
        keyword_matches = sum(1 for word in query_words if word in text)
        keyword_boost = keyword_matches * 0.05
        
        # Boost for "services" related content when query is about services
        service_keywords = ["services", "development", "consulting", "software", "ai", "cloud"]
        if any(sk in query_lower for sk in ["service", "offer", "list", "explore", "capabilities"]):
            service_boost = sum(0.02 for sk in service_keywords if sk in text)
        else:
            service_boost = 0
        
        final_score = base_score + keyword_boost + service_boost
        scored_results.append((final_score, result))
    
    # Sort by adjusted score (highest first)
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    logger.debug(f"[ReRank] Re-ranked {len(results)} results, returning top {top_n}")
    return [r for _, r in scored_results[:top_n]]


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
            # ONLY insert the LAST message (the new one being appended)
            # The other messages already exist in the DB (that's where they were loaded from)
            if chat_history:
                msg = chat_history[-1]  # Get only the newest message
                content = msg.get("content")
                role = msg.get("role") or msg.get("sender")
                metadata = msg.get("metadata") or {}
                
                # Skip if no content
                if content:
                    conn = base.get_db_conn()
                    cur = conn.cursor()
                    
                    # Use the same duplicate prevention as _save_message_sync
                    insert_sql = """
                        INSERT INTO messages (session_id, content, role, metadata, created_at, updated_at)
                        SELECT %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                        WHERE NOT EXISTS (
                            SELECT 1 FROM messages 
                            WHERE session_id = %s 
                              AND role = %s 
                              AND content = %s 
                              AND created_at > CURRENT_TIMESTAMP - INTERVAL '5 seconds'
                        )
                    """
                    cur.execute(insert_sql, (session_id, content, role, json.dumps(metadata), session_id, role, content))
                    conn.commit()
                    cur.close()
                    base.return_db_conn(conn)
            return True
        except Exception as db_e:
            logger.error(f"DB save failed for chat_history: {db_e}")
            return False

# Retrieve chat history from PostgreSQL (single source of truth)
def get_chat_history(session_id):
    """Retrieve chat history from PostgreSQL messages table.
    
    PostgreSQL is now the single source of truth for chat history.
    LangChain memory is used for efficient LLM context building.
    """
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
                "role": role,  # Use 'role' consistently (not 'sender')
                "content": content,
                "metadata": metadata,
                "timestamp": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
            })
        cur.close()
        base.return_db_conn(conn)
        return chat_history
    except Exception as db_e:
        logger.error(f"DB retrieval failed for chat_history: {db_e}")
        return []

# DEPRECATED: No longer used - messages are saved via save_message() in helpers.py
# Kept for backwards compatibility but does nothing
def append_message_to_chat_history(session_id, message):
    """DEPRECATED: Messages are now saved directly via save_message() in helpers.py.
    
    This function is kept for backwards compatibility but is a no-op.
    Use save_message() for persistence and LangChain memory for context.
    """
    logger.debug(f"[DEPRECATED] append_message_to_chat_history called but is now a no-op")
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

    # Perform knowledge base search
    # Priority: ParentDocumentRetriever > Hybrid Search > ChromaDB-only
    kb_results = []
    
    # Try ParentDocumentRetriever first (returns larger context chunks)
    if ENABLE_PARENT_RETRIEVER:
        parent_retriever = _get_parent_retriever()
        if parent_retriever:
            try:
                parent_results = parent_retriever.retrieve(search_query, k=top_n)
                for result in parent_results:
                    kb_results.append({
                        "text": result.get("content", ""),
                        "metadata": result.get("metadata", {}),
                        "similarity": result.get("similarity", 0),
                        "source": result.get("source", "parent"),
                    })
                logger.info(f"[ParentRetriever] Returned {len(kb_results)} parent context results")
            except Exception as e:
                logger.warning(f"ParentDocumentRetriever failed, falling back: {e}")
                kb_results = []
    
    # Fallback: Try hybrid search if ParentRetriever disabled or failed
    if not kb_results and ENABLE_HYBRID_SEARCH:
        # Try hybrid search first (combines semantic and BM25 keyword matching)
        hybrid_manager = _get_hybrid_search_manager()
        if hybrid_manager:
            try:
                # Use hybrid search with balanced alpha (0.5 = equal weight semantic + BM25)
                # Retrieve more candidates for better fusion, then take top_n
                hybrid_results = hybrid_manager.hybrid_search(
                    search_query,
                    top_n=top_n,
                    alpha=0.5,
                    semantic_top_n=top_n * 2,
                    bm25_top_n=top_n * 2,
                )
                # Convert to common format
                for result in hybrid_results:
                    kb_results.append({
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                        "similarity": result.get("similarity", 0),
                    })
                logger.info(f"[HybridSearch] Returned {len(kb_results)} results for query")
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to ChromaDB-only: {e}")
                kb_results = []
    
    # Fallback to ChromaDB-only if hybrid search disabled or failed
    if not kb_results:
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
                
                # Apply semantic re-ranking for better accuracy
                if kb_results:
                    kb_results = _semantic_rerank(query, kb_results, top_n)
                    logger.debug(f"After re-ranking: {len(kb_results)} results")
            except Exception as e:
                logger.warning(f"ChromaDB similarity search failed: {e}")

    # Chat history context is now handled by LangChain ConversationBufferMemory
    # No need for separate Redis chat_history semantic search

    # Process KB results into text chunks with source URLs
    processed_kb = []
    logger.info(f"[SourceURL_DEBUG] Processing {len(kb_results)} KB results for source URLs")
    for idx, item in enumerate(kb_results):
        text = item.get("text") or item.get("response") or item.get("query") or ""
        if text:
            # Include source URL for traceability if available
            metadata = item.get("metadata", {})
            # Check common keys for URL
            source_url = (
                metadata.get("url") or 
                metadata.get("source_url") or 
                metadata.get("source") or 
                metadata.get("link") or 
                ""
            )
            
            # Debug log to trace missing URLs
            if not source_url and idx < 3:
                logger.warning(f"[SourceURL_MISSING] Item {idx} metadata: {list(metadata.keys())}")
            
            # Validate URL format before including
            if source_url and isinstance(source_url, str):
                # Clean up URL if needed
                processed_kb.append(f"{text}\n[Source: {source_url}]")
            else:
                processed_kb.append(text)

    # Deduplicate results
    seen = set()
    merged: List[str] = []
    for t in processed_kb:
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


# NOTE: summarize_chunks_with_llm removed in v3.1.0
# LangChain ConversationSummaryBufferMemory handles summarization incrementally

