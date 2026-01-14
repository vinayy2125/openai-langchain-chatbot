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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import time

# Single-threaded executor for background embedding tasks (small default)
_EMBEDDING_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Executor for parallel search strategies (3 workers for 3 search methods)
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="search")

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


# =============================================================================
# POOL-AND-RERANK SEARCH (Accuracy-First Architecture)
# =============================================================================
# Run all search strategies in parallel, wait for all, pool results, rerank.
# 
# ARCHITECTURE:
# 1. Run all retrievers in parallel (Parent, Hybrid, Chroma)
# 2. Wait for completed results (collect stragglers, don't block on slow ones)
# 3. Pool results: collect top K from each = 12 candidates
# 4. Normalize scores across retrievers (different scales)
# 5. Rerank using keyword boosting + hybrid preference
# 6. Filter by adaptive relevance threshold
# 7. Check evidence sufficiency
# 8. Return best top_n with confidence flag
#
# HARDENING (v2):
# - Straggler tolerance: collect completed, don't block on slow retriever
# - Adaptive threshold: shorter queries get lower threshold
# - Evidence sufficiency gate: flag when < 2 chunks pass threshold
# =============================================================================

# Base relevance threshold - adjusted by query characteristics
BASE_RELEVANCE_THRESHOLD = 0.25

# Minimum chunks needed for "sufficient evidence"
MIN_EVIDENCE_CHUNKS = 2

# Timeout settings
POOL_TIMEOUT_SECONDS = 2.0  # Max wait for all retrievers
STRAGGLER_GRACE_MS = 200    # Extra time to collect late arrivals


def _get_adaptive_threshold(query: str) -> float:
    """
    Compute adaptive relevance threshold based on query characteristics.
    
    Rationale:
    - Short queries (< 5 words) often have lower similarity scores
    - Very specific queries (with numbers, codes) need exact matches
    - Broad exploratory queries can accept looser thresholds
    
    Returns:
        Adaptive threshold in [0.15, 0.35] range
    """
    if not query:
        return BASE_RELEVANCE_THRESHOLD
    
    query_words = len(query.split())
    query_length = len(query)
    
    threshold = BASE_RELEVANCE_THRESHOLD
    
    # Short queries: lower threshold (they inherently score lower)
    if query_words <= 3:
        threshold -= 0.08
    elif query_words <= 5:
        threshold -= 0.04
    
    # Very long queries: raise threshold (more context = better matches expected)
    if query_words >= 15:
        threshold += 0.05
    
    # Queries with special markers (codes, numbers) need precision
    has_code_like = any(c.isdigit() for c in query) or any(c in query for c in ['_', '-', '.'])
    if has_code_like:
        threshold += 0.03
    
    # Clamp to safe range
    return max(0.15, min(0.35, threshold))


def _normalize_scores(results: List[Dict], source_name: str) -> List[Dict]:
    """
    Normalize similarity scores to 0-1 range for cross-retriever comparison.
    
    Different retrievers use different scoring scales:
    - ChromaDB: cosine distance (lower = more similar)
    - Hybrid: RRF scores (0-1 range typically)
    - Parent: cosine similarity (higher = more similar)
    """
    if not results:
        return results
    
    # Get all scores
    scores = [r.get("similarity", 0) for r in results]
    if not scores:
        return results
    
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    normalized = []
    for r in results:
        r_copy = r.copy()
        raw_score = r.get("similarity", 0)
        
        # Normalize to 0-1 range
        if score_range > 0:
            norm_score = (raw_score - min_score) / score_range
        else:
            norm_score = 1.0 if raw_score > 0 else 0.0
        
        r_copy["normalized_score"] = norm_score
        r_copy["raw_score"] = raw_score
        r_copy["retriever_source"] = source_name
        normalized.append(r_copy)
    
    return normalized


def _cross_retriever_rerank(
    query: str, 
    pooled_results: List[Dict], 
    top_n: int
) -> List[Dict]:
    """
    Rerank pooled results from multiple retrievers.
    
    Scoring:
    1. Normalized similarity score (from each retriever)
    2. Keyword match boost (exact query terms in text)
    3. Hybrid source bonus (most accurate retriever)
    """
    if not pooled_results:
        return []
    
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]
    
    scored = []
    for r in pooled_results:
        text = (r.get("text", "") or "").lower()
        
        # Base: normalized similarity (0-1)
        base_score = r.get("normalized_score", r.get("similarity", 0))
        
        # Boost 1: Query keyword matches (up to +0.15)
        keyword_matches = sum(1 for word in query_words if word in text)
        keyword_boost = min(keyword_matches * 0.03, 0.15)
        
        # Boost 2: Hybrid results get slight preference (most accurate retriever)
        source = r.get("retriever_source", r.get("source", ""))
        hybrid_bonus = 0.05 if source == "hybrid" else 0
        
        # Final score
        final_score = base_score + keyword_boost + hybrid_bonus
        
        r_copy = r.copy()
        r_copy["rerank_score"] = final_score
        scored.append(r_copy)
    
    # Sort by rerank score (highest first)
    scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    
    # Deduplicate by text content (keep highest scored version)
    seen_texts = set()
    deduped = []
    for r in scored:
        text_key = (r.get("text", "") or "")[:200]  # First 200 chars as key
        if text_key and text_key not in seen_texts:
            seen_texts.add(text_key)
            deduped.append(r)
    
    return deduped[:top_n]


def _parallel_kb_search(
    search_query: str,
    top_n: int,
    enable_parent: bool,
    enable_hybrid: bool,
) -> List[Dict]:
    """
    Pool-and-rerank search across multiple retrievers.
    
    Architecture (Accuracy-First):
    1. Run ALL retrievers in parallel
    2. Wait for ALL to complete (2s timeout per retriever)
    3. Pool results (collect top K from each)
    4. Normalize scores across retrievers
    5. Rerank by relevance
    6. Filter by threshold
    7. Return best top_n
    
    Args:
        search_query: Derived search query
        top_n: Final number of results to return (after reranking)
        enable_parent: Whether ParentRetriever is enabled
        enable_hybrid: Whether HybridSearch is enabled
        
    Returns:
        List of reranked, filtered result dicts
    """
    start_time = time.perf_counter()
    futures = {}
    
    # Collect more candidates from each retriever for pooling
    candidates_per_retriever = max(top_n, 4)
    
    # Define search functions that return (results, source_name)
    def parent_search():
        method_start = time.perf_counter()
        retriever = _get_parent_retriever()
        if not retriever:
            logger.debug(f"[PoolSearch] parent: no retriever ({time.perf_counter() - method_start:.3f}s)")
            return [], "parent"
        try:
            results = retriever.retrieve(search_query, k=candidates_per_retriever)
            formatted = [{
                "text": r.get("content", ""),
                "metadata": r.get("metadata", {}),
                "similarity": r.get("similarity", 0),
                "source": "parent",
            } for r in results]
            logger.info(f"[PoolSearch] parent: {len(formatted)} results ({time.perf_counter() - method_start:.3f}s)")
            return formatted, "parent"
        except Exception as e:
            logger.debug(f"ParentRetriever search error: {e} ({time.perf_counter() - method_start:.3f}s)")
            return [], "parent"
    
    def hybrid_search():
        method_start = time.perf_counter()
        manager = _get_hybrid_search_manager()
        if not manager:
            logger.debug(f"[PoolSearch] hybrid: no manager ({time.perf_counter() - method_start:.3f}s)")
            return [], "hybrid"
        try:
            results = manager.hybrid_search(
                search_query,
                top_n=candidates_per_retriever,
                alpha=0.5,
                semantic_top_n=candidates_per_retriever * 2,
                bm25_top_n=candidates_per_retriever * 2,
            )
            formatted = [{
                "text": r.get("text", ""),
                "metadata": r.get("metadata", {}),
                "similarity": r.get("similarity", 0),
                "source": "hybrid",
            } for r in results]
            logger.info(f"[PoolSearch] hybrid: {len(formatted)} results ({time.perf_counter() - method_start:.3f}s)")
            return formatted, "hybrid"
        except Exception as e:
            logger.debug(f"HybridSearch error: {e} ({time.perf_counter() - method_start:.3f}s)")
            return [], "hybrid"
    
    def chroma_search():
        method_start = time.perf_counter()
        chroma = _get_chroma_manager()
        if not chroma:
            logger.debug(f"[PoolSearch] chroma: no manager ({time.perf_counter() - method_start:.3f}s)")
            return [], "chroma"
        try:
            results = chroma.similarity_search(search_query, n_results=candidates_per_retriever)
            formatted = [{
                "text": r.get("text", ""),
                "metadata": r.get("metadata", {}),
                "similarity": r.get("similarity", 0),
                "source": "chroma",
            } for r in results]
            logger.info(f"[PoolSearch] chroma: {len(formatted)} results ({time.perf_counter() - method_start:.3f}s)")
            return formatted, "chroma"
        except Exception as e:
            logger.debug(f"ChromaDB search error: {e} ({time.perf_counter() - method_start:.3f}s)")
            return [], "chroma"
    
    # Submit ALL enabled search methods
    if enable_parent:
        futures[_SEARCH_EXECUTOR.submit(parent_search)] = "parent"
    
    if enable_hybrid:
        futures[_SEARCH_EXECUTOR.submit(hybrid_search)] = "hybrid"
    
    # ChromaDB is always included
    futures[_SEARCH_EXECUTOR.submit(chroma_search)] = "chroma"
    
    if not futures:
        return []
    
    # ==========================================================================
    # STRAGGLER-TOLERANT WAIT STRATEGY
    # ==========================================================================
    # Don't block on slow retrievers. Collect what's done, give stragglers
    # a brief grace period, then proceed with available results.
    # Accuracy requires sufficiency, not unanimity.
    # ==========================================================================
    pooled_results = []
    source_counts = {}
    
    try:
        # Phase 1: Wait for first results to complete
        remaining = set(futures.keys())
        deadline = time.perf_counter() + POOL_TIMEOUT_SECONDS
        
        while remaining and time.perf_counter() < deadline:
            # Wait with short timeout to allow collecting as results complete
            wait_time = min(0.5, deadline - time.perf_counter())
            if wait_time <= 0:
                break
                
            done, remaining = wait(remaining, timeout=wait_time, return_when=FIRST_COMPLETED)
            
            # Collect completed results immediately
            for future in done:
                try:
                    results, source = future.result(timeout=0.1)
                    if results:
                        normalized = _normalize_scores(results, source)
                        pooled_results.extend(normalized)
                        source_counts[source] = len(results)
                except Exception as e:
                    logger.debug(f"Future result error: {e}")
            
            # If we have enough evidence, don't wait for stragglers
            if len(source_counts) >= 2 and len(pooled_results) >= MIN_EVIDENCE_CHUNKS * 2:
                # Give stragglers brief grace period
                if remaining:
                    grace_done, still_pending = wait(
                        remaining, 
                        timeout=STRAGGLER_GRACE_MS / 1000, 
                        return_when=FIRST_COMPLETED
                    )
                    for future in grace_done:
                        try:
                            results, source = future.result(timeout=0.1)
                            if results:
                                normalized = _normalize_scores(results, source)
                                pooled_results.extend(normalized)
                                source_counts[source] = len(results)
                        except Exception:
                            pass
                    # Cancel remaining stragglers
                    for f in still_pending:
                        f.cancel()
                break
        
        # Cancel any still-pending futures
        for f in remaining:
            f.cancel()
            
        if remaining:
            timed_out = [futures[f] for f in remaining if f in futures]
            if timed_out:
                logger.debug(f"[PoolSearch] Proceeding without stragglers: {timed_out}")
    
    except Exception as e:
        logger.warning(f"[PoolSearch] Exception during wait: {e}")
    
    pool_time = time.perf_counter() - start_time
    
    if not pooled_results:
        logger.warning(f"[PoolSearch] All retrievers returned empty in {pool_time:.3f}s")
        return []
    
    # ==========================================================================
    # RERANK + ADAPTIVE THRESHOLD + EVIDENCE SUFFICIENCY
    # ==========================================================================
    rerank_start = time.perf_counter()
    reranked = _cross_retriever_rerank(search_query, pooled_results, top_n * 2)
    
    # Compute adaptive threshold based on query
    threshold = _get_adaptive_threshold(search_query)
    
    # FILTER by adaptive relevance threshold
    filtered = [
        r for r in reranked 
        if r.get("rerank_score", r.get("normalized_score", 0)) >= threshold
    ]
    
    # ==========================================================================
    # EVIDENCE SUFFICIENCY GATE
    # ==========================================================================
    # Flag as low_confidence if we don't have enough high-quality chunks
    low_confidence = False
    
    if len(filtered) < MIN_EVIDENCE_CHUNKS:
        low_confidence = True
        if not filtered and reranked:
            # Fallback: use top results but mark as low confidence
            logger.warning(
                f"[PoolSearch] Insufficient evidence: {len(filtered)} chunks above threshold {threshold:.2f}, "
                f"falling back to top results (LOW CONFIDENCE)"
            )
            filtered = reranked[:top_n]
        elif filtered:
            logger.info(
                f"[PoolSearch] Low evidence: only {len(filtered)} chunks above threshold {threshold:.2f} "
                f"(need {MIN_EVIDENCE_CHUNKS})"
            )
    
    final_results = filtered[:top_n]
    
    # Tag results with confidence metadata
    for r in final_results:
        r["_low_confidence"] = low_confidence
        r["_threshold_used"] = threshold
    
    total_time = time.perf_counter() - start_time
    rerank_time = time.perf_counter() - rerank_start
    
    confidence_tag = " [LOW CONFIDENCE]" if low_confidence else ""
    logger.info(
        f"[PoolSearch] Pooled {len(pooled_results)} from {source_counts}, "
        f"threshold={threshold:.2f}, filtered={len(filtered)}, "
        f"returned={len(final_results)}{confidence_tag} "
        f"({total_time:.3f}s)"
    )
    
    return final_results


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

    # Perform knowledge base search using parallel execution
    # All enabled strategies run concurrently; first non-empty result wins
    kb_results = _parallel_kb_search(
        search_query=search_query,
        top_n=top_n,
        enable_parent=ENABLE_PARENT_RETRIEVER,
        enable_hybrid=ENABLE_HYBRID_SEARCH,
    )
    
    # Apply semantic re-ranking for better accuracy (if results came from ChromaDB-only)
    # ParentRetriever and Hybrid already have their own ranking
    if kb_results and len(kb_results) > 0:
        source = kb_results[0].get("source", "unknown")
        if source == "chroma":
            kb_results = _semantic_rerank(query, kb_results, top_n)
            logger.debug(f"After re-ranking ChromaDB results: {len(kb_results)} results")

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

