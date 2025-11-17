from app.logger import get_logger
from typing import List, Dict, Any, Optional
from app.db.redis_vector_helper import similarity_search
from app.utils.llm_client import call_llm_summarize_chunks

logger = get_logger("chatbot")


def get_redis_context_chunks(
    session_id: str,
    query: str,
    conversation_history: List[Dict[str, Any]],
    top_n: int = 4,
    key_terms: Optional[List[str]] = None,
    domain_prefix: str = "",
    fallback_keywords: str = "capabilities",
) -> List[str]:
    """
    Retrieve context chunks from Redis using dynamic semantic similarity search.

    Uses semantic similarity search to find relevant chunks from the knowledge base
    dynamically based on query meaning and context. No static keyword matching -
    relies on vector similarity to find the most relevant content.

    Args:
        session_id: Session identifier
        query: User query (semantic search handles intent inference)
        conversation_history: Previous conversation messages for context
        top_n: Number of top results to retrieve (calculated dynamically by caller)
        key_terms: Optional domain-specific terms (if provided, used for filtering)
        domain_prefix: Optional domain prefix for search query
        fallback_keywords: Fallback if query is empty

    Returns:
        List of relevant context chunks, ordered by semantic similarity
    """
    # Use key_terms if provided, otherwise rely on semantic similarity alone
    # Semantic search will handle relevance ranking dynamically

    q = (query or "").strip()
    # If the explicit query is empty, try to derive a reasonable search term
    # from recent user messages in the conversation_history. If that also
    # yields nothing, fall back to the provided fallback_keywords so the
    # similarity search is never executed with an empty query.
    if not q and conversation_history:
        try:
            for item in reversed(conversation_history):
                if isinstance(item, dict) and item.get("role") == "user":
                    content = (item.get("content") or "").strip()
                    if content:
                        q = content[:300]
                        break
        except Exception:
            q = q

    if not q:
        logger.info("[RedisContext] Empty query; using fallback keywords for search.")
        q = fallback_keywords or "capabilities"

    # Let semantic similarity search handle intent inference dynamically
    # Build search query from user query and context - semantic search will find relevant chunks
    search_query = f"{domain_prefix} {q}" if domain_prefix else q

    # Augment search with recent user messages in the conversation (last 2)
    if conversation_history:
        for item in reversed(conversation_history[-2:]):
            try:
                if isinstance(item, dict) and item.get("role") == "user":
                    content = (item.get("content") or "").strip()
                    if content:
                        # Only append short recent content to keep search focused
                        if len(content) < 300:
                            search_query = f"{search_query} {content}"
                            break
            except Exception:
                continue

    logger.info(f"[RedisContext] Semantic search query: {search_query}, top_n={top_n}")

    # Perform similarity search with enhanced query
    results = similarity_search(session_id, search_query, top_n=top_n)

    if not results:
        logger.info("[RedisContext] No context retrieved from Redis.")
        return []

    # Process and deduplicate results - semantic search already ranks by relevance
    # Maintain the order from similarity search as it's already optimized
    seen_content = set()
    normalized: List[str] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        text = (
            item.get("response") or item.get("text") or item.get("query") or ""
        ).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_content:
            continue

        # Preserve semantic search ranking - similarity search already ordered by relevance
        normalized.append(text)
        seen_content.add(key)

    if normalized:
        logger.info(
            f"[RedisContext] Selected {len(normalized)} relevant context chunks"
        )

    logger.info(
        f"[RedisContext] Retrieved {len(normalized)} context items (top_n={top_n})"
    )

    if normalized:
        # Skip LLM summarization to avoid blocking - use raw chunks directly for faster response
        # LLM summarization adds 5-10 seconds delay - return chunks directly instead
        # Return first few chunks (limit to avoid token bloat) - main LLM will process them
        max_chunks = min(len(normalized), 5)  # Limit to 5 chunks max for performance
        return normalized[:max_chunks]
    return []


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
