from app.logger import get_logger
from typing import List, Dict, Any, Optional
from app.db.redis_vector_helper import similarity_search
from app.core.llm_client import call_llm_summarize_chunks

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
    Retrieve context chunks from Redis using semantic search.

    This function attempts to infer the user's intent from the query and
    conversation history and builds a search query that favors business or
    product-context where appropriate rather than always driving the query to
    low-level technical terms.
    """
    # Default domain-aware keywords if none provided
    default_key_terms = ["Intent-aware search query", "healthcare", "development", "software", "tech stack", "case study", "portfolio"]
    key_terms = key_terms or default_key_terms

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

    search_terms = q.lower()

    # Heuristics to infer intent: prefer business/interest intent over technical
    business_indicators = [
        "price", "pricing", "cost", "quote", "hire", "partner", "collaborat", "interested", "options", "services", "solutions", "contact", "demo", "case study", "portfolio", "examples", "recommend"
    ]
    technical_indicators = [
        "api", "integration", "backend", "database", "error", "bug", "deploy", "server", "framework", "sdk", "implementation", "architecture"
    ]

    # Decide primary intent
    is_business = any(tok in search_terms for tok in business_indicators)
    is_technical = any(tok in search_terms for tok in technical_indicators)

    # Build a neutral, intent-aware search query
    if is_business and not is_technical:
        # Business intent: prefer case studies, services, outcomes
        search_query = f"{domain_prefix} services case study {q}"
    elif is_technical and not is_business:
        # Technical intent: include technical details
        search_query = f"{domain_prefix} {q} technical details"
    else:
        # Ambiguous or general intent: prefer product/context and let similarity search handle specifics
        search_query = f"{domain_prefix} {q}"

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

    logger.info(f"[RedisContext] Intent-aware search query: {search_query}")

    # Perform similarity search with enhanced query
    results = similarity_search(session_id, search_query, top_n=top_n)

    if not results:
        logger.info("[RedisContext] No context retrieved from Redis.")
        return []

    # Process and deduplicate results, prioritizing domain-relevant content
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

        # If the chunk mentions the domain prefix or other key_terms, push it forward
        if any(term in key for term in key_terms):
            normalized.insert(0, text)
        else:
            normalized.append(text)
        seen_content.add(key)

    if normalized:
        logger.info(f"[RedisContext] Selected {len(normalized)} relevant context chunks")

    logger.info(f"[RedisContext] Retrieved {len(normalized)} context items (top_n={top_n})")

    if normalized:
        summary = summarize_chunks_with_llm(normalized[:4], query)
        return [summary] if summary else normalized[:1]
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
        "Context Chunks:\n"
        + "\n---\n".join(chunks)
    )
    try:
        summary = call_llm_summarize_chunks(prompt)
        return summary.strip() if summary else ""
    except Exception as e:
        logger.error(f"[RedisContext] LLM summarization failed: {e}")
        return ""
 
