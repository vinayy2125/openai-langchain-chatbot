from app.logger import get_logger
from typing import List, Dict, Any
from app.db.redis_vector_helper import similarity_search
from app.core.llm_client import call_llm_summarize_chunks
logger = get_logger("chatbot")

def get_redis_context_chunks(session_id: str, query: str, conversation_history: List[Dict[str, Any]], top_n: int = 4) -> List[str]:
    """
    Retrieve context chunks from Redis using semantic search, prioritizing Ditstek-specific content.
    Returns deduplicated, relevant chunks from the knowledge base.
    """
    # Extract key terms for better context matching
    key_terms = ["ditstek", "healthcare", "development", "software", "tech stack", "case study", "portfolio"]
    search_terms = query.lower()
    
    # Add domain-specific context if relevant
    if any(term in search_terms for term in key_terms):
        search_query = f"ditstek {query} technical details case studies"
    else:
        search_query = f"ditstek capabilities {query}"
    
    # Get most recent relevant user query for context
    if conversation_history:
        for item in reversed(conversation_history[-2:]):  # Only last 2 messages
            try:
                if isinstance(item, dict) and item.get("role") == "user":
                    content = item.get("content")
                    if content and any(term in content.lower() for term in key_terms):
                        search_query = f"{search_query} {content.strip()}"
                        break  # Only get the most relevant recent message
            except Exception:
                continue
    
    # Perform similarity search with enhanced query
    logger.info(f"[RedisContext] Enhanced search query: {search_query}")
    results = similarity_search(session_id, search_query, top_n=top_n)
    
    if not results:
        logger.info("[RedisContext] No context retrieved from Redis.")
        return []

    # Process and deduplicate results, prioritizing high-quality content
    seen_content = set()
    normalized: List[str] = []
    
    for item in results:
        if not isinstance(item, dict):
            continue
            
        # Get content with fallbacks
        text = (item.get("response") or item.get("text") or item.get("query") or "").strip()
        
        # Skip if empty or already seen
        if not text or text.lower() in seen_content:
            continue
            
        # Prioritize content with Ditstek-specific information
        if any(term in text.lower() for term in key_terms):
            normalized.insert(0, text)  # Add to front
        else:
            normalized.append(text)  # Add to back
            
        seen_content.add(text.lower())
        
    # Log preview of selected context
    if normalized:
        logger.info(f"[RedisContext] Selected {len(normalized)} relevant context chunks")

    logger.info(f"[RedisContext] Retrieved {len(normalized)} context items (top_n={top_n})")
    # Summarize the top 4 chunks using the LLM, extracting links and details
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
 
