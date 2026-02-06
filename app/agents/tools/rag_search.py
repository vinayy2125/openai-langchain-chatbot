"""
RAG Search Tool

Wraps existing Redis/RAG infrastructure as a LangGraph tool.
Enables the agent to search the knowledge base dynamically.

Filters out already-shared content to prevent repetitive responses.
"""

import re
from typing import Optional, List
from langchain_core.tools import tool
from app.logger import get_logger

logger = get_logger("agent_tools")

# URL pattern for extracting/matching URLs in chunks
URL_PATTERN = re.compile(r'https?://[^\s\)\]>\"\']+|ditstek\.com[^\s\)\]>\"\']*', re.IGNORECASE)

# Global context for filtering (set before each agent invocation)
_shared_urls: List[str] = []


def set_search_context(session_id: str, shared_urls: List[str] = None) -> None:
    """
    Set the current session context for RAG search.
    
    This allows the tool to filter out already-shared content.
    Must be called before invoking the agent.
    
    Args:
        session_id: Current session ID (for logging)
        shared_urls: List of URLs already shared (to exclude from results)
    """
    global _shared_urls
    _shared_urls = shared_urls or []
    if _shared_urls:
        logger.debug(f"[RAG Tool] Context set: exclude {len(_shared_urls)} URLs")


def _extract_url_from_chunk(chunk: str) -> Optional[str]:
    """Extract first URL from a chunk if present."""
    match = URL_PATTERN.search(str(chunk).lower())
    return match.group(0) if match else None


@tool
def search_knowledge_base(query: str, top_k: int = 4) -> str:
    """
    Search the knowledge base for relevant information.
    
    Use this tool when the user asks a question that requires
    domain-specific knowledge like services, pricing, or company info.
    
    Args:
        query: The search query based on user's question
        top_k: Number of results to return (default 4)
        
    Returns:
        Relevant context from the knowledge base, or empty string if none found
    """
    global _shared_urls
    
    try:
        from app.utils.redis_context import get_redis_context_chunks
        
        # Request more results if we need to filter duplicates
        request_k = top_k + len(_shared_urls) if _shared_urls else top_k
        
        chunks = get_redis_context_chunks(
            session_id=None,
            query=query,
            top_n=request_k
        )
        
        if not chunks:
            logger.info(f"[RAG Tool] No results for: {query[:50]}...")
            return ""
        
        # Filter out already-shared content
        filtered_chunks = []
        duplicates_filtered = 0
        
        for chunk in chunks:
            if len(filtered_chunks) >= top_k:
                break
                
            chunk_url = _extract_url_from_chunk(str(chunk))
            
            # Check if this URL was already shared
            if chunk_url and _shared_urls:
                if any(shared in chunk_url or chunk_url in shared for shared in _shared_urls):
                    duplicates_filtered += 1
                    continue
            
            filtered_chunks.append(chunk)
        
        if not filtered_chunks:
            logger.info(f"[RAG Tool] All {len(chunks)} results were duplicates for: {query[:50]}...")
            return "I've already shared the main resources on this topic. Would you like to explore a different aspect, or shall we discuss how this applies to your specific situation?"
        
        # Format results
        context = "\n\n---\n\n".join(str(chunk) for chunk in filtered_chunks)
        
        if duplicates_filtered:
            logger.info(f"[RAG Tool] Found {len(filtered_chunks)} fresh results (filtered {duplicates_filtered} duplicates) for: {query[:50]}...")
        else:
            logger.info(f"[RAG Tool] Found {len(filtered_chunks)} results for: {query[:50]}...")
        
        return context
        
    except Exception as e:
        logger.error(f"[RAG Tool] Search failed: {e}")
        return ""
