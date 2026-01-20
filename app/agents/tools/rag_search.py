"""
RAG Search Tool

Wraps existing Redis/RAG infrastructure as a LangGraph tool.
Enables the agent to search the knowledge base dynamically.
"""

from typing import Optional
from langchain_core.tools import tool
from app.logger import get_logger

logger = get_logger("agent_tools")


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
    try:
        from app.utils.redis_context import get_redis_context_chunks
        
        # Use None for session_id - the RAG doesn't need session tracking
        chunks = get_redis_context_chunks(
            session_id=None,
            query=query,
            top_n=top_k
        )
        
        if not chunks:
            logger.info(f"[RAG Tool] No results for: {query[:50]}...")
            return ""
        
        # Format results
        context = "\n\n---\n\n".join(str(chunk) for chunk in chunks)
        logger.info(f"[RAG Tool] Found {len(chunks)} results for: {query[:50]}...")
        
        return context
        
    except Exception as e:
        logger.error(f"[RAG Tool] Search failed: {e}")
        return ""
