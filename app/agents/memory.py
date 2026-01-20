"""
Hybrid Memory Manager

Implements a cost-efficient memory strategy combining:
1. Buffer Memory: Recent messages for immediate context
2. Summarization: Compressed older history
3. Integration with existing Redis/conversation memory
"""

from typing import List, Optional
from app.logger import get_logger

logger = get_logger("agent_memory")


class HybridMemory:
    """
    Manages conversation memory with summarization for long sessions.
    
    This prevents context window overflow and reduces API costs while
    maintaining conversation coherence.
    """
    
    def __init__(self, buffer_size: int = 6, summary_threshold: int = 10):
        """
        Initialize hybrid memory.
        
        Args:
            buffer_size: Number of recent messages to keep in full
            summary_threshold: Trigger summarization after this many messages
        """
        self.buffer_size = buffer_size
        self.summary_threshold = summary_threshold
        self._summaries: dict = {}  # session_id -> summary string
    
    def get_context(self, session_id: str, messages: List[dict]) -> dict:
        """
        Get optimized context for LLM call.
        
        Returns recent messages plus summary of older content.
        
        Args:
            session_id: Session identifier
            messages: Full message history
            
        Returns:
            dict with 'recent_messages' and 'summary'
        """
        if len(messages) <= self.buffer_size:
            return {
                "recent_messages": messages,
                "summary": None
            }
        
        recent = messages[-self.buffer_size:]
        older = messages[:-self.buffer_size]
        
        # Get or create summary for older messages
        summary = self._get_or_create_summary(session_id, older)
        
        return {
            "recent_messages": recent,
            "summary": summary
        }
    
    def _get_or_create_summary(self, session_id: str, messages: List[dict]) -> str:
        """
        Get cached summary or create new one.
        
        For now, uses simple concatenation. Can be upgraded to LLM summarization.
        """
        if session_id in self._summaries:
            return self._summaries[session_id]
        
        # Simple summary: extract key points
        points = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]  # Truncate
            if role == "user":
                points.append(f"User: {content}")
            elif role == "assistant":
                points.append(f"Bot: {content[:50]}...")
        
        summary = " | ".join(points[-5:])  # Last 5 key exchanges
        self._summaries[session_id] = summary
        
        logger.info(f"[Memory] Created summary for session {session_id[:8]}...")
        return summary
    
    def update_summary(self, session_id: str, new_summary: str) -> None:
        """Update cached summary for a session."""
        self._summaries[session_id] = new_summary
    
    def clear(self, session_id: str) -> None:
        """Clear memory for a session."""
        self._summaries.pop(session_id, None)


# Singleton instance
_memory_instance: Optional[HybridMemory] = None


def get_hybrid_memory() -> HybridMemory:
    """Get singleton hybrid memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = HybridMemory()
    return _memory_instance
