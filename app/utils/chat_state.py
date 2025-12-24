from typing import Optional
from app.utils.redis_context import get_chat_history
from app.utils.llm_client import call_llm_conversation_summary
from app.logger import get_logger

logger = get_logger("chat_state")


def build_llm_context_from_history(session_id: str, query: Optional[str] = None) -> str:
    """
    Build a robust LLM context from chat history using conversation-aware summarization.
    
    This function:
    1. Retrieves chat history from Redis/DB
    2. Uses a specialized LLM prompt to extract ALL user-provided details
    3. Returns a structured summary that preserves:
       - User details (name, email, location, industry, etc.)
       - Meeting availability and preferences
       - Questions already asked to prevent repetition
       - Key conversation context
    
    Returns a string suitable for LLM prompt context.
    """
    chat_history = get_chat_history(session_id)
    if not chat_history:
        logger.info(f"[ChatState] No chat history found for session {session_id}")
        return ""

    # For very short conversations (1-2 messages), skip summarization
    # Just return the messages directly to save LLM call
    if len(chat_history) <= 2:
        logger.info(f"[ChatState] Short conversation ({len(chat_history)} msgs), skipping summary")
        parts = []
        for msg in chat_history:
            role = msg.get("role") or msg.get("sender", "unknown")
            content = msg.get("content", "")
            if content:
                parts.append(f"{role.capitalize()}: {content}")
        return "\n".join(parts)

    # For longer conversations, use the conversation-aware summarizer
    logger.info(f"[ChatState] Building conversation summary for {len(chat_history)} messages")
    
    # Use the specialized conversation summarizer that extracts user details
    summary = call_llm_conversation_summary(chat_history)
    
    if summary:
        logger.info(f"[ChatState] Generated structured summary ({len(summary)} chars)")
        # Log a preview for debugging
        preview = summary[:300] + "..." if len(summary) > 300 else summary
        logger.debug(f"[ChatState] Summary preview: {preview}")
        return summary
    
    # Fallback: If summarization fails, return basic formatted history
    logger.warning("[ChatState] Summary generation failed, using basic format")
    return _format_history_basic(chat_history)


def _format_history_basic(chat_history: list) -> str:
    """
    Basic fallback formatter for chat history.
    Used when LLM summarization fails.
    """
    if not chat_history:
        return ""
    
    parts = []
    for msg in chat_history:
        role = msg.get("role") or msg.get("sender", "unknown")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(parts)
