from typing import Optional
from app.utils.redis_context import get_chat_history, summarize_chunks_with_llm


def build_llm_context_from_history(session_id: str, query: Optional[str] = None) -> str:
    """
    Build a robust LLM context from chat history:
    - Summarize all previous messages except the latest user message.
    - Include the latest user message in full.
    - Returns a string suitable for LLM prompt context.
    """
    chat_history = get_chat_history(session_id)
    if not chat_history:
        return ""

    # Find the latest user message (from the end)
    latest_user_idx = None
    for idx in range(len(chat_history) - 1, -1, -1):
        role = chat_history[idx].get("role") or chat_history[idx].get("sender")
        if role == "user":
            latest_user_idx = idx
            break

    if latest_user_idx is None:
        # No user message found, summarize all
        summary = summarize_chunks_with_llm([
            m.get("content", "") for m in chat_history if m.get("content")
        ], query or "")
        return summary

    # Split history: all before latest user message, and the latest user message
    earlier_msgs = chat_history[:latest_user_idx]
    latest_user_msg = chat_history[latest_user_idx]

    # Summarize earlier messages (if any)
    summary = ""
    if earlier_msgs:
        summary = summarize_chunks_with_llm([
            m.get("content", "") for m in earlier_msgs if m.get("content")
        ], query or latest_user_msg.get("content", ""))

    # Build context: summary (if any) + latest user message
    context_parts = []
    if summary:
        context_parts.append(f"Summary of previous conversation:\n{summary}")
    context_parts.append(f"Latest user message:\n{latest_user_msg.get('content', '')}")
    return "\n\n".join(context_parts)
