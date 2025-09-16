from backend.llm_client import llm
from backend.services.chatbot_optimizer import OptimizedChatbot
from backend.nested_follow_up_manager import FollowUpManager

import json
import re
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Utility helpers (restored minimal versions for optimizer imports)
def _maybe_expand_queries(query: str) -> list:
    """Return lightweight expanded query variants (deduplicated)."""
    variants = [
        query,
        f"Details about {query}",
        f"In-depth explanation of {query}"
    ]
    # Preserve order while deduping
    seen = set()
    deduped = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped

def _dedupe_chunks(docs) -> list:
    """Simplify dedupe: accepts list of doc objects or strings, returns list of (text, meta)."""
    seen = set()
    result = []
    for d in docs:
        text = getattr(d, 'page_content', str(d)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        meta = getattr(d, 'metadata', {}) if hasattr(d, 'metadata') else {}
        result.append((text, meta))
    return result

async def build_chatbot_response(
    session_id: str,
    follow_up_manager,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    prompt_context: Optional[str] = None,
    mode: str = "complete"
) -> AsyncGenerator[str, None]:
    """
    Build a streaming response from the chatbot that handles both follow-up generation and complete responses.
    Uses OptimizedChatbot for context optimization and token management.

    Args:
        session_id: The session identifier
        follow_up_manager: Instance of FollowUpManager
        conversation_history: List of conversation messages
        prompt_context: Original prompt context
        mode: Either "follow_up" or "complete" to determine response type

    Yields:
        Formatted SSE messages for streaming response
    """
    try:
        # Get session data and validate
        session_data = follow_up_manager.get_session_data(session_id)
        if not session_data:
            yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
            return

        # Get the latest user query from conversation history
        latest_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    latest_query = msg.get("content", "")
                    break
        # Fallback to prompt_context snippet when starting with prompt selection only
        if not latest_query:
            if prompt_context:
                latest_query = prompt_context.split("\n")[0][:140]
            else:
                latest_query = "User requirements clarification"
        if not latest_query and mode != "follow_up":
            yield f"data: {json.dumps({'error': 'No query found in conversation'})}\n\n"
            return

        # Prepare chat history
        chat_history = [(msg["role"], msg["content"]) for msg in (conversation_history or [])]

        # Get streaming response from chatbot
        response_stream = follow_up_manager.chatbot.get_detailed_response(
            query=latest_query,
            chat_history=chat_history,
            stream=True
        )

        # Notify that processing has started (different message for follow-up)
        if mode == "follow_up":
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Preparing follow-up question...'})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating response...'})}\n\n"

        if mode == "follow_up":
            # Use optimized chatbot's follow-up streaming instead of full answer generation
            full_text = ""
            yield f"data: {json.dumps({'status': 'follow_up_chunk', 'chunk': ''})}\n\n"
            stream_gen = follow_up_manager.chatbot.stream_follow_up_generation(
                conversation_history=conversation_history or [],
                latest_query=latest_query,
                prompt_context=prompt_context or ""
            )
            for token in stream_gen:
                if not token:
                    continue
                full_text += token
                yield f"data: {json.dumps({'status': 'follow_up_chunk', 'chunk': token})}\n\n"

            cleaned = full_text.strip()
            # If streaming produced nothing, perform a synchronous fallback generation
            if not cleaned:
                fallback_prompt = (
                    "Produce 3-5 numbered targeted follow-up questions to clarify user needs. "
                    "Format exactly like: 1. \"First question?\"\n2. ... No extra commentary."
                )
                try:
                    sync_resp = follow_up_manager.llm.invoke(fallback_prompt)
                    text = getattr(sync_resp, 'content', str(sync_resp))
                except Exception:
                    text = (
                        '1. "What specific problem are you trying to solve?"\n'
                        '2. "Who will use this solution?"\n'
                        '3. "What is the desired timeline?"'
                    )
                cleaned = text.strip()
                # Re-stream fallback tokens so client still sees chunks
                for tok in re.findall(r'\s*\S+', cleaned):
                    yield f"data: {json.dumps({'status': 'follow_up_chunk', 'chunk': tok})}\n\n"
            if cleaned:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", cleaned)
                follow_up_payload = {
                    'status': 'follow_up',
                    'message': 'Follow-up ready',
                    'follow_up': {
                        'type': 'clarification',
                        'question': cleaned,
                        'context': 'Gathering more information to provide a complete response',
                        'options': None
                    },
                    'content': None,
                    'suggestions': None,
                    'sources': None,
                    'conversation_history': conversation_history or []
                }
                yield f"data: {json.dumps(follow_up_payload)}\n\n"

        else:  # mode == "complete"
            complete_response = ""
            # Stream chunked completion similar to follow_up_chunk pattern
            for chunk in response_stream:  # synchronous generator
                if not chunk:
                    continue
                text_chunk = str(chunk)
                complete_response += text_chunk
                yield f"data: {json.dumps({'status': 'complete_chunk', 'chunk': text_chunk})}\n\n"

            if complete_response:
                # Save full response
                follow_up_manager.add_to_conversation_history(session_id, "assistant", complete_response)
                final_payload = {
                    'status': 'complete',
                    'content': complete_response,
                    'final': True
                }
                yield f"data: {json.dumps(final_payload)}\n\n"

    except Exception as e:
        error_msg = f"Error in build_chatbot_response: {str(e)}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"



# ✅ ADD - Fallback function (simplified version of your original logic)
def _fallback_to_original(query: str, chat_history: list, site: str):
    """Fallback disabled (legacy dependencies removed)."""
    return "Fallback not available.", False

def get_prompt_response(session_id: str, selected_prompt_id: int):
    """
    Return nested follow-ups for a given prompt ID.
    Include detailed answer if requested.
    """
    # NOTE: Legacy DB prompt retrieval below is currently disabled (dependencies missing).
    return None, "Legacy prompt retrieval disabled", False

    # conn = _get_conn()
    # cursor = conn.cursor()

    # Fetch the selected prompt
    # (Disabled legacy implementation)

# Legacy synchronous variant (renamed to avoid clashing with async build_chatbot_response used by FollowUpManager)
def legacy_build_chatbot_response(query, history):
    # Initialize meta
    meta = {}

    # Enforce website-only responses
    if not meta.get("used_web"):
        return "I don’t have this detail.", False, meta

    # ...existing logic...
    return "Response generated.", True, meta
