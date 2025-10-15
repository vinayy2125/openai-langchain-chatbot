import json
from app.logger import get_logger
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.core.response_formatter import format_response

# Initialize logger
logger = get_logger(__name__)


async def build_chatbot_response(
    session_id: str,
    follow_up_manager,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    prompt_context: Optional[str] = None,
    mode: str = "complete",
) -> AsyncGenerator[Any, None]:

    try:
        # Get session data and validate
        session_data = follow_up_manager.get_session_data(session_id)
        if not session_data:
            yield f"{json.dumps({'error': 'Session not found'})}\n\n"
            return

        # Get the latest user query from conversation history
        latest_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    latest_query = msg.get("content", "")
                    break
        if not latest_query and prompt_context:
            latest_query = prompt_context.split("\n")[0][:140]

        try:
            response_stream = follow_up_manager.chatbot.get_detailed_response(
                query=latest_query,
                chat_history=[(msg["role"], msg["content"]) for msg in (conversation_history or [])],
                session_id=session_id,
                stream=True,
            )

            main_response = ""
            for chunk in response_stream:
                if not chunk:
                    continue
                # If upstream emits structured dict events, prefer those
                if isinstance(chunk, dict) and "chunk" in chunk:
                    text_chunk = str(chunk["chunk"]) or ""
                else:
                    text_chunk = str(chunk)

                if not text_chunk:
                    continue

                # Keep a concatenated copy for history
                main_response += text_chunk

                # Apply response formatting per chunk for consistency
                formatted = format_response(text_chunk, latest_query, conversation_history)
                yield {"status": "chunk", "chunk": formatted}

            # Final completion event
            yield {"status": "complete_chunk", "chunk": ""}

            # Persist assistant message to session history
            if main_response:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", main_response)

        except Exception as e:
            logger.exception("[ChatLogic] Unified response generation failed: %s", e)
            yield {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("[ChatLogic] Unexpected failure in build_chatbot_response: %s", e)
        yield {"status": "error", "message": str(e)}
        
