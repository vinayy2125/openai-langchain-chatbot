import json
from app.logger import get_logger
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.core.nested_follow_up_manager import FollowUpManager

# Initialize logger
logger = get_logger(__name__)


async def build_chatbot_response(
    session_id: str,
    follow_up_manager: FollowUpManager,
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
                
                # Forward all events to the caller (including form_trigger)
                # If upstream emits session-level metadata, persist it in manager
                if isinstance(chunk, dict) and chunk.get("status") == "meta":
                    try:
                        raw_meta = chunk.get("chunk") or {}
                        # Coerce meta into dict if it's a JSON string
                        meta = raw_meta
                        if isinstance(raw_meta, str):
                            try:
                                meta = json.loads(raw_meta)
                            except Exception:
                                meta = {}

                        if not isinstance(meta, dict):
                            meta = {}

                        session = follow_up_manager.get_session_data(session_id)
                        # Only set flag once per session
                        if meta.get("user_details_known"):
                            session.setdefault("state", {})["user_details_known"] = True
                        if meta.get("user_network_id"):
                            session.setdefault("state", {})["user_network_id"] = meta.get("user_network_id")
                        try:
                            follow_up_manager.set_session_data(session_id, session)
                        except Exception:
                            follow_up_manager.sessions[session_id] = session
                        # don't forward meta to client
                        continue
                    except Exception:
                        # fallthrough and forward if something goes wrong
                        pass

                yield chunk
                
                # If upstream emits structured dict events, prefer those for accumulation
                if isinstance(chunk, dict):
                    if chunk.get("status") == "chunk":
                        text_chunk = str(chunk.get("chunk", "")) or ""
                        main_response += text_chunk
                    # Don't accumulate form_trigger or other special events
                else:
                    text_chunk = str(chunk)
                    if text_chunk:
                        main_response += text_chunk

            # Persist assistant message to session history
            if main_response:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", main_response)

        except Exception as e:
            logger.exception("[ChatLogic] Unified response generation failed: %s", e)
            yield {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("[ChatLogic] Unexpected failure in build_chatbot_response: %s", e)
        yield {"status": "error", "message": str(e)}
        
