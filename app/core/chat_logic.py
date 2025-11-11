import json
from app.logger import get_logger
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.core.nested_follow_up_manager import FollowUpManager
 # session_manager import removed
import threading


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

        # Always resolve and use the full conversation history (user and assistant messages)
        # Use resolve_history for general context (allows client optimization)
        conversation_history = follow_up_manager.resolve_history(session_id, conversation_history)
        logger.info(f"[CHAT_LOGIC] Resolved conversation history before LLM call: {len(conversation_history)} messages")
        logger.info(f"[CHAT_LOGIC] Complete conversation history for session {session_id}: {conversation_history}")
        latest_query = ""
        for msg in reversed(conversation_history or []):
            if msg.get("role") == "user":
                latest_query = msg.get("content", "")
                break

        if not latest_query and prompt_context:
            # Smarter prompt_context handling:
            # - split into non-empty lines
            # - prefer the first substantive line (length > 20 and not a simple heading like 'Goals:')
            # - otherwise join the first up to 3 lines to form a concise query
            try:
                lines = [l.strip() for l in prompt_context.splitlines() if l.strip()]
                chosen = None
                for line in lines:
                    # skip short headings that end with ':' or are very short
                    if len(line) > 20 and not line.endswith(":"):
                        chosen = line
                        break
                if not chosen and lines:
                    # pick the longest line as a fallback
                    chosen = max(lines, key=lambda s: len(s))

                if chosen:
                    latest_query = chosen[:300]
                else:
                    # join first few lines (up to 3) to make a composite query
                    latest_query = " ".join(lines[:3])[:300]
            except Exception:
                latest_query = prompt_context.split("\n")[0][:140]

        logger.info(f"[build_chatbot_response] Using latest_query='{latest_query}' prompt_context_provided={'yes' if prompt_context else 'no'} prompt_context_lines={len(prompt_context.splitlines()) if prompt_context else 0}")

        try:
            response_stream = follow_up_manager.chatbot.get_detailed_response(
                query=latest_query,
                chat_history=conversation_history,
                session_id=session_id,
                stream=True,
            )

            main_response = ""
            end_chat_triggered = False
            async for chunk in response_stream:
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

                if isinstance(chunk, dict) and chunk.get("status") == "end_chat":
                    end_chat_triggered = True

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

            # After saving assistant message, check if session should be closed and send email only on end_chat status.
            try:
                session = follow_up_manager.get_session_data(session_id)
                state = session.setdefault("state", {})
                # Only close session and trigger email if status=end_chat was yielded in the response
                if end_chat_triggered:
                    if state.get("is_active") is not False:
                        # Check for closure condition: user_details_known True
                        if state.get("user_details_known"):
                            reason = "user_details_known"
                            try:
                                def _bg_send():
                                    try:
                                        import asyncio
                                        from app.api.v1.helpers import end_session_helper
                                        asyncio.run(end_session_helper(session_id))
                                    except Exception:
                                        logger.exception("[chat_logic] background send_closure_email failed")
                                t = threading.Thread(target=_bg_send, daemon=True)
                                t.start()
                            except Exception as e:
                                logger.exception(f"[chat_logic] Failed to end session: {e}")
            except Exception as e:
                logger.exception(f"[chat_logic] Post-response closure check failed: {e}")

        except Exception as e:
            logger.exception("[ChatLogic] Unified response generation failed: %s", e)
            yield {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("[ChatLogic] Unexpected failure in build_chatbot_response: %s", e)
        yield {"status": "error", "message": str(e)}
        
