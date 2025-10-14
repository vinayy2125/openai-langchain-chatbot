import json
import re
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.core.prompts import enhanced_query_prompt
from app.core.redis_context import get_redis_context_chunks
from app.core.fallback_handler import get_smart_fallback
from app.core.response_formatter import format_response

# Initialize logger
logger = logging.getLogger(__name__)


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

        requirements_complete = follow_up_manager.check_requirements(session_id)

        if requirements_complete:
            logger.info(
                f"Requirements complete for session {session_id}, generating comprehensive response"
            )

            try:
                raw_comp = follow_up_manager.generate_comprehensive_response(
                    session_id
                )
            except Exception as exc:
                logger.exception(
                    "[ChatLogic] generate_comprehensive_response raised; using fallback"
                )
                raw_comp = None

            if raw_comp is None:
                context_chunks = get_redis_context_chunks(
                    session_id, 
                    latest_query, 
                    conversation_history or []
                )
                comprehensive_response = get_smart_fallback(
                    latest_query,
                    context_chunks=context_chunks
                )
            elif isinstance(raw_comp, str):
                comprehensive_response = raw_comp
            elif isinstance(raw_comp, dict):
                # common shapes: {"content": "..."} or {"chunk": "..."}
                comprehensive_response = (
                    raw_comp.get("content")
                    or raw_comp.get("chunk")
                    or raw_comp.get("text")
                    or json.dumps(raw_comp)
                )
            else:
                # object-like (AIMessage etc.) — prefer .content, .text, fallback to str()
                comprehensive_response = (
                    getattr(raw_comp, "content", None)
                    or getattr(raw_comp, "text", None)
                    or str(raw_comp)
                )

            comprehensive_response = (comprehensive_response or "").strip()
            logger.info(
                f"[ChatLogic] Normalized comprehensive_response len={len(comprehensive_response)} preview={comprehensive_response[:300]!r}"
            )

            if not comprehensive_response:
                comprehensive_response = (
                    "I'm sorry —. "
                    "Could you provide more details or rephrase your question?"
                )
                logger.warning(
                    "[ChatLogic] Using fallback comprehensive_response for session %s", session_id
                )

            lines = comprehensive_response.split("\n")
            current_section = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    if current_section:
                        yield {
                            "status": "chunk",
                            "chunk": "\n".join(current_section),
                        }
                        current_section = []

                    header = re.sub(r"^#{1,6}\s*", "###### ", line)
                    yield {"status": "chunk", "chunk": "\n\n" + header + "\n"}
                else:
                    line = re.sub(r"\*\*([^*]+)\*\*", r"**\1**", line)
                    if line.lstrip().startswith("- "):
                        line = line.strip()
                    current_section.append(line)

            if current_section:
                
                formatted_content = format_response(
                    "\n".join(current_section),
                    latest_query,
                    conversation_history
                )
                yield {"status": "complete_chunk", "chunk": formatted_content}

            yield {
                "status": "separator",
                "chunk": "\n\n",
            }  

            yield {"status": "separator", "chunk": "\n\n"}

            try:
                suggestions = follow_up_manager.generate_suggestions(
                    session_id, context="comprehensive response completed"
                )
                follow_ups = follow_up_manager.generate_follow_ups(
                    session_id, latest_query, context="comprehensive response"
                )

                if suggestions:
                    formatted_suggestions = []
                    for s in suggestions[:2]:  
                        s = s.strip()
                        if not s.startswith("- "):
                            s = "- " + s
                        # Ensure key terms are bold
                        if "**" not in s:
                            words = s.split()
                            for i, word in enumerate(words):
                                if word.lower() in [
                                    "implement",
                                    "create",
                                    "use",
                                    "develop",
                                    "build",
                                    "integrate",
                                ]:
                                    words[i] = f"**{word}**"
                            s = " ".join(words)
                        formatted_suggestions.append(s)
                    yield {
                        "status": "suggestions",
                        "chunk": "\n".join(formatted_suggestions),
                    }
                else:
                    yield {
                        "status": "suggestions",
                        "chunk": "- Consider implementing a **proof of concept** to validate the approach",
                    }

                yield {"status": "separator", "chunk": "\n\n"}

                if follow_ups and len(follow_ups) > 0:
                    formatted_followups = []
                    for f in follow_ups[:2]:  
                        f = f.strip()
                        if not f.startswith("- "):
                            f = "- " + f
                        formatted_followups.append(f)
                    yield {
                        "status": "followups",
                        "chunk": "\n".join(formatted_followups),
                    }
                else:
                    yield {
                        "status": "followups",
                        "chunk": "- What specific aspects would you like to explore further?",
                    }

            except Exception as e:
                logger.error(f"Error generating suggestions and follow-ups: {e}")
                yield {
                    "status": "suggestions",
                    "chunk": "Consider implementing a proof of concept to validate the approach",
                }
                yield {"status": "separator", "chunk": "\n\n"}
                yield {
                    "status": "followups",
                    "chunk": "Would you like to explore any specific aspect of the solution?",
                }

        else:
            logger.info(
                f"Requirements incomplete for session {session_id}, generating concise response with follow-ups"
            )

            assistant_messages = [
                msg
                for msg in (conversation_history or [])
                if msg.get("role") == "assistant"
            ]
            is_prompt_selection = (
                prompt_context is not None and len(assistant_messages) == 0
            )
            is_manual_query = prompt_context is None and len(assistant_messages) == 0

            context_chunks = get_redis_context_chunks(
                session_id=session_id,
                query=latest_query,
                conversation_history=conversation_history or [],
                top_n=4,
            )
            try:
                preview_items = [c[:300] for c in context_chunks[:4]]
                logger.debug("[ChatLogic] Retrieved context chunks preview: %s", preview_items)
            except Exception:
                logger.debug("[ChatLogic] Retrieved context chunks but failed to produce preview")
            context_text = "\n".join(context_chunks)
            if is_prompt_selection or is_manual_query:
                enhanced_query = enhanced_query_prompt(context_text=context_text, latest_query=latest_query, conversation_history=conversation_history or [])
            logger.info(f"[ChatLogic] Enhanced query prepared (first 300 chars): {str(enhanced_query)[:300]}")
            main_response = ""
            response_stream = follow_up_manager.chatbot.get_detailed_response(
                query=latest_query,
                chat_history=[(msg["role"], msg["content"]) for msg in (conversation_history or [])],
                session_id=session_id,
                stream=True,
            )

            current_section = []
            saw_structured = False
            for chunk in response_stream:
                if not chunk:
                    continue

                if isinstance(chunk, dict):
                    saw_structured = True
                    if "chunk" in chunk:
                        yield chunk
                        continue
                    text_chunk = str(chunk)
                else:
                    if saw_structured:
                        continue
                    text_chunk = str(chunk)

                if not text_chunk:
                    continue

                main_response += text_chunk

                lines = text_chunk.split("\n")
                for line in lines:
                    line = line.rstrip()
                    if not line.strip():
                        if current_section:
                            section_text = "\n".join(current_section)
                            yield {"status": "chunk", "chunk": section_text}
                            current_section = []
                        continue

                    if line.lstrip().startswith("#"):
                        if current_section:
                            section_text = "\n".join(current_section)
                            yield {"status": "chunk", "chunk": section_text}
                            current_section = []

                        header = re.sub(r"^#{1,6}\s*", "###### ", line.lstrip())
                        yield {"status": "chunk", "chunk": "\n\n" + header + "\n"}
                    else:
                        if line.lstrip().startswith("- "):
                            if current_section:
                                section_text = "\n".join(current_section)
                                yield {"status": "chunk", "chunk": section_text}
                                current_section = []
                            yield {"status": "chunk", "chunk": line.lstrip()}
                        else:
                            current_section.append(line)

            if current_section:
                section_text = "\n".join(current_section)
                yield {"status": "complete_chunk", "chunk": section_text}

            yield {"status": "separator", "chunk": "\n\n"}  # Just spacing, no lines

            if main_response:
                follow_up_manager.add_to_conversation_history(
                    session_id, "assistant", main_response
                )

            yield {"status": "separator", "chunk": "\n\n"}

            try:
                suggestion = (
                    follow_up_manager.generate_suggestions(
                        session_id, context=main_response[:200]
                    )[0]
                    if follow_up_manager.generate_suggestions(
                        session_id, context=main_response[:200]
                    )
                    else None
                )
                follow_up = (
                    follow_up_manager.generate_follow_ups(
                        session_id, latest_query, context=main_response[:200]
                    )[0]
                    if follow_up_manager.generate_follow_ups(
                        session_id, latest_query, context=main_response[:200]
                    )
                    else None
                )

                # Format suggestion
                if suggestion:
                    # Clean up and format suggestion
                    clean_suggestion = re.sub(r"#{1,6}\s*", "", suggestion.strip())
                    clean_suggestion = re.sub(r"^\d+\.\s*|-\s*", "", clean_suggestion)
                    # Ensure proper bold formatting for key terms
                    for term in [
                        "implement",
                        "create",
                        "use",
                        "integrate",
                        "develop",
                        "leverage",
                    ]:
                        pattern = f"(?i)\\b{term}\\b"
                        clean_suggestion = re.sub(
                            pattern, f"**{term}**", clean_suggestion
                        )
                    yield {
                        "status": "suggestions",
                        "chunk": f"\n- {clean_suggestion.strip()}\n",
                    }
                else:
                    yield {
                        "status": "suggestions",
                        "chunk": "- Consider **implementing** a proof of concept to validate your approach",
                    }

                # Single separator
                yield {"status": "separator", "chunk": "\n\n"}

                # Format follow-up
                # Format and send a single follow-up
                follow_up = (
                    follow_up_manager.generate_follow_ups(
                        session_id, latest_query, context=main_response[:150]
                    )[0]
                    if follow_up_manager.generate_follow_ups(
                        session_id, latest_query, context=main_response[:150]
                    )
                    else None
                )

                if follow_up:
                    # Clean up and format follow-up
                    clean_followup = re.sub(r"#{1,6}\s*", "", follow_up.strip())
                    clean_followup = re.sub(r"^\d+\.\s*|-\s*", "", clean_followup)

                    # Make it more engaging if it's not already a question
                    if not any(
                        clean_followup.lower().startswith(q)
                        for q in ["what", "how", "could", "would", "can", "which"]
                    ):
                        clean_followup = (
                            f"Could you tell us more about {clean_followup.lower()}"
                        )

                    yield {
                        "status": "followup",
                        "chunk": f"\n- {clean_followup.strip()}?\n",
                    }
                else:
                    # Context-aware fallback follow-up
                    context_keywords = latest_query.lower() if latest_query else ""
                    if any(
                        word in context_keywords
                        for word in ["app", "mobile", "web", "application"]
                    ):
                        yield {
                            "status": "followup",
                            "chunk": "\n- What specific features or functionalities are most important for your app?\n",
                        }
                    elif any(
                        word in context_keywords
                        for word in ["ai", "machine learning", "ml", "chatbot"]
                    ):
                        yield {
                            "status": "followup",
                            "chunk": "\n- What type of AI functionality do you envision for your project?\n",
                        }
                    elif any(
                        word in context_keywords
                        for word in ["website", "site", "web", "portal"]
                    ):
                        yield {
                            "status": "followup",
                            "chunk": "\n- What's the main purpose of your website - e-commerce, corporate, or service-based?\n",
                        }
                    else:
                        yield {
                            "status": "followup",
                            "chunk": "\n- What specific aspects would you like to explore further?\n",
                        }
            except Exception as e:
                logger.error(f"Error generating follow-ups: {e}")
                # Fallback with generic suggestion and follow-up
                yield {
                    "status": "suggestion",
                    "chunk": "\n- Consider **implementing** a basic prototype to test core functionality\n",
                }
                yield {"status": "separator", "chunk": "\n\n"}
                yield {
                    "status": "followup",
                    "chunk": "\n- What specific features are most important to you?\n",
                }

    except Exception as e:
        yield {"status": "error", "message": str(e)}
        
