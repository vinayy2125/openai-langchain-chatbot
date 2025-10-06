import json
import re
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.core.prompts import enhanced_query_prompt, enhanced_query_prompt_no_context
from app.core.redis_context import get_redis_context_chunks

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Utility helpers (restored minimal versions for optimizer imports)
def _maybe_expand_queries(query: str) -> list:
    """Return lightweight expanded query variants (deduplicated)."""
    variants = [query, f"Details about {query}", f"In-depth explanation of {query}"]
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
        text = getattr(d, "page_content", str(d)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
        result.append((text, meta))
    return result


async def build_chatbot_response(
    session_id: str,
    follow_up_manager,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    prompt_context: Optional[str] = None,
    mode: str = "complete",
) -> AsyncGenerator[Any, None]:
    """
    Build a streaming response from the chatbot that handles both direct responses and follow-up suggestions.

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

        # Check if requirements are complete and decide response strategy
        requirements_complete = follow_up_manager.check_requirements(session_id)

        if requirements_complete:
            logger.info(
                f"Requirements complete for session {session_id}, generating comprehensive response"
            )

            # Call and normalize the comprehensive response safely.
            try:
                raw_comp = follow_up_manager.generate_comprehensive_response(
                    session_id
                )
            except Exception as exc:
                logger.exception(
                    "[ChatLogic] generate_comprehensive_response raised; using fallback"
                )
                raw_comp = None

            # Normalize many possible return shapes into a safe string
            if raw_comp is None:
                comprehensive_response = ""
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

            # Guarantee at least one user-visible chunk so SSE doesn't stall
            if not comprehensive_response:
                comprehensive_response = (
                    "I'm sorry —. "
                    "Could you provide more details or rephrase your question?"
                )
                logger.warning(
                    "[ChatLogic] Using fallback comprehensive_response for session %s", session_id
                )

            # Format and stream the response in a concise, structured way
            # Process the response to ensure proper markdown formatting and structure
            lines = comprehensive_response.split("\n")
            current_section = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Convert headers to consistent format
                if line.startswith("#"):
                    # If we have accumulated content, send it
                    if current_section:
                        yield {
                            "status": "chunk",
                            "chunk": "\n".join(current_section),
                        }
                        current_section = []

                    # Format header and send
                    header = re.sub(r"^#{1,6}\s*", "###### ", line)
                    yield {"status": "chunk", "chunk": "\n\n" + header + "\n"}
                else:
                    # Process regular content
                    # Ensure proper bold formatting
                    line = re.sub(r"\*\*([^*]+)\*\*", r"**\1**", line)
                    # Ensure bullet points are properly formatted
                    if line.lstrip().startswith("- "):
                        line = line.strip()
                    current_section.append(line)

            # Send any remaining content
            if current_section:
                yield {"status": "complete_chunk", "chunk": "\n".join(current_section)}

            # Add spacing and final suggestions
            yield {
                "status": "separator",
                "chunk": "\n\n",
            }  # Extra spacing after main content

            # Add clear separation before suggestions and follow-ups
            yield {"status": "separator", "chunk": "\n\n"}

            try:
                # Generate both suggestions and follow-ups
                suggestions = follow_up_manager.generate_suggestions(
                    session_id, context="comprehensive response completed"
                )
                follow_ups = follow_up_manager.generate_follow_ups(
                    session_id, latest_query, context="comprehensive response"
                )

                # Format suggestions with bullet points and bold highlights
                if suggestions:
                    formatted_suggestions = []
                    for s in suggestions[:2]:  # Limit to 2 suggestions
                        # Ensure suggestion starts with bullet point
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

                # Add spacing between sections
                yield {"status": "separator", "chunk": "\n\n"}

                # Format follow-ups with clear structure
                if follow_ups and len(follow_ups) > 0:
                    formatted_followups = []
                    for f in follow_ups[:2]:  # Limit to 2 follow-ups
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
                # Fallback with generic grouped suggestions and follow-ups
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
            # Generate concise response + specific follow-ups to gather more requirements
            logger.info(
                f"Requirements incomplete for session {session_id}, generating concise response with follow-ups"
            )

            # FIX: Smart context detection instead of hardcoded patterns
            assistant_messages = [
                msg
                for msg in (conversation_history or [])
                if msg.get("role") == "assistant"
            ]
            is_prompt_selection = (
                prompt_context is not None and len(assistant_messages) == 0
            )
            is_manual_query = prompt_context is None and len(assistant_messages) == 0
            is_followup_response = len(assistant_messages) > 0

            # FIX: Unified prompt instruction with smart context awareness
            # Always use Redis for context retrieval
            context_chunks = get_redis_context_chunks(
                session_id=session_id,
                query=latest_query,
                conversation_history=conversation_history or [],
                top_n=6
            )
            context_text = "\n".join(context_chunks)
            if is_prompt_selection or is_manual_query:
                enhanced_query = enhanced_query_prompt(context_text=context_text, latest_query=latest_query)
            else:
                enhanced_query = enhanced_query_prompt_no_context(context_text=context_text, latest_query=latest_query, conversation_history=conversation_history)
                # Log the enhanced query (short preview) for debugging
            logger.info(f"[ChatLogic] Enhanced query prepared (first 300 chars): {str(enhanced_query)[:300]}")
            # Generate and format the main response
            main_response = ""
            # Call the chatbot with the raw latest_query and session_id so it
            # performs Redis context retrieval and prompt construction itself.
            response_stream = follow_up_manager.chatbot.get_detailed_response(
                query=latest_query,
                chat_history=[(msg["role"], msg["content"]) for msg in (conversation_history or [])],
                session_id=session_id,
                stream=True,
            )

            # Process and format the response
            current_section = []
            saw_structured = False
            for chunk in response_stream:
                if not chunk:
                    continue

                # If upstream already emits structured dict events, prefer those
                # and avoid reprocessing plain strings which can cause duplicates.
                if isinstance(chunk, dict):
                    saw_structured = True
                    if "chunk" in chunk:
                        yield chunk
                        continue
                    # If dict lacks a 'chunk' field, stringify it as a fallback
                    text_chunk = str(chunk)
                else:
                    # If we've already observed structured dict events from the
                    # upstream generator, skip raw string chunks to avoid duplicate
                    # emission paths (optimizer emits dicts). Otherwise process.
                    if saw_structured:
                        continue
                    text_chunk = str(chunk)

                if not text_chunk:
                    continue

                main_response += text_chunk

                # Split into lines to process sections while preserving leading markers
                lines = text_chunk.split("\n")
                for line in lines:
                    # Trim only trailing whitespace to preserve leading '-' or '#'
                    line = line.rstrip()
                    if not line.strip():
                        if current_section:
                            # Join and send accumulated section preserving newlines
                            section_text = "\n".join(current_section)
                            yield {"status": "chunk", "chunk": section_text}
                            current_section = []
                        continue

                    # Handle headers (allow leading spaces before '#')
                    if line.lstrip().startswith("#"):
                        if current_section:
                            section_text = "\n".join(current_section)
                            yield {"status": "chunk", "chunk": section_text}
                            current_section = []

                        # Format header and send (preserve a blank line before)
                        header = re.sub(r"^#{1,6}\s*", "###### ", line.lstrip())
                        yield {"status": "chunk", "chunk": "\n\n" + header + "\n"}
                    else:
                        # Process regular content
                        # Bullet points: send them as their own chunk (preserve leading '-')
                        if line.lstrip().startswith("- "):
                            if current_section:
                                section_text = "\n".join(current_section)
                                yield {"status": "chunk", "chunk": section_text}
                                current_section = []
                            # Send bullet points as chunk events (preserve markdown)
                            yield {"status": "chunk", "chunk": line.lstrip()}
                        else:
                            current_section.append(line)

            # Send any remaining content (preserve newlines)
            if current_section:
                section_text = "\n".join(current_section)
                yield {"status": "complete_chunk", "chunk": section_text}

            # Add spacing after main response (no separator lines)
            yield {"status": "separator", "chunk": "\n\n"}  # Just spacing, no lines

            # Save the main response to the conversation history
            if main_response:
                follow_up_manager.add_to_conversation_history(
                    session_id, "assistant", main_response
                )

            # Add spacing after main response
            yield {"status": "separator", "chunk": "\n\n"}

            try:
                # Generate single suggestion and follow-up
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
        
