import json
import re
import tiktoken
from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.logger import get_logger
from app.utils.llm_utils import generate_llm_response
from app.utils.redis_context import get_redis_context_chunks

# format_response is now imported only when needed (async version with URL validation)
from app.utils.prompts import final_response_prompt, PROMPT_VERSION
from app.utils.response_formatter import format_response, _normalize_url
import asyncio
import functools
from app.api.models import MessageCreate

logger = get_logger("chatbot")


def _calculate_dynamic_top_n(query: str, conversation_history: list) -> int:
    """
    Dynamically calculate top_n based on query characteristics rather than static keywords.

    Uses query length, question type indicators, and conversation context to determine
    how many chunks to retrieve. Adapts to different query types automatically.

    Args:
        query: User's current query
        conversation_history: Previous conversation messages

    Returns:
        int: Dynamic top_n value (4-30 range)
    """
    if not query:
        return 4  # Default for empty queries

    query_lower = query.lower().strip()
    query_length = len(query_lower)
    query_words = len(query_lower.split())

    # Start with base value
    top_n = 4

    # Factor 1: Query length and complexity
    # Longer, more detailed queries may need more context
    if query_length > 50:
        top_n += 4
    if query_length > 100:
        top_n += 4
    if query_words > 10:
        top_n += 3

    # Factor 2: Broad exploratory questions (question words indicate broad queries)
    # Questions starting with "what", "tell me about", "show me" suggest broader searches
    broad_question_patterns = [
        query_lower.startswith("what"),
        query_lower.startswith("tell me"),
        query_lower.startswith("show me"),
        query_lower.startswith("explore"),
        query_lower.startswith("list"),
        "all" in query_lower,
        "everything" in query_lower,
        "comprehensive" in query_lower,
    ]

    if any(broad_question_patterns):
        top_n += 8  # Increase for broad exploratory queries

    # Factor 3: Comparison or multiple items queries
    if "compare" in query_lower or "difference" in query_lower or "vs" in query_lower:
        top_n += 6

    # Factor 4: Conversation context - analyze message patterns for breadth indicators
    if conversation_history:
        recent_user_msgs = [
            (
                msg.get("content", "")
                if isinstance(msg, dict)
                else (msg[1] if isinstance(msg, (list, tuple)) and len(msg) > 1 else "")
            )
            for msg in conversation_history[-3:]
            if (isinstance(msg, dict) and msg.get("role") == "user")
            or (isinstance(msg, (list, tuple)) and len(msg) > 0 and msg[0] == "user")
        ]

        # If recent messages are longer and have broad question patterns, increase top_n
        for msg in recent_user_msgs:
            if msg and len(msg) > 30:
                msg_lower = msg.lower()
                # Detect broad exploratory patterns (structural, not keyword-based)
                has_broad_pattern = (
                    msg_lower.startswith(("what", "tell", "show", "explore", "list"))
                    or "all" in msg_lower
                    or "everything" in msg_lower
                    or len(msg_lower.split())
                    > 8  # Longer messages often need more context
                )
                if has_broad_pattern:
                    top_n += 4
                    break

    # Factor 5: Question marks indicate information-seeking queries
    if "?" in query:
        top_n += 2

    # Cap the maximum (to avoid excessive token usage) but allow for comprehensive searches
    top_n = min(max(top_n, 4), 30)  # Range: 4-10

    logger.info(
        f"[DynamicTopN] Calculated top_n={top_n} for query length={query_length}, words={query_words}"
    )

    return top_n


class ContextOptimizer:
    """
    Handles context optimization to fit within model token limits
    while maximizing relevance and information retention.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.model_limits = {"gpt-4o": 128000}
        self.context_limit = self.model_limits.get(model, 4096)

    @lru_cache(maxsize=1000)
    def count_tokens_cached(self, text: str) -> int:
        return len(self.encoding.encode(text))


class OptimizedChatbot:
    def _is_too_similar(
        self, last_response: str, new_response: str, threshold: float = 0.9
    ) -> bool:
        """
        Returns True if the new response is too similar to the last one (using a simple ratio).
        """
        if not last_response or not new_response:
            return False
        import difflib

        ratio = difflib.SequenceMatcher(
            None, last_response.strip().lower(), new_response.strip().lower()
        ).ratio()
        return ratio >= threshold

    def __init__(self, llm=None, model: str = "gpt-4o"):
        """Initialize OptimizedChatbot.

        Parameters:
        - llm: optional external LLM client object to use (must implement .invoke or content access). If provided, it will be used as self.query_llm.
        - model: model name to instantiate a default ChatOpenAI if llm is not provided.
        """
        self.model = model
        # ContextOptimizer is unused in the critical path, skipping initialization to save overhead
        # self.context_optimizer = ContextOptimizer(model)
        # self.encoding = self.context_optimizer.encoding
        # self.model_limits = self.context_optimizer.model_limits
        # self.context_limit = self.context_optimizer.context_limit

        self.query_llm = llm if llm is not None else ChatOpenAI(model=model)

    async def get_detailed_response(
        self, query: str, chat_history, session_id: str, stream: bool = True
    ):
        from app.api.helpers import (
            get_user_details_known_from_db,
        )  # Do Not Move Outside Function

        """
        Generate a detailed response for a query and stream structured events.
        Yields dict events used by the API layer: 
        {status: 'chunk'|'form_trigger'|'meta', 'chunk': ...}
        """
        # Track last assistant response for anti-repetition
        last_assistant_response = None
        for msg in reversed(chat_history or []):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_assistant_response = msg.get("content")
                break
            elif (
                isinstance(msg, (list, tuple))
                and len(msg) >= 2
                and msg[0] == "assistant"
            ):
                last_assistant_response = msg[1]
                break
        try:
            # Step 1: Retrieve Redis context
            try:
                # Convert chat_history to proper format for Redis context retrieval
                conversation_history_for_redis = []
                for msg in chat_history or []:
                    if isinstance(msg, dict):
                        conversation_history_for_redis.append(msg)
                    elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                        conversation_history_for_redis.append(
                            {"role": msg[0], "content": msg[1]}
                        )
                # Dynamic top_n calculation based on query characteristics
                top_n_value = _calculate_dynamic_top_n(
                    query, conversation_history_for_redis
                )
                
                # Offload blocking Redis/Embedding call to executor
                import asyncio
                import functools
                loop = asyncio.get_event_loop()
                context_chunks = await loop.run_in_executor(
                    None,
                    functools.partial(
                        get_redis_context_chunks,
                        session_id,
                        query,
                        top_n=top_n_value,
                    ),
                )
            except Exception as e:
                logger.warning(
                    "Redis context retrieval failed, using empty context: %s", e
                )
                context_chunks = []

            context = "\n\n---\n\n".join(map(str, context_chunks or []))
            # Build a concise LLM-ready context from history (summary + latest user message)
            try:
                from app.utils.chat_state import build_llm_context_from_history

                history = build_llm_context_from_history(session_id, query) or self._format_history(chat_history)
            except Exception:
                history = self._format_history(chat_history)
            count = len(chat_history)
            logger.info(
                f"[Chatbot] Enhanced Conversation Summary generated ({len(history)} chars) for {count} messages"
            )
            logger.info(f"[COMPLETE_CONVERSATION_HISTORY] Session: {session_id}")
            logger.info(
                f"[COMPLETE_CONVERSATION_HISTORY] Raw chat_history being passed to LLM: {chat_history}"
            )
            logger.info(
                f"[COMPLETE_CONVERSATION_HISTORY] Formatted conversation summary for LLM:\n{history}"
            )
            logger.info(
                f"[MESSAGE_COUNT_DEBUG] Final message count for form trigger logic: {count}"
            )
            logger.info(
                f"[Chatbot] Redis Context Retrieved: {len(context)} chars, {len(context_chunks)} chunks"
            )
            logger.info(f"[Chatbot] Chat History: {count} messages")

            # Additional debug info for message count validation
            user_messages = [
                msg
                for msg in conversation_history_for_redis
                if msg.get("role") == "user"
            ]
            assistant_messages = [
                msg
                for msg in conversation_history_for_redis
                if msg.get("role") == "assistant"
            ]
            logger.info(
                f"[MESSAGE_COUNT_BREAKDOWN] Total: {count}, User: {len(user_messages)}, Assistant: {len(assistant_messages)}"
            )

            # 1. Yield processing chunk (handled by outer function, do not yield here)
            # yield {"status": "processing", "message": "Preparing response..."}

            # 2. Get user details once (Cache for this request)
            from app.api.helpers import get_user_details_from_db, get_user_details_known_from_db
            
            # Fetch both in parallel if possible, or just sequentially but ONCE
            user_details = get_user_details_from_db(session_id)
            user_details_known = user_details.get("user_details_known", False)
            
            # If get_user_details_from_db doesn't return the flag (it seems it does based on previous code), 
            # we might need to check how it was implemented. 
            # Looking at helpers.py, get_user_details_from_db returns a dict, and it includes "user_details_known" if true.
            # However, get_user_details_known_from_db is a separate call that returns a boolean.
            # Let's stick to the safe bet: use the specific function if the dict doesn't have it, 
            # OR just trust the specific function. 
            # Actually, let's just call both ONCE.
            
            if not user_details:
                 # Fallback if user_details is empty but we need the flag
                 user_details_known = get_user_details_known_from_db(session_id)
            
            logger.info(f"[Chatbot] Context: user_details_known={user_details_known}, details_found={bool(user_details)}")

            # Handle simple affirmative "yes" to a choice question
            if (
                query.strip().lower() == "yes"
                and last_assistant_response
                and " or " in last_assistant_response
            ):
                # If the user says "yes" to a choice, ask for clarification.
                clarification_response = (
                    "Great! Which of those options works best for you?"
                )
                yield {"status": "chunk", "chunk": clarification_response}
                # Save this clarification to history
                try:
                    # import asyncio
                    # from app.api.models import MessageCreate
                    from app.api.helpers import save_message

                    assistant_msg = MessageCreate(
                        session_id=session_id,
                        content=clarification_response,
                        role="assistant",
                        reply_to=None,
                        follow_up_to=None,
                        follow_up_depth=0,
                        metadata={},
                    )
                    asyncio.create_task(save_message(assistant_msg))
                    logger.info(
                        f"[Chatbot] Saved clarification response for 'yes' answer."
                    )
                except Exception as e:
                    logger.error(f"Failed to save clarification response: {e}")
                return

            # ============================================================
            # PROMPT SOURCE CLARITY: Using prompts.py, NOT Redis chunks
            # ============================================================
            logger.info(
                "[PROMPT_SOURCE] ✓ Loading prompt instructions from prompts.py via final_response_prompt() function"
            )
            logger.info(
                "[PROMPT_SOURCE] ✗ NOT using Redis chat_prompt_json chunks (Redis only used for knowledge base context)"
            )
            logger.info(
                f"[PROMPT_SOURCE] Prompt version: {PROMPT_VERSION} from app.utils.prompts"
            )
            
            prompt = final_response_prompt(
                prompt_context=context,
                conversation_summary=history or "",
                query=query,
                count=count,
                user_details_known=user_details_known,
                user_details=user_details,
            )
            logger.info(
                f"[Chatbot] Prompt prepared (user_details_known={user_details_known})"
            )

            # Step 3: Generate response from LLM
            try:
                # import asyncio

                if asyncio.iscoroutinefunction(generate_llm_response):
                    final_text = await generate_llm_response(prompt)
                else:
                    # import functools

                    loop = asyncio.get_event_loop()
                    final_text = await loop.run_in_executor(
                        None, functools.partial(generate_llm_response, prompt)
                    )
                safe_final_text = final_text if isinstance(final_text, str) else ""
            except Exception as llm_exc:
                logger.exception(f"[Chatbot] LLM call failed: {llm_exc}")
                async for fallback in self._fallback_response_stream(
                    "I couldn't generate a response right now."
                ):
                    yield fallback
                return
            # Fallback if LLM returns None or empty string (e.g., due to rate limit or internal error)
            if not safe_final_text or not safe_final_text.strip():
                logger.warning(
                    "[Chatbot] LLM returned empty or None, triggering fallback response."
                )
                async for fallback in self._fallback_response_stream(
                    "I couldn't generate a response right now."
                ):
                    yield fallback
                return

            # Step 4: Parse LLM JSON output (if present)
            funnel_stage = ""
            response_text = ""

            def extract_json_from_markdown(txt: str) -> str:
                # First look for a fenced JSON block
                pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
                match = re.search(pattern, txt, re.DOTALL)
                if match:
                    return match.group(1).strip()

                # If no fence found, try to find the first balanced JSON object by brace matching
                start = None
                depth = 0
                for i, ch in enumerate(txt):
                    if ch == "{":
                        if start is None:
                            start = i
                        depth += 1
                    elif ch == "}" and start is not None:
                        depth -= 1
                        if depth == 0:
                            return txt[start : i + 1].strip()

                # Fall back to returning the original stripped text
                return txt.strip()

            try:
                cleaned_json = extract_json_from_markdown(safe_final_text)
                # First attempt: JSON inside a code fence or a balanced JSON object in the text
                llm_json = json.loads(cleaned_json)
                response_text = llm_json.get("response", "") or ""
                funnel_stage = (llm_json.get("funnel_stage", "") or "").lower()
                # Ignore LLM's user_details_known, always use DB value for meta chunk
                user_network_id = llm_json.get("user_network_id") or None
                logger.info(f"[Chatbot] Parsed JSON: {llm_json}")
            except Exception as json_exc:
                logger.warning(
                    f"[Chatbot] Failed to parse LLM output as JSON: {json_exc}"
                )
                # As a best-effort, try to extract a user-facing 'response' field
                # using a simple regex fallback before giving up.
                resp_match = re.search(
                    r'"response"\s*:\s*"([\s\S]*?)"\s*(,|})', safe_final_text
                )
                if resp_match:
                    response_text = resp_match.group(1).strip()
                    logger.info(
                        "[Chatbot] Recovered 'response' value from LLM text via regex fallback."
                    )
                else:
                    response_text = safe_final_text
                user_network_id = None

            logger.info(
                f"[Chatbot] Final output len={len(safe_final_text)}, funnel_stage='{funnel_stage}'"
            )
            if "llm_json" in locals():
                logger.info(
                    f"[LLM_DEBUG] Raw LLM JSON response: {llm_json}"
                )  # Debug what LLM actually returns

            # Step 5: Continue with regular response flow

            # Step 6: Stream formatted response
            cleaned = (response_text or "").strip()

            # Debug markdown preservation
            has_markdown_before = any(
                marker in cleaned for marker in ["**", "*", "_", "#", "-", "`", "```"]
            )
            if has_markdown_before:
                logger.info(
                    f"[MARKDOWN_DEBUG] Raw response contains markdown: {cleaned[:100]}..."
                )

            # Step 6: DYNAMIC Funnel-based form trigger (LLM-driven intelligence)
            # Use cached user_details_known
            logger.info(
                f"[FormLogic] Stage='{funnel_stage}', Count={count}, Known={user_details_known}"
            )
            trigger_form = False
            trigger_reason = ""
            if not user_details_known:
                # DYNAMIC LOGIC: Let LLM determine funnel stage based on conversation analysis
                # Only enforce minimum safety (count >= 2) and fallback (count >= 10)
                
                if funnel_stage == "action":
                    # LLM detected Action stage - user wants to connect or shows strong buying intent
                    if count >= 2:
                        # Safety: minimum 2 user messages before any form
                        trigger_form = True
                        trigger_reason = "action_stage_llm_detected"
                        logger.info(
                            f"[FORM_DEBUG] Action stage detected by LLM: funnel_stage='{funnel_stage}', count={count}"
                        )
                    else:
                        # Too early - need at least 2 user messages
                        logger.info(
                            f"[FORM_DEBUG] Action stage detected but count={count} < 2 - waiting for minimum messages"
                        )
                elif funnel_stage == "intent":
                    # LLM detected Intent stage - user shows buying signals
                    if count >= 2:
                        # Trigger form if LLM detected intent and we have minimum messages
                        trigger_form = True
                        trigger_reason = "intent_stage_llm_detected"
                        logger.info(
                            f"[FORM_DEBUG] Intent stage detected by LLM: funnel_stage='{funnel_stage}', count={count}"
                        )
                    else:
                        logger.info(
                            f"[FORM_DEBUG] Intent stage detected but count={count} < 2 - waiting for minimum messages"
                        )
                elif funnel_stage == "interest":
                    # LLM detected Interest stage - user is engaged
                    if count >= 3:
                        # For interest stage, wait for at least 3 messages to ensure engagement
                        trigger_form = True
                        trigger_reason = "interest_stage_llm_detected"
                        logger.info(
                            f"[FORM_DEBUG] Interest stage detected by LLM with sufficient engagement: funnel_stage='{funnel_stage}', count={count}"
                        )
                    else:
                        logger.info(
                            f"[FORM_DEBUG] Interest stage detected but count={count} < 3 - continue building rapport"
                        )
                elif count >= 10:
                    # Fallback: If conversation is extended without form trigger, trigger anyway
                    trigger_form = True
                    trigger_reason = "conversation_length_fallback"
                    logger.info(
                        f"[FORM_DEBUG] Fallback trigger: Extended conversation without form (count={count})"
                    )
                else:
                    logger.info(
                        f"[FORM_DEBUG] No trigger conditions met. funnel_stage='{funnel_stage}', count={count} - LLM will determine next stage"
                    )

            # Only yield the assistant response if we are NOT about to trigger a form
            if not trigger_form:
                # Use smart formatter - format immediately, validate URLs in background
                # from app.utils.response_formatter import format_response

                # Format response immediately (no blocking URL validation)
                formatted = format_response(cleaned, query, None)
                # Note: format_response already handles \\n replacement internally

                # Quick URL normalization (no network calls) - fix spaces in URLs
                # Skip blocking URL validation - normalize only (fixes spaces like "real- estate")
                # from app.utils.response_formatter import _normalize_url

                formatted = self._normalize_urls_in_text(formatted)

                # Sanitize horizontal rules that cause Setext H2 rendering
                # Remove lines containing only dashes, asterisks, or underscores (3+)
                # These create horizontal rules in markdown and can cause text above to render as headers
                formatted = re.sub(
                    r'^\s*[-*_]{3,}\s*$',  # Match lines with 3+ dashes, asterisks, or underscores
                    '',
                    formatted,
                    flags=re.MULTILINE
                )
                # Clean up any resulting multiple blank lines
                formatted = re.sub(r'\n{3,}', '\n\n', formatted)


                # Anti-repetition: compare with last assistant response
                if last_assistant_response and isinstance(last_assistant_response, str):
                    if self._is_too_similar(last_assistant_response, formatted):
                        logger.warning(
                            "[ANTI-REPETITION] New response is too similar to last assistant response. Consider regenerating or expanding."
                        )
                        # Only add clarifying question if user_details_known=False
                        if not user_details_known:
                            formatted += "\n\n(Can you share more details or specify what you'd like to know next?)"

                if has_markdown_before:
                    has_markdown_after = any(
                        marker in formatted
                        for marker in ["**", "*", "_", "#", "-", "`", "```"]
                    )
                    logger.info(
                        f"[MARKDOWN_DEBUG] Formatted response markdown preserved: {has_markdown_after}"
                    )
                    if not has_markdown_after:
                        logger.warning(
                            "[MARKDOWN_WARNING] Markdown lost during formatting!"
                        )
                    
                    # Additional debug for bold markdown specifically
                    bold_before = "**" in cleaned
                    bold_after = "**" in formatted
                    if bold_before and not bold_after:
                        logger.warning(
                            f"[MARKDOWN_WARNING] Bold markdown (**) was removed! known={user_details_known}"
                        )
                    elif bold_before and bold_after:
                        # Check for spacing issues in bold markdown
                        if "** " in formatted or " **" in formatted:
                            logger.warning(
                                f"[MARKDOWN_WARNING] Bold markdown spacing issue detected: '** ' or ' **' found in response"
                            )

                # Only yield a chunk if it is non-empty and not just whitespace
                if formatted and formatted.strip():
                    yield {"status": "chunk", "chunk": formatted}
                    if count >= 50:
                        logger.info(
                            f"[Chatbot] [DEBUG] Yielding end_chat chunk: user_message_count={count}"
                        )
                        print(
                            f"[DEBUG] Yielding end_chat chunk: user_message_count={count}"
                        )
                        yield {
                            "status": "end_chat",
                            "chunk": "Our sales team will reach out within 1 business day, Thank you for your interest in Ditstek innovations.",
                        }

            # Step 7: Emit meta update if present, but skip if form_trigger is about to be yielded
            # Use cached user_details_known
            meta_chunk = {
                "user_details_known": user_details_known,
                **({"user_network_id": user_network_id} if user_network_id else {}),
            }
            if not trigger_form and meta_chunk:
                yield {"status": "meta", "chunk": meta_chunk}

            # Step 8: Form trigger event (if needed)
            if trigger_form:
                logger.info(
                    f"[Chatbot] FORM TRIGGER ACTIVATED! Reason: {trigger_reason}, funnel_stage={funnel_stage}, message_count={count}"
                )
                yield {"status": "form_trigger", "chunk": ""}
            else:
                logger.info(
                    f"[Chatbot] No form trigger. Stage={funnel_stage}"
                )

        except Exception as e:
            logger.exception("[Chatbot] Redis-based response generation failed")
            async for fallback in self._fallback_response_stream(
                "I couldn't generate a response right now."
            ):
                yield fallback

    async def _fallback_response_stream(self, query: str):
        """Fallback response when enhanced flow fails"""
        # If the query is the default fallback string, show a simple error message
        if query.strip() == "I couldn't generate a response right now.":
            fallback_text = "Sorry, I couldn't generate a response right now. Please try again later."
        elif query.strip():
            fallback_text = f"Sorry, I couldn't answer your request: {query}"
        else:
            fallback_text = "Sorry, I couldn't generate a response."
        yield {"status": "chunk", "chunk": fallback_text}

    def _format_history(self, chat_history: list) -> str:
        """
        Format chat history for LLM with enhanced conversation flow context.
        Includes timestamps when available and structures the conversation for better understanding.
        Handles both dict and tuple formats.
        """
        if not chat_history:
            return ""

        formatted = []
        conversation_count = 0

        # Normalize messages to dict format for consistent processing
        normalized_msgs = []
        for msg in chat_history:
            if isinstance(msg, dict):
                normalized_msgs.append(msg)
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                normalized_msgs.append(
                    {
                        "role": msg[0],
                        "content": msg[1],
                        "timestamp": msg[2] if len(msg) > 2 else "",
                    }
                )

        # Count actual conversation pairs for context
        user_msgs = [
            msg for msg in normalized_msgs if msg.get("role", "").lower() == "user"
        ]
        assistant_msgs = [
            msg for msg in normalized_msgs if msg.get("role", "").lower() == "assistant"
        ]
        conversation_count = min(len(user_msgs), len(assistant_msgs))

        # Add conversation flow summary at the top
        if conversation_count > 0:
            formatted.append(
                f"=== CONVERSATION FLOW ({conversation_count} exchanges, {len(normalized_msgs)} total messages) ==="
            )

        # Format each message with enhanced context
        for i, msg in enumerate(normalized_msgs):
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # Add timestamp if available
            time_info = f" [{timestamp}]" if timestamp else ""

            if role == "user":
                formatted.append(f"User{time_info}: {content}")
            elif role == "assistant":
                # Don't manipulate assistant content to preserve markdown formatting
                # Simply append as-is to avoid breaking markdown
                formatted.append(f"Assistant{time_info}: {content}")
            else:
                # In case of unknown role, include as-is for debugging
                formatted.append(f"{role.capitalize()}{time_info}: {content}")

        # Add conversation context summary at the end
        if conversation_count > 0:
            formatted.append(f"=== END CONVERSATION FLOW ===")

        return "\n".join(formatted)

    def _normalize_urls_in_text(self, text: str) -> str:
        """Helper to normalize URLs in text without blocking validation."""
        url_pattern = r"\[([^\]]+)\]\(([^)]+)\)|(https?://[^\s\)]+)"

        def normalize_match(match):
            if match.group(2):  # Markdown link
                normalized = _normalize_url(match.group(2))
                return f"[{match.group(1)}]({normalized})"
            elif match.group(3):  # Plain URL
                return _normalize_url(match.group(3))
            return match.group(0)

        return re.sub(url_pattern, normalize_match, text)
