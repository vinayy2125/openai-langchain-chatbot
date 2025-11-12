import json
import re
import tiktoken
from typing import List, Generator
from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.logger import get_logger
from app.core.utils import generate_llm_response
from app.core.response_formatter import format_response
from app.core.redis_context import get_redis_context_chunks
from app.core.prompts import key_generate_prompt, final_response_prompt

logger = get_logger("chatbot")

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
    def _is_too_similar(self, last_response: str, new_response: str, threshold: float = 0.9) -> bool:
        """
        Returns True if the new response is too similar to the last one (using a simple ratio).
        """
        if not last_response or not new_response:
            return False
        import difflib
        ratio = difflib.SequenceMatcher(None, last_response.strip().lower(), new_response.strip().lower()).ratio()
        return ratio >= threshold
    
    
    def __init__(self, llm=None, model: str = "gpt-4o"):
        """Initialize OptimizedChatbot.

        Parameters:
        - llm: optional external LLM client object to use (must implement .invoke or content access). If provided, it will be used as self.query_llm.
        - model: model name to instantiate a default ChatOpenAI if llm is not provided.
        """
        self.model = model
        # Initialize a ContextOptimizer to centralize encoding and limits
        self.context_optimizer = ContextOptimizer(model)
        self.encoding = self.context_optimizer.encoding
        self.model_limits = self.context_optimizer.model_limits
        self.context_limit = self.context_optimizer.context_limit

        self.query_llm = llm if llm is not None else ChatOpenAI(model=model)


    async def get_detailed_response(self, query: str, chat_history, session_id: str, stream: bool = True):
        from app.api.v1.helpers import get_user_details_known_from_db  # Do Not Move Outside Function
        """
        Generate a detailed response for a query and stream structured events.
        Yields dict events used by the API layer: 
        {status: 'chunk'|'form_trigger'|'meta', 'chunk': ...}
        """
        # Track last assistant response for anti-repetition
        last_assistant_response = None
        for msg in reversed(chat_history or []):
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                last_assistant_response = msg.get('content')
                break
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == 'assistant':
                last_assistant_response = msg[1]
                break
        try:
            # Step 1: Retrieve Redis context
            try:
                # Convert chat_history to proper format for Redis context retrieval
                conversation_history_for_redis = []
                for msg in (chat_history or []):
                    if isinstance(msg, dict):
                        conversation_history_for_redis.append(msg)
                    elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                        conversation_history_for_redis.append({
                            "role": msg[0],
                            "content": msg[1]
                        })
                context_chunks = get_redis_context_chunks(session_id, query, conversation_history_for_redis, top_n=4)
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            context = "\n\n---\n\n".join(map(str, context_chunks or []))
            history = self._format_history(chat_history)
            count = len(chat_history)
            logger.info(f"[Chatbot] Enhanced Conversation Summary generated ({len(history)} chars) for {count} messages")
            logger.info(f"[COMPLETE_CONVERSATION_HISTORY] Session: {session_id}")
            logger.info(f"[COMPLETE_CONVERSATION_HISTORY] Raw chat_history being passed to LLM: {chat_history}")
            logger.info(f"[COMPLETE_CONVERSATION_HISTORY] Formatted conversation summary for LLM:\n{history}")
            logger.info(f"[MESSAGE_COUNT_DEBUG] Final message count for form trigger logic: {count}")
            logger.info(f"[Chatbot] Redis Context Retrieved: {len(context)} chars, {len(context_chunks)} chunks")
            logger.info(f"[Chatbot] Chat History: {count} messages")
            
            # Additional debug info for message count validation
            user_messages = [msg for msg in conversation_history_for_redis if msg.get('role') == 'user']
            assistant_messages = [msg for msg in conversation_history_for_redis if msg.get('role') == 'assistant'] 
            logger.info(f"[MESSAGE_COUNT_BREAKDOWN] Total: {count}, User: {len(user_messages)}, Assistant: {len(assistant_messages)}")

            # 1. Yield processing chunk (handled by outer function, do not yield here)
            # yield {"status": "processing", "message": "Preparing response..."}

            # 2. Generate and yield response chunk
            user_details_known = get_user_details_known_from_db(session_id)
            logger.info(f"[Chatbot] user_details_known (DB) = {user_details_known} for session {session_id}")

            prompt = final_response_prompt(
                prompt_context=context,
                conversation_summary=history or "",
                query=query,
                count=count,
                user_details_known=user_details_known,
            )
            logger.info(f"[Chatbot] Prompt prepared (user_details_known={user_details_known})")

            # Step 3: Generate response from LLM
            try:
                import asyncio
                if asyncio.iscoroutinefunction(generate_llm_response):
                    final_text = await generate_llm_response(prompt)
                else:
                    import functools
                    loop = asyncio.get_event_loop()
                    final_text = await loop.run_in_executor(None, functools.partial(generate_llm_response, prompt))
                safe_final_text = final_text if isinstance(final_text, str) else ""
            except Exception as llm_exc:
                logger.exception(f"[Chatbot] LLM call failed: {llm_exc}")
                async for fallback in self._fallback_response_stream("I couldn't generate a response right now."):
                    yield fallback
                return
            # Fallback if LLM returns None or empty string (e.g., due to rate limit or internal error)
            if not safe_final_text or not safe_final_text.strip():
                logger.warning("[Chatbot] LLM returned empty or None, triggering fallback response.")
                async for fallback in self._fallback_response_stream("I couldn't generate a response right now."):
                    yield fallback
                return

            # Step 4: Parse LLM JSON output (if present)
            funnel_stage = ""
            response_text = ""

            def extract_json_from_markdown(txt: str) -> str:
                # First look for a fenced JSON block
                pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                match = re.search(pattern, txt, re.DOTALL)
                if match:
                    return match.group(1).strip()

                # If no fence found, try to find the first balanced JSON object by brace matching
                start = None
                depth = 0
                for i, ch in enumerate(txt):
                    if ch == '{':
                        if start is None:
                            start = i
                        depth += 1
                    elif ch == '}' and start is not None:
                        depth -= 1
                        if depth == 0:
                            return txt[start:i+1].strip()

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
                logger.warning(f"[Chatbot] Failed to parse LLM output as JSON: {json_exc}")
                # As a best-effort, try to extract a user-facing 'response' field
                # using a simple regex fallback before giving up.
                resp_match = re.search(r'"response"\s*:\s*"([\s\S]*?)"\s*(,|})', safe_final_text)
                if resp_match:
                    response_text = resp_match.group(1).strip()
                    logger.info("[Chatbot] Recovered 'response' value from LLM text via regex fallback.")
                else:
                    response_text = safe_final_text
                user_network_id = None

            logger.info(f"[Chatbot] Final output len={len(safe_final_text)}, funnel_stage='{funnel_stage}'")
            if 'llm_json' in locals():
                logger.info(f"[LLM_DEBUG] Raw LLM JSON response: {llm_json}")  # Debug what LLM actually returns

            # Step 5: Continue with regular response flow

            # Step 6: Stream formatted response
            cleaned = (response_text or "").replace("FORM_TRIGGER", "").strip()

            # Debug markdown preservation
            has_markdown_before = any(marker in cleaned for marker in ["**", "*", "_", "#", "-", "`", "```"])
            if has_markdown_before:
                logger.info(f"[MARKDOWN_DEBUG] Raw response contains markdown: {cleaned[:100]}...")


            # Step 6: Enhanced Funnel stage handling (form trigger) BEFORE yielding assistant response
            user_details_known_db = get_user_details_known_from_db(session_id)
            logger.info(f"[FORM_DEBUG] session_id={session_id}, funnel_stage='{funnel_stage}', message_count={count}, user_details_known={user_details_known_db}")
            trigger_form = False
            trigger_reason = ""
            if not user_details_known_db:
                if funnel_stage == "action":
                    trigger_form = True
                    trigger_reason = "action_stage"
                elif funnel_stage == "intent" and count >= 3:
                    trigger_form = True
                    trigger_reason = "intent_stage_depth"
                    logger.info(f"[FORM_DEBUG] Intent stage trigger conditions met: funnel_stage='{funnel_stage}', count={count}")
                elif funnel_stage == "interest" and count >= 6:
                    trigger_form = True
                    trigger_reason = "interest_stage_engagement"
                    logger.info(f"[FORM_DEBUG] Interest stage trigger conditions met: funnel_stage='{funnel_stage}', count={count}")
                elif count >= 12:
                    trigger_form = True
                    trigger_reason = "conversation_length_fallback"
                    logger.info(f"[FORM_DEBUG] Fallback trigger conditions met: count={count}")
                else:
                    logger.info(f"[FORM_DEBUG] No trigger conditions met. funnel_stage='{funnel_stage}', count={count}")

            # Only yield the assistant response if we are NOT about to trigger a form
            if not trigger_form:
                # Always use format_response, which now preserves markdown newlines if detected
                formatted = format_response(cleaned, query, None)
                if isinstance(formatted, str):
                    formatted = formatted.replace('\\n', '\n')

                # Anti-repetition: compare with last assistant response
                if last_assistant_response and isinstance(last_assistant_response, str):
                    if self._is_too_similar(last_assistant_response, formatted):
                        logger.warning("[ANTI-REPETITION] New response is too similar to last assistant response. Consider regenerating or expanding.")
                        # Optionally, you could modify the response here, e.g., append a clarifying question or ask for more specifics
                        formatted += "\n\n(Can you share more details or specify what you'd like to know next?)"

                if has_markdown_before:
                    has_markdown_after = any(marker in formatted for marker in ["**", "*", "_", "#", "-", "`", "```"])
                    logger.info(f"[MARKDOWN_DEBUG] Formatted response markdown preserved: {has_markdown_after}")
                    if not has_markdown_after:
                        logger.warning(f"[MARKDOWN_WARNING] Markdown lost during formatting!")

                # Only yield a chunk if it is non-empty and not just whitespace
                if formatted and formatted.strip():
                    yield {"status": "chunk", "chunk": formatted}
                    if count >= 50:
                        logger.info(f"[Chatbot] [DEBUG] Yielding end_chat chunk: user_message_count={count}")
                        print(f"[DEBUG] Yielding end_chat chunk: user_message_count={count}")
                        yield {"status": "end_chat", "chunk": "Our sales team will reach out within 1 business day, Thank you for your interest in Ditstek innovations."}

            # Step 7: Emit meta update if present, but skip if form_trigger is about to be yielded
            user_details_known_db = get_user_details_known_from_db(session_id)
            meta_chunk = {
                "user_details_known": user_details_known_db,
                **({"user_network_id": user_network_id} if user_network_id else {}),
            }
            if not trigger_form and meta_chunk:
                yield {"status": "meta", "chunk": meta_chunk}

            # Step 8: Form trigger event (if needed)
            if trigger_form:
                logger.info(f"[Chatbot] FORM TRIGGER ACTIVATED! Reason: {trigger_reason}, funnel_stage={funnel_stage}, message_count={count}")
                yield {"status": "form_trigger", "chunk": ""}
            else:
                logger.info(f"[Chatbot] No form trigger. funnel_stage={funnel_stage}, message_count={count}, user_details_known={user_details_known_db}")

        except Exception as e:
            logger.exception("[Chatbot] Redis-based response generation failed")
            async for fallback in self._fallback_response_stream("I couldn't generate a response right now."):
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
                normalized_msgs.append({
                    "role": msg[0],
                    "content": msg[1],
                    "timestamp": msg[2] if len(msg) > 2 else ""
                })
        
        # Count actual conversation pairs for context
        user_msgs = [msg for msg in normalized_msgs if msg.get('role', '').lower() == 'user']
        assistant_msgs = [msg for msg in normalized_msgs if msg.get('role', '').lower() == 'assistant']
        conversation_count = min(len(user_msgs), len(assistant_msgs))
        
        # Add conversation flow summary at the top
        if conversation_count > 0:
            formatted.append(f"=== CONVERSATION FLOW ({conversation_count} exchanges, {len(normalized_msgs)} total messages) ===")
        
        # Format each message with enhanced context
        for i, msg in enumerate(normalized_msgs):
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            # Add timestamp if available
            time_info = f" [{timestamp}]" if timestamp else ""
            
            if role == 'user':
                formatted.append(f"User{time_info}: {content}")
            elif role == 'assistant':
                # Highlight questions in assistant messages to help prevent repetition
                if '?' in content:
                    questions = [q.strip() + '?' for q in content.split('?') if q.strip()]
                    if questions:
                        content_with_questions = content + f" [QUESTIONS ASKED: {'; '.join(questions[-2:])}]"  # Last 2 questions
                        formatted.append(f"Assistant{time_info}: {content_with_questions}")
                    else:
                        formatted.append(f"Assistant{time_info}: {content}")
                else:
                    formatted.append(f"Assistant{time_info}: {content}")
            else:
                # In case of unknown role, include as-is for debugging
                formatted.append(f"{role.capitalize()}{time_info}: {content}")
        
        # Add conversation context summary at the end
        if conversation_count > 0:
            formatted.append(f"=== END CONVERSATION FLOW ===")
        
        return "\n".join(formatted)
