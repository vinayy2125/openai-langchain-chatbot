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
        try:
            # Step 1: Retrieve Redis context
            try:
                # Convert incoming chat_history (list of tuples) to the dict shape
                # expected by get_redis_context_chunks so it can augment the
                # search query with recent user messages.
                conversation_history_for_redis = []
                if chat_history:
                    try:
                        for role, content in chat_history:
                            conversation_history_for_redis.append({"role": role, "content": content})
                    except Exception:
                        # If chat_history isn't iterable of tuples, ignore and pass empty
                        conversation_history_for_redis = []

                context_chunks = get_redis_context_chunks(session_id, query, conversation_history_for_redis, top_n=4)
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            context = "\n\n---\n\n".join(map(str, context_chunks or []))
            history = self._format_history(chat_history)
            count = len(chat_history)
            logger.info(f"[Chatbot] Redis Context Retrieved: {len(context)} chars, {len(context_chunks)} chunks")
            logger.info(f"[Chatbot] Chat History: {len(chat_history)} messages")

            # 1. Yield processing chunk
            yield {"status": "processing", "message": "Preparing response..."}

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
                user_details_known = bool(llm_json.get("user_details_known", False))
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

            # Step 5: Stream formatted response
            cleaned = (response_text or "").replace("FORM_TRIGGER", "").strip()
            formatted = format_response(cleaned, query, None)
            if formatted:
                yield {"status": "chunk", "chunk": formatted}
                if count >= 50:
                    logger.info(f"[Chatbot] [DEBUG] Yielding end_chat chunk: user_message_count={count}")
                    print(f"[DEBUG] Yielding end_chat chunk: user_message_count={count}")
                    yield {"status": "end_chat", "chunk": "Our sales team will reach out within 1 business day, Thank you for your interest in Ditstek innovations."}

            # Step 6: Emit meta update if present
            meta_chunk = {
                "user_details_known": user_details_known,
                **({"user_network_id": user_network_id} if user_network_id else {}),
            }
            if meta_chunk:
                yield {"status": "meta", "chunk": meta_chunk}

            # Step 7: Funnel stage handling (form trigger)
            user_details_known_db = get_user_details_known_from_db(session_id)
            trigger_form = funnel_stage == "action" and not user_details_known_db

            if trigger_form:
                logger.info("[Chatbot] FORM TRIGGER ACTIVATED! (user_details_known=False)")
                yield {"status": "form_trigger", "chunk": ""}
            else:
                logger.info(f"[Chatbot] No form trigger. funnel_stage={funnel_stage}, user_details_known={user_details_known_db}")

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
        return "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )
