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

    def get_detailed_response(self, query: str, chat_history: List[tuple], session_id: str, stream: bool = True):
        from app.api.v1.helpers import get_user_details_known_from_db  # Do Not Move Outside Function

        """
        Generate a detailed response for a query and stream structured events.
        Yields dict events used by the API layer: 
        {status: 'chunk'|'complete_chunk'|'form_trigger'|'meta', 'chunk': ...}
        """
        try:
            # Step 1: Retrieve Redis context
            try:
                context_chunks = get_redis_context_chunks(session_id, query, [], top_n=4)
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            context = "\n\n---\n\n".join(map(str, context_chunks or []))
            history = self._format_history(chat_history)
            logger.info(f"[Chatbot] Redis Context Retrieved: {len(context)} chars, {len(context_chunks)} chunks")
            logger.info(f"[Chatbot] Chat History: {len(chat_history)} messages")

            # Step 2: Prepare prompt
            user_details_known = get_user_details_known_from_db(session_id)
            logger.info(f"[Chatbot] user_details_known (DB) = {user_details_known} for session {session_id}")

            prompt = final_response_prompt(
                prompt_context=context,
                conversation_summary=history or "",
                query=query,
                user_details_known=user_details_known,
            )
            logger.info(f"[Chatbot] Prompt prepared (user_details_known={user_details_known})")

            # Step 3: Generate response from LLM
            try:
                final_text = generate_llm_response(prompt)
                safe_final_text = final_text if isinstance(final_text, str) else ""
            except Exception as llm_exc:
                logger.exception(f"[Chatbot] LLM call failed: {llm_exc}")
                yield from self._fallback_response_stream("I couldn't generate a response right now.")
                return

            # Step 4: Parse LLM JSON output (if present)
            funnel_stage = ""
            response_text = ""

            def extract_json_from_markdown(txt: str) -> str:
                pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                match = re.search(pattern, txt, re.DOTALL)
                return match.group(1).strip() if match else txt.strip()

            try:
                cleaned_json = extract_json_from_markdown(safe_final_text)
                llm_json = json.loads(cleaned_json)
                response_text = llm_json.get("response", "") or ""
                funnel_stage = (llm_json.get("funnel_stage", "") or "").lower()
                user_details_known = bool(llm_json.get("user_details_known", False))
                user_network_id = llm_json.get("user_network_id") or None
                logger.info(f"[Chatbot] Parsed JSON: {llm_json}")
            except Exception as json_exc:
                logger.warning(f"[Chatbot] Failed to parse LLM output as JSON: {json_exc}")
                response_text = safe_final_text
                user_network_id = None

            logger.info(f"[Chatbot] Final output len={len(safe_final_text)}, funnel_stage='{funnel_stage}'")

            # Step 5: Stream formatted response
            cleaned = (response_text or "").replace("FORM_TRIGGER", "").strip()
            formatted = format_response(cleaned, query, None)
            if formatted:
                yield {"status": "chunk", "chunk": formatted}
            yield {"status": "complete_chunk", "chunk": ""}

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
            yield from self._fallback_response_stream("I couldn't generate a response right now.")



    def _fallback_response_stream(self, query: str) -> Generator[str, None, None]:
        """Fallback response when enhanced flow fails"""
        fallback_message = f"I understand you're asking about: {query}. Let me provide a general response based on my knowledge."
        yield fallback_message


    def _format_history(self, chat_history: list) -> str:
        return "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )
