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
        from app.api.v1.helpers import get_user_details_known_from_db    #Do Not Move Outside funtion

        """Generate a detailed response for a query and stream structured events.

        Yields dict events used by the API layer: {status: 'chunk'|'complete_chunk'|'form_trigger', 'chunk': ...}
        """
        try:
            try:
                context_chunks = get_redis_context_chunks(session_id, query, [], top_n=4)
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            context = "\n\n---\n\n".join([str(chunk) for chunk in (context_chunks or [])])
            history = self._format_history(chat_history)
            logger.info(f"[Chatbot] Redis Context Retrieved: {len(context)} characters, {len(context_chunks) if context_chunks else 0} chunks")
            logger.info(f"[Chatbot] Chat History: {len(chat_history)} messages")

            # Construct prompt
            history_str = history or ""
            user_details_known = get_user_details_known_from_db(session_id)
            logger.info(f"[Chatbot] user_details_known (DB) = {user_details_known} for session {session_id}")
            prompt = final_response_prompt(prompt_context=context, conversation_summary=history_str, query=query, user_details_known=user_details_known)
            logger.info(f"[Chatbot] Prompt prepared from final_response_prompt (user_details_known={user_details_known})")

            try:
                # Primary path: centralized helper that returns a final text
                final_text = generate_llm_response(prompt)

                funnel_stage = ""
                response_text = ""
                safe_final_text = final_text if isinstance(final_text, str) else ""

                # Extract JSON from markdown code blocks if present
                def extract_json_from_markdown(txt: str) -> str:
                    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                    match = re.search(json_pattern, txt, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    return txt.strip()

                try:
                    # Try to extract JSON from markdown first
                    cleaned_json = extract_json_from_markdown(safe_final_text)
                    logger.info(f"[Chatbot] Cleaned JSON for parsing: {cleaned_json}")
                    llm_json = json.loads(cleaned_json)
                    logger.info(f"[Chatbot] Parsed JSON object: {llm_json}")
                    response_text = llm_json.get("response", "") or ""
                    funnel_stage = (llm_json.get("funnel_stage", "") or "").lower()
                    # New optional fields to signal backend/session
                    user_details_known = bool(llm_json.get("user_details_known", False))
                    user_network_id = llm_json.get("user_network_id") if llm_json.get("user_network_id") else None
                    logger.info(f"[Chatbot] Extracted response_text: '{response_text}'")
                    logger.info(f"[Chatbot] Extracted funnel_stage: '{funnel_stage}' (original: '{llm_json.get('funnel_stage', '')}')")
                except Exception as json_exc:
                    logger.error(f"[Chatbot] Failed to parse LLM output as JSON: {json_exc}")
                    logger.info(f"[Chatbot] Raw LLM output: {safe_final_text}")
                    response_text = safe_final_text

                logger.info(f"[Chatbot] Final LLM output length={len(safe_final_text)}; preview: {safe_final_text}")
                logger.info(f"[Chatbot] Parsed funnel_stage: '{funnel_stage}' (type: {type(funnel_stage)})")

                # Stream the main response chunk (minimal cleaning, preserve LLM output)
                cleaned = str(response_text or "").replace("FORM_TRIGGER", "").strip()
                formatted = format_response(cleaned, query, None)
                if formatted:
                    yield {"status": "chunk", "chunk": formatted}
                yield {"status": "complete_chunk", "chunk": ""}

                # Emit a backend meta event so upstream can update session state once
                meta = {"status": "meta", "chunk": {"user_details_known": user_details_known}}
                if user_network_id:
                    meta["chunk"]["user_network_id"] = user_network_id
                # Only emit meta event if any value is present
                if meta["chunk"].get("user_details_known") or meta["chunk"].get("user_network_id"):
                    yield meta

                # Funnel stage handling for form trigger (backend can use this value)
                logger.info(f"[Chatbot] Checking funnel stage for form trigger: '{funnel_stage}' == 'action'? {funnel_stage == 'action'}")
                # Defensive: only emit form_trigger if form_shown is not already set
                user_details_known_db = get_user_details_known_from_db(session_id)
                logger.info(f"[Chatbot] DB value: user_details_known={user_details_known_db} for session {session_id}")
                if funnel_stage == "action" and not user_details_known_db:
                    logger.info("[Chatbot] FORM TRIGGER ACTIVATED! (user_details_known is False)")
                    yield {"status": "form_trigger", "chunk": ""}
                elif funnel_stage == "action" and user_details_known_db:
                    logger.info("[Chatbot] Funnel stage is action but user_details_known is True; not emitting trigger again.")
                else:
                    logger.info(f"[Chatbot] Form trigger NOT activated. Funnel stage is: '{funnel_stage}'")

            except Exception as llm_exc:
                logger.exception(f"[Chatbot] LLM call failed with exception: {llm_exc}")
                # Use a safer, shorter fallback message instead of echoing entire prompt
                yield from self._fallback_response_stream("I couldn't generate a response right now.")
        except Exception as e:
            logger.exception("[Chatbot] Redis-based response generation failed")
            yield from self._fallback_response_stream("I couldn't generate a response right now.")


    def _fallback_response_stream(self, query: str) -> Generator[str, None, None]:
        """Fallback response when enhanced flow fails"""
        fallback_message = f"I understand you're asking about: {query}. Let me provide a general response based on my knowledge."
        yield fallback_message


    def _generate_search_keys(self, query: str) -> List[str]:
        try:
            # key_generate_prompt is a function that returns a prompt string when called
            try:
                prompt = key_generate_prompt(query)
            except Exception:
                # fallback: if key_generate_prompt was accidentally replaced by a template string
                try:
                    prompt = str(key_generate_prompt).format(query=query)
                except Exception:
                    prompt = f"Generate search keys for: {query}"
            
            messages = [
                {"role": "system", "content": "Generate short, comma-separated search keys for retrieval."},
                {"role": "user", "content": prompt},
            ]
            
            try:
                resp = generate_llm_response(messages)
            except Exception:
                resp = None
                try:
                    invoke_payload = None
                    try:
                        invoke_payload = messages
                        resp_obj = self.query_llm.invoke(invoke_payload)
                    except Exception:
                        invoke_payload = [(m["role"], m["content"]) for m in messages]
                        resp_obj = self.query_llm.invoke(invoke_payload)

                    # Extract content safely from various possible response object shapes
                    resp = None
                    try:
                        resp = getattr(resp_obj, "content", None)
                    except Exception:
                        resp = None

                    if not resp:
                        try:
                            # dict-like objects may implement .get; obtain it safely
                            get_fn = getattr(resp_obj, "get", None)
                            if callable(get_fn):
                                resp = get_fn("content")
                            else:
                                resp = None
                        except Exception:
                            try:
                                resp = str(resp_obj)
                            except Exception:
                                resp = None
                except Exception as e:
                    logger.exception("Direct query_llm invoke failed: %s", e)

            if not resp:
                return []

            raw = str(resp).strip()
            keys = []
            try:
                

                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    keys = [str(k).strip() for k in parsed if k]
            except Exception:
                # fallback heuristics
                for sep in ("\n", ",", ";"):
                    if sep in raw:
                        keys = [k.strip() for k in raw.split(sep) if k.strip()]
                        break
                if not keys:
                    keys = [raw]

            return keys[:10]
        except Exception as exc:
            logger.exception("Error generating search keys: %s", exc)
            return []

    def _format_history(self, chat_history: list) -> str:
        return "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )
