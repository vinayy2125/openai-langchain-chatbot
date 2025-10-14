import logging
import json
import sys
import re
import os
import tiktoken
from pydantic import SecretStr
from typing import List, Generator
from functools import lru_cache
from langchain_openai import ChatOpenAI
from datetime import datetime
from app.core.prompts import key_generate_prompt, Requirements, final_response_prompt
from app.core.redis_context import get_redis_context_chunks
from app.core.utils import generate_llm_response
from app.core.response_formatter import format_response



# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("chatbot")


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
    """
    Streaming-only chatbot service with optimized context handling.
    """

    def __init__(self, llm, model: str = "gpt-4o"):
        self.llm = llm
        self.model = model
        self.follow_ups = {}
        self.session_data = {}
        self.conversation_history = {}
        self.context_optimizer = ContextOptimizer(model)
        self.response_cache = {}
        self.generated_followups = []  

        api_key = os.getenv("OPENAI_API_KEY")
        self.query_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=SecretStr(api_key) if api_key else None,
        )
        logger.info(
            "Initialized OptimizedChatbot with GPT-4o-mini for query processing"
        )
        self.requirement_categories = Requirements.requirement_categories
        self.collected_requirements: dict[str, dict] = {}

        self.requirement_categories = Requirements.requirement_categories

        self.conversation_state = {} 


    def reset_follow_up_count(self, session_id: str):
        """Reset the follow-up count for a session, useful when switching prompts."""
        if session_id in self.conversation_state:
            self.conversation_state[session_id]["follow_up_count"] = 0
            return True
        return False

    def get_session_data(self, session_id: str) -> dict:
        """Get session data for a given session ID."""
        return self.session_data.get(session_id, {})

    def initialize_session(self, session_id: str, initial_data: dict) -> None:
        """Initialize a new session with data."""
        self.session_data[session_id] = initial_data
        self.conversation_history[session_id] = []

    def add_to_conversation_history(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Add a message to the conversation history."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        self.conversation_history[session_id].append({"role": role, "content": content})

    def get_conversation_history(self, session_id: str) -> List[dict]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id, [])

    def format_conversation_history(self, history: List[dict]) -> str:
        """Format conversation history into a string."""
        formatted = []
        for msg in history:
            formatted.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(formatted)

    def check_requirements(self, session_id: str) -> bool:
        """Check if all required information is collected."""
        data = self.get_session_data(session_id)
        return all(data.get(key) for key in ["initial_prompt", "state"])

    def get_detailed_response(
        self,
        query: str,
        chat_history: list,
        session_id: str = "default",
        stream: bool = True,
    ) -> Generator:

        try:
            try:
                search_keys = self._generate_search_keys(query) or []
                logger.debug("Generated search keys for query: %s", search_keys)
            except Exception as e:
                logger.debug("Search-key generation failed, continuing without keys: %s", e)
                search_keys = []

            try:

                if search_keys:
                    context_chunks = get_redis_context_chunks(session_id, query, [], top_n=4)
                else:
                    context_chunks = get_redis_context_chunks(session_id, query, [], top_n=4)
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            context = "\n\n---\n\n".join([str(chunk) for chunk in (context_chunks or [])])
            history = self._format_history(chat_history)
            logger.info(f"[Chatbot] Redis Context Retrieved: {len(context)} characters, {len(context_chunks) if context_chunks else 0} chunks")
            logger.info(f"[Chatbot] Chat History: {len(chat_history)} messages")

            # Construct the unified final response prompt using centralized template
            history_str = history or ""
            prompt = final_response_prompt(prompt_context=context, conversation_summary=history_str)
            logger.info("[Chatbot] Prompt prepared (first 300 chars): %s", str(prompt)[:300])
            try:
                # Use the centralized helper which creates proper message objects

                final_text = generate_llm_response(prompt)

                if not final_text:
                    logger.warning("[Chatbot] LLM produced no content (None/empty). Falling back.")
                    logger.debug("[Chatbot] Prompt that caused fallback (first 800 chars): %s", str(prompt)[:800])
                    try:
                        with open(r"d:\Chatbot\logs\llm_server_diag.log", "a", encoding="utf-8") as f:
                            f.write(f"{datetime.utcnow().isoformat()} - generate_llm_response returned falsy for session={session_id} repr={repr(final_text)[:200]}\n")
                    except Exception:
                        logger.debug("Failed to write llm_server_diag.log")
                    yield from self._fallback_response_stream("I couldn't generate a response right now.")
                    return

                logger.info(f"[Chatbot] Final LLM output length={len(final_text)}; preview: {final_text[:300]}")

                # Stream paragraph-level chunks to callers; apply cleaning and formatting
                paragraphs = [p for p in final_text.split("\n\n") if p.strip()]
                for p_idx, paragraph in enumerate(paragraphs):
                    cleaned = self._clean_response_formatting(paragraph)
                    # Apply higher-level formatting rules per paragraph
                    formatted = format_response(cleaned, query, None)
                    if formatted:
                        yield {"status": "chunk", "chunk": formatted}
                    # Preserve paragraph separation as an explicit small chunk
                    if p_idx < len(paragraphs) - 1:
                        yield {"status": "chunk", "chunk": "\n\n"}

                yield {"status": "complete_chunk", "chunk": ""}

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

    def _clean_response_formatting(self, text: str) -> str:

        if not text:
            return ""

        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove unwanted leading headers
        unwanted_headers = [
            r"^#{1,6}\s*Quick Overview.*?\n",
            r"^#{1,6}\s*Overview of.*?\n",
            r"^#{1,6}\s*About.*?\n",
            r"^#{1,6}\s*Introduction.*?\n",
            r"^#{1,6}\s*Implementation\s+Approach.*?\n",
            r"^#{1,6}\s*Next\s+Steps.*?\n",
        ]
        for pattern in unwanted_headers:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        text = re.sub(r"(?<!\*) {2,}(?!\*)", " ", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n +(?![#*-])", "\n", text)
        text = re.sub(r"([.!?:;,])([ \t]+)", r"\1 ", text)
        text = re.sub(r"(\n)(#{1,3})\s+", r"\1\2 ", text)
        text = re.sub(r"(\n)([-*•])\s+", r"\1\2 ", text)
        text = re.sub(r"(\n)(\d+\.)\s+", r"\1\2 ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\*\*\s*(.*?)\s*\*\*", r"**\1**", text, flags=re.DOTALL)
        text = re.sub(r"`\s*(.*?)\s*`", r"`\1`", text)
        text = re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", lambda m: re.sub(r"\s+", "", m.group(0)), text)
        text = re.sub(r"(?<=\S)[ \t]*-[ \t]*(?=\S)", "-", text)

        def _fix_protocol(m):
            scheme = m.group(1)
            rest = m.group(2)
            # remove internal whitespace in the URL-like segment
            cleaned = re.sub(r"\s+", "", rest)
            return f"{scheme}://{cleaned}"

        text = re.sub(r"(https?)\s*:\s*/\s*/\s*([^\s)]+)", _fix_protocol, text, flags=re.IGNORECASE)

        # Clean up markdown links where URL parts have spaces (conservative)
        def _fix_link(m):
            label = m.group(1).strip()
            url = re.sub(r"\s+", "", m.group(2))
            return f"[{label}]({url})"

        text = re.sub(r"\[\s*(.*?)\s*\]\s*\(\s*([^\)]+)\s*\)", _fix_link, text)
        text = re.sub(r"([A-Za-z0-9])\s+\.\s+([A-Za-z]{2,})", r"\1.\2", text)
        text = re.sub(r"([A-Za-z0-9/._-])\s+/\s+([A-Za-z0-9/._-])", r"\1/\2", text)
        text = re.sub(r"(\S)(\*\*[^\*]+\*\*)", r"\1 \2", text)
        text = re.sub(r"(\S)(`[^`]+`)", r"\1 \2", text)
        text = re.sub(r":\s*([-*•]\s+)", r":\n\1", text)
        text = re.sub(r"(?m)^[ \t]*([-*•])\s*", r"\1 ", text)

        return text.strip()

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
                    logger.debug("Direct query_llm invoke failed: %s", e)

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
