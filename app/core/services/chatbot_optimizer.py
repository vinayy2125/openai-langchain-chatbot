import logging
import json
import sys
import re
import os
import tiktoken
from pydantic import SecretStr
from app.core.utils import generate_llm_response
from typing import List, Tuple, Generator
from functools import lru_cache
from langchain_openai import ChatOpenAI
from datetime import datetime
from app.core.prompts import key_generate_prompt, count_tokens_template, Requirements, final_response_prompt
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

    def score_chunk_relevance(self, chunk: str, question: str) -> float:
        if not question:
            return 0.0
        question_words = set(question.lower().split())
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        total_question_words = max(len(question_words), 1)
        relevance_score = overlap / total_question_words
        key_terms = ["what", "how", "why", "when", "where", "who", "which"]
        for term in key_terms:
            if term in question.lower() and term in chunk.lower():
                relevance_score += 0.1
        return min(relevance_score, 1.0)

    def prioritize_chunks(self, chunks: List[str], question: str, max_chunks: int = 8) -> List[str]:
        """Deterministically score and return top chunks by relevance and length."""
        if not chunks:
            return []

        scored: List[Tuple[float, int, str]] = []
        for idx, chunk in enumerate(chunks):
            try:
                score = self.score_chunk_relevance(chunk, question)
            except Exception:
                score = 0.0
            # Prefer longer chunks when relevance is equal
            scored.append((score, len(chunk), chunk))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        prioritized = [c for _, _, c in scored[:max_chunks]]
        return prioritized

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

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
        # Initialize optimization + caching utilities (previously misplaced)
        self.context_optimizer = ContextOptimizer(model)
        self.response_cache = {}
        self.generated_followups = []  # Store generated follow-ups internally

        # ADD: Separate LLM for query key generation with GPT-4o-mini
       

        # Ensure api_key is None if not set, to match expected type
        api_key = os.getenv("OPENAI_API_KEY")
        self.query_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=SecretStr(api_key) if api_key else None,
        )
        logger.info(
            "Initialized OptimizedChatbot with GPT-4o-mini for query processing"
        )
        # Ordered discovery categories for requirement elicitation (10 criteria)
        self.requirement_categories = Requirements.requirement_categories
        # Track collected category answers per session
        self.collected_requirements: dict[str, dict] = {}

        # Keep requirement_categories as reference topics but don't rigidly follow them
        self.requirement_categories = Requirements.requirement_categories

        # Track conversation state differently - more flexible
        self.conversation_state = {}  # session_id -> state object

    def _init_conversation_state(self, session_id: str):
        """Initialize a flexible conversation state tracker."""
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                "topics_covered": set(),  # Topics we've discussed
                "topics_to_explore": set(),  # Dynamically discovered topics to ask about
                "user_context": {},  # Key insights about user/project
                "follow_up_strategy": "explore",  # explore, deepen, clarify, challenge, summarize
                "follow_up_count": 0,  # How many follow-ups we've asked
                "last_generated": None,  # Timestamp of last generation
            }
        return self.conversation_state[session_id]

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
        """
        Enhanced response generation with Redis context retrieval:
        1. Query + chat_history → Redis context chunks
        2. Query + Context + History + Instructions → Main LLM → Response
        """
        try:
            # Try generating compact search keys via GPT-4o-mini to improve retrieval
            try:
                search_keys = self._generate_search_keys(query) or []
                logger.debug("Generated search keys for query: %s", search_keys)
            except Exception as e:
                logger.debug("Search-key generation failed, continuing without keys: %s", e)
                search_keys = []

            # Prefer keyed retrieval when available; fall back to legacy retrieval call
            try:

                if search_keys:
                    # If redis helper supports search_keys, pass them (non-breaking if ignored)
                    context_chunks = get_redis_context_chunks(session_id, query, [], top_n=4)
                else:
                    context_chunks = get_redis_context_chunks(session_id, query, [], top_n=4)
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            # Continue existing flow: build prompt + call LLM (unchanged)
            # Build a joined context string (may be empty)
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

                # If helper returned None or empty, treat as LLM failure and fallback
                if not final_text:
                    logger.warning("[Chatbot] LLM produced no content (None/empty). Falling back.")
                    # Log the prompt that caused the fallback (truncated)
                    logger.debug("[Chatbot] Prompt that caused fallback (first 800 chars): %s", str(prompt)[:800])
                    # Write a small diagnostic file with the exact returned value repr to aid debugging
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

                # Final completion event: don't re-emit the full cleaned text (it was
                # already streamed paragraph-by-paragraph). Emit an empty completion
                # chunk as a completion signal to avoid duplicate content on the client.
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
        """Clean response formatting to ensure consistency and remove unwanted elements.

        This function applies conservative, deterministic regex fixes to common issues:
        - collapses accidental spaced letters (heuristic)
        - normalizes bold and inline code markers
        - fixes spaces around hyphens and in URLs/markdown links
        - preserves list structure and avoids aggressive transformations that may break Markdown
        """

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

        # Collapse excessive internal spaces but preserve markdown delimiters
        text = re.sub(r"(?<!\*) {2,}(?!\*)", " ", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n +(?![#*-])", "\n", text)

        # Ensure single space after punctuation but don't collapse newlines into spaces
        # (use spaces/tabs only in the match so newlines are preserved)
        text = re.sub(r"([.!?:;,])([ \t]+)", r"\1 ", text)

        # Normalize headers, bullets and numbered lists spacing
        text = re.sub(r"(\n)(#{1,3})\s+", r"\1\2 ", text)
        text = re.sub(r"(\n)([-*•])\s+", r"\1\2 ", text)
        text = re.sub(r"(\n)(\d+\.)\s+", r"\1\2 ", text)

        # Limit repeated blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize bold (**bold**)
        text = re.sub(r"\*\*\s*(.*?)\s*\*\*", r"**\1**", text, flags=re.DOTALL)

        # Normalize inline code ticks: ` code ` -> `code`
        text = re.sub(r"`\s*(.*?)\s*`", r"`\1`", text)

        # Collapse spaced letters heuristic: 'D i t s t e k' -> 'Ditstek'
        text = re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", lambda m: re.sub(r"\s+", "", m.group(0)), text)

        # Remove spaces around hyphens between non-newline non-space characters: 'AI - powered' -> 'AI-powered'
        # Use [ \t] instead of \s so we don't match newlines (which would collapse list markers)
        text = re.sub(r"(?<=\S)[ \t]*-[ \t]*(?=\S)", "-", text)

        # Normalize protocols/URLs that may have been split across spaces, be conservative:
        # Only remove spaces that appear directly around :// or within domain/path tokens, but
        # avoid touching natural language that may contain spaced words.
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

        # Post-process common URL-space artifacts: remove spaces around dots and slashes in URLs
        # e.g., 'example .com' -> 'example.com', 'example / path' -> 'example/path'
        text = re.sub(r"([A-Za-z0-9])\s+\.\s+([A-Za-z]{2,})", r"\1.\2", text)
        text = re.sub(r"([A-Za-z0-9/._-])\s+/\s+([A-Za-z0-9/._-])", r"\1/\2", text)

        # Ensure a space before an immediately following bold or code token if missing
        text = re.sub(r"(\S)(\*\*[^\*]+\*\*)", r"\1 \2", text)
        text = re.sub(r"(\S)(`[^`]+`)", r"\1 \2", text)

        # If colon is immediately followed by a bullet, put the bullet on a new line
        text = re.sub(r":\s*([-*•]\s+)", r":\n\1", text)

        # Final pass: normalize bullets to have a single space after hyphen on their own lines
        # Normalize start-of-line hyphens and preserve existing newlines. We intentionally
        # avoid inserting newlines before bullets (except when a colon explicitly precedes them)
        # to reduce the risk of merging or deleting lines.
        text = re.sub(r"(?m)^[ \t]*([-*•])\s*", r"\1 ", text)

        return text.strip()

    def _generate_search_keys(self, query: str) -> List[str]:
        """
        Use self.query_llm (GPT-4o-mini) with key_generate_prompt to produce compact search keys.
        Returns a list of simple tokens/phrases to bias Redis vector retrieval.
        This is resilient: on any error it returns an empty list so retrieval falls back to normal flow.
        """
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
            # query_llm is a LangChain ChatOpenAI-like object; use generate_llm_response wrapper where suitable
            # Use the local query_llm for deterministic low-cost key generation
            messages = [
                {"role": "system", "content": "Generate short, comma-separated search keys for retrieval."},
                {"role": "user", "content": prompt},
            ]
            # Prefer using generate_llm_response utility if available
            # Prefer using generate_llm_response utility if available (handles message conversion)
            try:
                resp = generate_llm_response(messages)
            except Exception:
                # Fallback to direct call on self.query_llm using a tolerant interface
                resp = None
                try:
                    # Some LLM wrappers accept a list of simple dict messages or tuples; try both safely
                    invoke_payload = None
                    try:
                        # try passing the raw messages list first
                        invoke_payload = messages
                        resp_obj = self.query_llm.invoke(invoke_payload)
                    except Exception:
                        # try converting to tuples (role, content)
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

            # Parse: split on newlines/commas and clean tokens
            raw = str(resp).strip()
            # Accept common formats: comma separated, newline separated, JSON array
            keys = []
            # quick JSON array parse attempt
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
                    # single-line fallback: take up to 5 space-separated phrases
                    keys = [raw]

            # Limit to reasonable count
            return keys[:10]
        except Exception as exc:
            logger.exception("Error generating search keys: %s", exc)
            return []

    # _retrieve_context is obsolete and removed: all context now comes from Redis
    def _format_history(self, chat_history: list) -> str:
        return "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )

    @lru_cache(maxsize=1)
    def _count_template_tokens(self) -> int:
        template = count_tokens_template()
        return self.context_optimizer.count_tokens_cached(template)

    


