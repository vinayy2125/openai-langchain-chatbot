"""
Chatbot Optimization Service with Streaming Support
This service provides optimized chatbot response generation with:
- Context length management
- Performance optimization
- Detailed response generation
- Robust fallback handling
- Streaming response support
"""

import logging
import sys
import tiktoken
from typing import List, Tuple, Dict, Generator, Union, AsyncGenerator, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from langchain.schema import AIMessage
import re
import time
import random
from datetime import datetime
from app.core.prompts import key_generate_prompt, stream_follow_up_generation_prompt, stream_follow_up_only_prompt, optimized_prompt, count_tokens_template, Requirements, fallback_response_prompt

# Configure the logger (configured centrally in app.main)
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

    def optimize_context(self, context: Union[str, List[str]], question: str, history: str, template_tokens: int, response_reservation: int = 512, safety_buffer: int = 64) -> Tuple[str, Dict[str, int]]:
        """Create an optimized context string that fits within token limits."""
        # Normalize context into list of chunks
        if isinstance(context, str):
            if "\n\n---\n\n" in context:
                chunks = [c for c in context.split("\n\n---\n\n") if c.strip()]
            else:
                chunks = [p for p in context.split("\n\n") if p.strip()]
        else:
            chunks = [str(c) for c in context or []]

        question_tokens = self.count_tokens_cached(question or "")
        history_tokens = self.count_tokens_cached(history or "")

        available_for_context = max(
            self.context_limit - template_tokens - question_tokens - history_tokens - response_reservation - safety_buffer,
            0,
        )

        prioritized = self.prioritize_chunks(chunks, question, max_chunks=32)

        final_chunks: List[str] = []
        current_tokens = 0
        sep = "\n\n---\n\n"
        sep_tokens = self.count_tokens_cached(sep)

        for chunk in prioritized:
            ctokens = self.count_tokens_cached(chunk)
            if current_tokens + ctokens + sep_tokens <= available_for_context:
                final_chunks.append(chunk)
                current_tokens += ctokens + sep_tokens
            else:
                remaining = available_for_context - current_tokens - sep_tokens
                if remaining > 50:
                    truncated = self.truncate_to_tokens(chunk, max(0, remaining))
                    final_chunks.append(truncated)
                    current_tokens += self.count_tokens_cached(truncated) + sep_tokens
                break

        final_context = sep.join(final_chunks)
        stats = {
            "original_chunks": len(chunks),
            "prioritized_chunks": len(prioritized),
            "final_chunks": len(final_chunks),
            "original_tokens": self.count_tokens_cached(sep.join(chunks)) if chunks else 0,
            "final_tokens": current_tokens,
            "available_for_context": available_for_context,
        }
        logger.debug(f"Context optimization stats: {stats}")
        return final_context, stats


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
        import os
        from langchain_openai import ChatOpenAI

        # Ensure api_key is None if not set, to match expected type
        from pydantic import SecretStr
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

    # ---------------- Requirement Collection Helpers -----------------
    def _init_requirement_state(self, session_id: str):
        if session_id not in self.collected_requirements:
            self.collected_requirements[session_id] = {
                "answers": {},  # key -> {'question':..., 'answer':...}
                "asked": set(),  # keys already asked
            }

    def stream_follow_up_generation(
        self,
        conversation_history: list[dict],
        latest_query: str,
        prompt_context: str,
        combined: bool = False,
        followup_count: int = 2,
    ):
        """
        Generate follow-up questions or combined answer+follow-ups.

        Key changes:
        - Yields raw text chunks only (no `data:` or extra JSON inside).
        - Follow-ups + suggestions handled via prompt.
        - Context-switch options included when unrelated query detected.
        """
        history = conversation_history or []
        session_id = next(
            (msg.get("session_id") for msg in history if msg.get("session_id")),
            "generic",
        )

        # Initialize or get conversation state
        state = self._init_conversation_state(
            str(session_id) if session_id is not None else "generic"
        )

        # Build transcript (last 10 messages)
        transcript = "\n".join(
            [
                f"{'USER' if msg.get('role') == 'user' else 'ASSISTANT'}: {msg.get('content', '')}"
                for msg in history[-10:]
            ]
        )

        category_names = ", ".join([cat["name"] for cat in self.requirement_categories])

        if combined:
            # Prompt for answer + follow-ups + suggestions
            prompt = stream_follow_up_generation_prompt(
                prompt_context=prompt_context,
                transcript=transcript,
                latest_query=latest_query,
                category_names=category_names,
                followup_count=followup_count,
            )
        else:
            # Prompt for follow-ups only
            prompt = stream_follow_up_only_prompt(
                prompt_context=prompt_context,
                latest_query=latest_query,
                transcript=transcript,
            )

        # Track generation state
        state["follow_up_count"] += 1
        state["last_generated"] = datetime.now()

        try:
            # Streaming output from LLM
            if hasattr(self.llm, "stream"):
                for chunk in self.llm.stream(prompt):
                    content = getattr(chunk, "content", str(chunk))
                    if content:
                        yield content
            else:
                # Fallback for non-streaming LLMs
                response = self.llm.invoke(prompt)
                text = getattr(response, "content", str(response))
                for line in text.split("\n"):
                    yield line

        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}", exc_info=True)
            # Simple fallback
            fallback = (
                "Could you tell me more about your goals for this project?\n"
                "- Business growth\n"
                "- Process improvement"
            )
            for line in fallback.split("\n"):
                yield line

    def get_follow_ups(self, session_id: str) -> list[str]:
        """Get follow-up questions for a session."""
        return self.follow_ups.get(session_id, [])

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
                from app.core.redis_context import get_redis_context_chunks

                if search_keys:
                    # If redis helper supports search_keys, pass them (non-breaking if ignored)
                    context_chunks = get_redis_context_chunks(session_id, query, [], top_n=8)
                else:
                    context_chunks = get_redis_context_chunks(session_id, query, [], top_n=8)

                # Log retrieved context preview for debugging
                try:
                    preview = [c[:300] for c in (context_chunks or [])[:6]]
                    logger.debug("[Optimizer] Retrieved Redis context preview: %s", preview)
                except Exception:
                    logger.debug("[Optimizer] Retrieved Redis context but failed to prepare preview")
            except Exception as e:
                logger.warning("Redis context retrieval failed, using empty context: %s", e)
                context_chunks = []

            # Continue existing flow: build prompt + call LLM (unchanged)
            # Build a joined context string (may be empty)
            context = "\n\n---\n\n".join([str(chunk) for chunk in (context_chunks or [])])
            history = self._format_history(chat_history)
            logger.info(f"[Chatbot] Redis Context Retrieved: {len(context)} characters, {len(context_chunks) if context_chunks else 0} chunks")
            logger.info(f"[Chatbot] Chat History: {len(chat_history)} messages")

            # Always construct the standard prompt using the retrieved context
            prompt = (
                "You are Ditstek Assistant. Use the following context to answer the user's question...\n"
                f"CONTEXT:\n{context}\n\nUSER QUERY: {query}"
            )

            # At this point we have `prompt` ready to send to the LLM
            logger.info(f"[Chatbot] Prompt prepared (first 300 chars): {str(prompt)[:300]}")
            try:
                # Use the centralized helper which creates proper message objects
                from app.core.utils import generate_llm_response

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

                # Stream paragraph-level chunks to callers (cleaned)
                paragraphs = [p for p in final_text.split("\n\n") if p.strip()]
                for p_idx, paragraph in enumerate(paragraphs):
                    cleaned = self._clean_response_formatting(paragraph)
                    if cleaned:
                        yield {"status": "chunk", "chunk": cleaned}
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

    def _generate_response_stream(
        self, question: str, context: str, history: str
    ) -> Generator[Any, None, None]:
        logger.debug("Entered _generate_response_stream")

        # Format context with separators and metadata
        if isinstance(context, list):
            context_chunks = []
            for i, chunk in enumerate(context):
                try:
                    if isinstance(chunk, tuple) or isinstance(chunk, list):
                        # Defensive unpack: find the first string-like element as text
                        text = None
                        meta = None
                        for el in chunk:
                            if isinstance(el, str) and el.strip():
                                text = el
                                break
                        # If not found, try first element as fallback
                        if text is None and len(chunk) > 0:
                            text = str(chunk[0])

                        # Find dict-like metadata if present
                        for el in chunk:
                            if isinstance(el, dict):
                                meta = el
                                break

                        source_info = None
                        if isinstance(meta, dict):
                            source_info = meta.get("source") or meta.get("url")
                        if not source_info:
                            source_info = "N/A"

                        context_chunks.append(f"Source {i+1} ({source_info}):\n{text}")
                    else:
                        context_chunks.append(str(chunk))
                except Exception:
                    # Best-effort: fall back to stringified chunk
                    context_chunks.append(str(chunk))
            context = "\n\n---\n\n".join(context_chunks)

        # Log the formatted context
        logger.debug("Formatted Context:")

        template_tokens = self._count_template_tokens()
        optimized_context, stats = self.context_optimizer.optimize_context(
            context, question, history, template_tokens
        )
        logger.debug("Optimization stats: ")

        prompt = self._create_optimized_prompt(history, optimized_context, question)
        # logger.debug("Final Prompt: %s", prompt)

        cache_key = f"{question[:50]}_{hash(optimized_context[:100])}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            logger.info("Returning cached response for question (source=cached)")
            # Ensure callers can see the source and receive chunk events
            yield {"status": "chunk", "chunk": cached_response, "source": "cache"}
            return

        try:
            if hasattr(self.llm, "stream"):
                stream = self.llm.stream(prompt)
                buffer = ""
                full_response = ""

                for chunk in stream:
                    content = chunk.content if hasattr(chunk, "content") else chunk
                    if not content:
                        continue

                    full_response += content
                    buffer += content

                    # Process buffer for paragraph/header/bullet-level chunking
                    while True:
                        # Paragraph boundary
                        if "\n\n" in buffer:
                            idx = buffer.find("\n\n") + 2
                            piece = buffer[:idx]
                            buffer = buffer[idx:]
                            cleaned = self._clean_response_formatting(piece)
                            if cleaned:
                                # fingerprint for debugging
                                try:
                                    fp = f"{hash(cleaned)}|{cleaned[:60].replace('\n',' ')}"
                                except Exception:
                                    fp = cleaned[:60]
                                logger.debug(f"[Optimizer] Yielding stream chunk fp={fp}")
                                yield {"status": "chunk", "chunk": cleaned, "source": "stream"}
                            continue

                        # Header at the very start
                        m = re.match(r'^(#{1,6} [^\n]+\n)', buffer)
                        if m:
                            piece = m.group(1)
                            buffer = buffer[len(piece):]
                            cleaned = self._clean_response_formatting(piece)
                            if cleaned:
                                try:
                                    fp = f"{hash(cleaned)}|{cleaned[:60].replace('\n',' ')}"
                                except Exception:
                                    fp = cleaned[:60]
                                logger.debug(f"[Optimizer] Yielding tail chunk fp={fp}")
                                yield {"status": "chunk", "chunk": cleaned, "source": "stream"}
                            continue

                        # Header later in buffer
                        idx_header = buffer.find('\n#')
                        if idx_header != -1 and idx_header > 0:
                            piece = buffer[: idx_header + 1]
                            buffer = buffer[idx_header + 1 :]
                            cleaned = self._clean_response_formatting(piece)
                            if cleaned:
                                yield {"status": "chunk", "chunk": cleaned, "source": "stream"}
                            continue

                        # Bullet/list break
                        idx_bullet = buffer.find('\n- ')
                        if idx_bullet != -1 and idx_bullet > 0:
                            piece = buffer[: idx_bullet + 1]
                            buffer = buffer[idx_bullet + 1 :]
                            cleaned = self._clean_response_formatting(piece)
                            if cleaned:
                                yield {"status": "chunk", "chunk": cleaned, "source": "stream"}
                            continue

                        # Safety flush for long buffers (avoid excessive latency)
                        if len(buffer) > 300:
                            piece = buffer[:300]
                            buffer = buffer[300:]
                            cleaned = self._clean_response_formatting(piece)
                            if cleaned:
                                yield {"status": "chunk", "chunk": cleaned, "source": "stream"}
                            continue

                        # No complete chunk ready yet
                        break


                # Send remaining content as cleaned paragraph/header chunks
                if buffer.strip():
                    cleaned = self._clean_response_formatting(buffer)
                    if cleaned:
                        paragraphs = cleaned.split("\n\n")
                        for p_idx, paragraph in enumerate(paragraphs):
                            if paragraph.strip():
                                try:
                                    fp = f"{hash(paragraph)}|{paragraph[:60].replace('\n',' ')}"
                                except Exception:
                                    fp = paragraph[:60]
                                logger.debug(f"[Optimizer] Yielding buffered paragraph fp={fp}")
                                yield {"status": "chunk", "chunk": paragraph, "source": "stream"}
                            if p_idx < len(paragraphs) - 1:
                                # preserve paragraph separation as an empty chunk separator
                                yield {"status": "chunk", "chunk": "\n\n", "source": "stream"}

                # Cache the cleaned response
                cleaned_full = self._clean_response_formatting(full_response)
                self.response_cache[cache_key] = cleaned_full
                logger.info("Completed streaming response (source=stream). Caching cleaned response.")
            else:
                raw_answer = self.llm.invoke(prompt)
                response = (
                    raw_answer.content
                    if isinstance(raw_answer, AIMessage)
                    else str(raw_answer)
                )
                # Apply response cleaning and cache
                if isinstance(response, str):
                    cleaned_response = self._clean_response_formatting(response)
                else:
                    cleaned_response = self._clean_response_formatting(str(response))
                self.response_cache[cache_key] = cleaned_response

                # Emit cleaned paragraphs as chunk events and tag source
                paragraphs = cleaned_response.split("\n\n")
                for p_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        yield {"status": "chunk", "chunk": paragraph, "source": "non-stream"}
                    if p_idx < len(paragraphs) - 1:
                        yield {"status": "chunk", "chunk": "\n\n", "source": "non-stream"}
        except Exception:
            logger.error("LLM streaming call failed", exc_info=True)
            fallback_response = self._generate_fallback_response(question, context)
            yield fallback_response

    def _would_break_markdown(self, text: str) -> bool:
        """Check if breaking at this point would damage Markdown formatting"""
        # Count unclosed bold markers
        bold_count = text.count("**")
        if bold_count % 2 != 0:
            return True

        # Check if we're in the middle of a header
        lines = text.split("\n")
        if (
            lines
            and lines[-1].strip().startswith("#")
            and not lines[-1].strip().endswith(" ")
        ):
            return True

        # Check if we're breaking a word that might be part of markdown
        if text.endswith("**") or text.endswith("*") or text.endswith("#"):
            return True

        # Check for incomplete list items
        if text.strip().endswith("-") and not text.strip().endswith(" -"):
            return True

        # Check for incomplete numbered lists
        if re.search(r"\d+\.$", text.strip()):
            return True

        return False

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
                from app.core.utils import generate_llm_response
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

                    resp = getattr(resp_obj, "content", None) or (resp_obj.get("content") if hasattr(resp_obj, "get") else str(resp_obj))
                except Exception as e:
                    logger.debug("Direct query_llm invoke failed: %s", e)

            if not resp:
                return []

            # Parse: split on newlines/commas and clean tokens
            raw = resp.strip()
            # Accept common formats: comma separated, newline separated, JSON array
            keys = []
            # quick JSON array parse attempt
            try:
                import json

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

    def _create_optimized_prompt(
        self, history: str, context: str, question: str
    ) -> str:
        length_rule = (
            "Provide direct, concise responses with minimal formatting. "
            "Limit responses to 200 words maximum. "
            "Use bold only for critical terms or concepts. "
            "Avoid headers unless absolutely necessary. "
            "Focus on answering the specific question asked. "
            "Use natural paragraph breaks sparingly. "
            "Ensure consistent single spacing between words. "
        )

        prompt = optimized_prompt(history=history, context=context, question=question, length_rule=length_rule)
        # Append an explicit mandatory formatting safety block to avoid LLM introducing spacing/markdown corruption
        full_prompt = (prompt.strip() + "\n\n").strip()
        return full_prompt

    def get_flow_debug_info(
        self, query: str, session_id: str = "default"
    ) -> Dict[str, Any]:
        """Get detailed flow information for debugging the Redis-based process"""
        try:
            # Generate search keys (for reference)
            search_keys = self._generate_search_keys(query)

            # Get context from Redis
            from app.core.redis_context import get_redis_context_chunks
            context_chunks = get_redis_context_chunks(session_id, query, [], top_n=8)
            # Prepare context preview (safe) and log it
            try:
                preview_list = [str(c)[:300] for c in (context_chunks or [])[:8]]
                logger.debug("[Optimizer:get_flow_debug_info] Redis context preview: %s", preview_list)
            except Exception:
                logger.debug("[Optimizer:get_flow_debug_info] Retrieved context but failed to prepare preview")
            context = "\n\n---\n\n".join([str(chunk) for chunk in context_chunks])

            return {
                "original_query": query,
                "generated_keys": search_keys,
                "context_length": len(context),
                "context_preview": (
                    context[:300] + "..." if len(context) > 300 else context
                ),
                "model_used_for_keys": "gpt-4o-mini",
                "model_used_for_response": getattr(self.llm, "model_name", "Unknown"),
                "enhancement_status": "Redis context flow active",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "enhancement_status": "Redis context flow failed, using fallback",
                "timestamp": datetime.now().isoformat(),
            }

    def _fallback_web_search(self, query: str, site: str) -> str:
        try:
            from app.core.search_client import search_site
            from core_services.crawler.scraper import scrape_url

            search_results = search_site(query, site)
            scraped_texts = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for res in search_results:
                    url = res.get("url")
                    if url:
                        futures.append(executor.submit(scrape_url, url))
                for future in futures:
                    try:
                        result = future.result(timeout=10)
                        if result:
                            scraped_texts.append(result)
                    except Exception:
                        continue
            return "\n\n".join(scraped_texts[:8])
        except Exception as e:
            logger.error("Web search failed", exc_info=True)
            return "No additional context available from web search."

    def _generate_fallback_response(self, question: str, context: str) -> str:
        return fallback_response_prompt(question=question, context=context)


# Follow-ups are now handled internally by the OptimizedChatbot class
