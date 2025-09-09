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
from typing import List, Tuple, Dict, Generator, Union
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import queue
import threading
import re
import time
from langchain.schema import AIMessage

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("chatbot")

class ContextOptimizer:
    """
    Handles context optimization to fit within model token limits
    while maximizing relevance and information retention.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.model_limits = {
            "gpt-4o-mini": 128000
        }
        self.context_limit = self.model_limits.get(model, 4096)
    
    @lru_cache(maxsize=1000)
    def count_tokens_cached(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def score_chunk_relevance(self, chunk: str, question: str) -> float:
        question_words = set(question.lower())
        chunk_words = set(chunk.lower())
        overlap = len(question_words.intersection(chunk_words))
        total_question_words = len(question_words)
        if total_question_words == 0:
            return 0.0
        relevance_score = overlap / total_question_words
        key_terms = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        for term in key_terms:
            if term in question.lower() and term in chunk.lower():
                relevance_score += 0.1
        return min(relevance_score, 1.0)
    
    def prioritize_chunks(self, chunks: List[str], question: str, max_chunks: int = 8) -> List[str]:
        if not chunks:
            return []
        with ThreadPoolExecutor(max_workers=4) as executor:
            scores = list(executor.map(lambda chunk: self.score_chunk_relevance(chunk, question), chunks))
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        prioritized_chunks = [chunk for chunk, score in chunk_scores[:max_chunks]]
        logger.debug(f"Prioritized {len(chunks)} chunks to {len(prioritized_chunks)} most relevant")
        return prioritized_chunks
    
    def optimize_context(self, context: str, question: str, history: str, template_tokens: int) -> Tuple[str, Dict]:
        question_tokens = self.count_tokens_cached(question)
        history_tokens = self.count_tokens_cached(history)
        response_reservation = 1000
        safety_buffer = 500
        available_for_context = (
            self.context_limit - template_tokens - question_tokens - 
            history_tokens - response_reservation - safety_buffer
        )
        logger.debug(f"Available tokens for context: {available_for_context}")
        if isinstance(context, list):
            context = "\n\n---\n\n".join(context)
        chunks = context.split("\n\n---\n\n")
        prioritized_chunks = self.prioritize_chunks(chunks, question)
        optimized_context = []
        current_tokens = 0
        for chunk in prioritized_chunks:
            chunk_tokens = self.count_tokens_cached(chunk)
            separator_tokens = self.count_tokens_cached("\n\n---\n\n")
            if current_tokens + chunk_tokens + separator_tokens <= available_for_context:
                optimized_context.append(chunk)
                current_tokens += chunk_tokens + separator_tokens
            else:
                remaining_tokens = available_for_context - current_tokens
                if remaining_tokens > 100:
                    partial_chunk = self.truncate_to_tokens(chunk, remaining_tokens - separator_tokens)
                    optimized_context.append(partial_chunk)
                    current_tokens += self.count_tokens_cached(partial_chunk) + separator_tokens
                break
        final_context = "\n\n---\n\n".join(optimized_context)
        optimization_stats = {
            "original_chunks": len(chunks),
            "prioritized_chunks": len(prioritized_chunks),
            "final_chunks": len(optimized_context),
            "original_tokens": self.count_tokens_cached(context),
            "final_tokens": self.count_tokens_cached(final_context),
            "tokens_saved": self.count_tokens_cached(context) - self.count_tokens_cached(final_context)
        }
        logger.debug(f"Context optimization stats: {optimization_stats}")
        return final_context, optimization_stats
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])

class OptimizedChatbot:
    """
    Streaming-only chatbot service with optimized context handling.
    """
    
    def __init__(self, llm, model: str = "gpt-4o-mini"):
        self.llm = llm
        self.model = model
        self.context_optimizer = ContextOptimizer(model)
        self.response_cache = {}
        
    def get_detailed_response(self, query: str, chat_history: list, site: str = "ditstek.com", stream: bool = False) -> Generator:
        context = self._retrieve_context(query, site)
        history = self._format_history(chat_history)
        logger.debug(">>> Query: %s", query)
        logger.debug(">>> Context Retrieved: %s", context)
        logger.debug(">>> Chat History: %s", history)
        return self._generate_response_stream(query, context, history) if stream else self._generate_response(query, context, history)
    
    def _generate_response_stream(self, question: str, context: str, history: str) -> Generator[str, None, None]:
        logger.debug("Entered _generate_response_stream")
        if isinstance(context, list):
            context_chunks = []
            for i, chunk in enumerate(context):
                if isinstance(chunk, tuple):
                    text, meta = chunk
                    source_info = meta.get("source", meta.get("url", "N/A"))
                    context_chunks.append(f"Source {i+1} ({source_info}):\n{text}")
                else:
                    context_chunks.append(str(chunk))
            context = "\n\n---\n\n".join(context_chunks)

        template_tokens = self._count_template_tokens()
        optimized_context, stats = self.context_optimizer.optimize_context(context, question, history, template_tokens)
        logger.debug("Optimization stats: %s", stats)

        prompt = self._create_optimized_prompt(history, optimized_context, question)

        cache_key = f"{question[:50]}_{hash(optimized_context[:100])}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            sentences = re.split(r'(?<=[.!?]) +', cached_response)
            for s in sentences:
                if s.strip():
                    yield s + " "
                    time.sleep(0.05)
            return

        try:
            if hasattr(self.llm, 'stream'):
                stream = self.llm.stream(prompt)
                buffer = ""
                full_response = ""
                for chunk in stream:
                    content = chunk.content if hasattr(chunk, 'content') else chunk
                    if not content:
                        continue
                    buffer += content
                    full_response += content
                    while re.search(r'(?<=[.!?\n]) +', buffer):
                        sentences = re.split(r'(?<=[.!?\n]) +', buffer, maxsplit=1)
                        yield sentences[0]
                        buffer = sentences[1] if len(sentences) > 1 else ""
                if buffer:
                    yield buffer
                self.response_cache[cache_key] = full_response
            else:
                raw_answer = self.llm.invoke(prompt)
                response = raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
                self.response_cache[cache_key] = response
                sentences = re.split(r'(?<=[.!?]) +', response)
                for s in sentences:
                    if s.strip():
                        yield s + " "
                        time.sleep(0.05)
        except Exception:
            logger.error("LLM streaming call failed", exc_info=True)
            fallback_response = self._generate_fallback_response(question, context)
            sentences = re.split(r'(?<=[.!?]) +', fallback_response)
            for s in sentences:
                if s.strip():
                    yield s + " "
                    time.sleep(0.05)

    def generate_followups(self, followup_prompt: str, num: int = 5) -> List[str]:
        """Generate follow-up questions based on the given follow-up prompt."""
        logger.debug("Generating follow-up questions for prompt: %s", followup_prompt)
        try:
            response = self.llm.invoke(followup_prompt)
            if isinstance(response, AIMessage):
                followups = response.content.split("\n")
            else:
                followups = str(response).split("\n")

            # Filter out duplicates, overly verbose questions, remove markdown formatting, and strip serial numbers
            unique_followups = list(set(
                re.sub(r"^\d+\.\s*", "", re.sub(r"\*\*|\*", "", followup.strip()))  # Remove serial numbers and markdown symbols
                for followup in followups
                if followup.strip() and not followup.startswith("-") and len(followup.strip()) < 200
            ))
            return unique_followups[:num]
        except Exception as e:
            logger.error("Failed to generate follow-up questions: %s", e)
            return ["Follow-up generation failed. Please try again."]

    def _retrieve_context(self, query: str, site: str) -> str:
        from backend.chat_logic import _maybe_expand_queries, _dedupe_chunks
        from backend.retriever import retriever

        variant_queries = _maybe_expand_queries(query)
        with ThreadPoolExecutor(max_workers=4) as executor:
            all_docs = []
            for q in variant_queries:
                docs = list(executor.map(retriever.get_relevant_documents, [q]))[0]
                all_docs.extend(docs)
        unique_texts = _dedupe_chunks(all_docs)
        MAX_CHUNKS = 4
        context_chunks = []
        for i, (text, meta) in enumerate(unique_texts[:MAX_CHUNKS]):
            context_chunks.append(text)
        context_text = "\n\n---\n\n".join(context_chunks)
        if not context_text.strip():
            context_text = self._fallback_web_search(query, site)
        return context_text

    def _format_history(self, chat_history: list) -> str:
        return "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )

    @lru_cache(maxsize=1)
    def _count_template_tokens(self) -> int:
        template = """
You are a knowledgeable and thorough assistant providing comprehensive information.
Your goal is to give detailed, well-structured answers that fully address the user's question.
Conversation so far:
{history}
Relevant context from the knowledge base:
{context}
User's latest question:
{question}
Instructions:
1. Provide a comprehensive answer that thoroughly addresses all aspects of the question
2. Include specific details, examples, and explanations where appropriate
3. Structure your response with clear sections using markdown formatting
4. Use **bold** for key terms and concepts
5. When relevant, include bullet points or numbered lists to organize information
6. If the context contains multiple relevant pieces of information, synthesize them into a cohesive response
7. If context is limited or partial, still provide the most complete answer possible using your general knowledge
8. Do NOT be overly brief - aim for thoroughness and completeness
9. Include relevant background information that helps understand the topic
Format your response with:
- A clear introductory paragraph
- Well-organized body sections with appropriate headings
- A brief conclusion when appropriate
"""
        return self.context_optimizer.count_tokens_cached(template)

    def _create_optimized_prompt(self, history: str, context: str, question: str) -> str:
        length_rule = (
            "Provide a comprehensive, detailed answer using the context and "
            "examples where possible. Expand on key points thoroughly. "
            "Do not restrict the output length."
        )
        prompt = f"""
You are a helpful assistant restricted to answering only with information from the provided context or the relevant website.
If the user's question is outside the website’s domain, politely decline or use fallback web search context.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

Instructions:
1. {length_rule}
2. Use **bold** for key terms and short markdown lists if needed.
3. If the context contains multiple relevant pieces of information, synthesize them into a cohesive response.
4. Do not fabricate information outside the knowledge base or site scope.
5. If the query is irrelevant (e.g., medical diagnostics, treatments), politely respond that you cannot answer.
6. If context is insufficient but query seems relevant, fall back to the site-specific web search content.
"""
        return prompt.strip()

    def _fallback_web_search(self, query: str, site: str) -> str:
        try:
            from backend.search_client import search_site
            from crawler.scraper import scrape_url
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
        return f"""
I apologize, but I'm experiencing technical difficulties. Based on the available information, here's what I can tell you about your question "{question}":
**Available Context:**
{context[:500]}...
**Recommendation:**
Please try rephrasing your question or contact support for more detailed assistance.
Would you like me to try a different approach to answer your question?
"""
