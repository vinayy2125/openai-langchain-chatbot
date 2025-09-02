"""
Chatbot Optimization Service with Streaming Support
This service provides optimized chatbot response generation with:
- Context length management
- Performance optimization
- Detailed response generation
- Robust fallback handling
- Streaming response support
"""
import tiktoken
from typing import List, Tuple, Dict, Generator, Union
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache
from langchain.schema import AIMessage
import queue
import threading

class ContextOptimizer:
    """
    Handles context optimization to fit within model token limits
    while maximizing relevance and information retention.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        self.context_limit = self.model_limits.get(model, 4096)
    
    @lru_cache(maxsize=1000)
    def count_tokens_cached(self, text: str) -> int:
        """Cached token counting for better performance"""
        return len(self.encoding.encode(text))
    
    def score_chunk_relevance(self, chunk: str, question: str) -> float:
        """Score chunk relevance based on keyword overlap"""
        question_words = set(question.lower().split())
        chunk_words = set(chunk.lower().split())
        
        overlap = len(question_words.intersection(chunk_words))
        total_question_words = len(question_words)
        
        if total_question_words == 0:
            return 0.0
        
        relevance_score = overlap / total_question_words
        
        # Bonus for key question terms
        key_terms = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        for term in key_terms:
            if term in question.lower() and term in chunk.lower():
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def prioritize_chunks(self, chunks: List[str], question: str, max_chunks: int = 8) -> List[str]:
        """Prioritize chunks by relevance"""
        if not chunks:
            return []
        
        # Score chunks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            scores = list(executor.map(
                lambda chunk: self.score_chunk_relevance(chunk, question), 
                chunks
            ))
        
        # Sort by relevance
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top chunks
        prioritized_chunks = [chunk for chunk, score in chunk_scores[:max_chunks]]
        
        print(f"[DEBUG] Prioritized {len(chunks)} chunks to {len(prioritized_chunks)} most relevant")
        return prioritized_chunks
    
    def optimize_context(self, context: str, question: str, history: str, template_tokens: int) -> Tuple[str, Dict]:
        """Optimize context to fit within token limits"""
        # Calculate available tokens
        question_tokens = self.count_tokens_cached(question)
        history_tokens = self.count_tokens_cached(history)
        
        # Reserve tokens for response and safety
        response_reservation = 1000
        safety_buffer = 500
        
        available_for_context = (
            self.context_limit - template_tokens - question_tokens - 
            history_tokens - response_reservation - safety_buffer
        )
        
        print(f"[DEBUG] Available tokens for context: {available_for_context}")
        
        # Ensure context is a string before splitting
        if isinstance(context, list):
            context = "\n\n---\n\n".join(context)
        
        # Split and prioritize chunks
        chunks = context.split("\n\n---\n\n")
        prioritized_chunks = self.prioritize_chunks(chunks, question)
        
        # Build optimized context
        optimized_context = []
        current_tokens = 0
        
        for chunk in prioritized_chunks:
            chunk_tokens = self.count_tokens_cached(chunk)
            chunk_with_separator = chunk + "\n\n---\n\n"
            separator_tokens = self.count_tokens_cached("\n\n---\n\n")
            
            if current_tokens + chunk_tokens + separator_tokens <= available_for_context:
                optimized_context.append(chunk)
                current_tokens += chunk_tokens + separator_tokens
            else:
                # Try to add partial chunk
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
        
        print(f"[DEBUG] Context optimization stats: {optimization_stats}")
        
        return final_context, optimization_stats
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

class OptimizedChatbot:
    """
    Main chatbot service that provides optimized response generation with streaming support.
    """
    
    def __init__(self, llm, model: str = "gpt-3.5-turbo"):
        self.llm = llm
        self.model = model
        self.context_optimizer = ContextOptimizer(model)
        self.response_cache = {}
        
    def get_detailed_response(self, query: str, chat_history: list, site: str = "ditstek.com", stream: bool = False, detailed: bool = False) -> Union[Tuple[str, bool], Generator]:
        """
        Get detailed response with all optimizations.
        
        Args:
            query: User's question
            chat_history: Conversation history
            site: Website for fallback search
            stream: Whether to return a streaming response
            
        Returns:
            If stream=False: Tuple of (response_text, success_flag)
            If stream=True: Generator that yields response chunks
        """
        # Retrieve context based on the query and site
        context = self._retrieve_context(query, site)
        
        # Format chat history
        history = self._format_history(chat_history)
        
        if stream:
            return self._get_streaming_response(query, context, history, detailed)
        else:
            return self._get_complete_response(query, context, history, detailed), True
    
    def _get_complete_response(self, query: str, context: str, history: str, detailed: bool = False) -> str:
        """Get response in non-streaming mode"""
        return self._generate_response(context, history, query, detailed)
    
    def _get_streaming_response(self, query: str, context: str, history: str, detailed: bool = False) -> Generator:
        """Stream response chunks from the LLM"""
        return self._generate_response_stream(context, history, query, detailed)
    
    def _retrieve_context(self, query: str, site: str) -> str:
        """Retrieve and optimize context with metadata for traceability"""
        # Import here to avoid circular imports
        from backend.chat_logic import _maybe_expand_queries, _dedupe_chunks
        from backend.retriever import retriever  # Adjust import path as needed

        # Get variant queries
        variant_queries = _maybe_expand_queries(query)

        # Retrieve documents in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            all_docs = []
            for q in variant_queries:
                # `get_relevant_documents` returns a list of document objects
                docs = list(executor.map(retriever.get_relevant_documents, [q]))[0]
                all_docs.extend(docs)

        # De-duplicate and preserve metadata
        unique_texts = _dedupe_chunks(all_docs)  # now returns List[Tuple[str, dict]]

        # Format context with source labels
        MAX_CHUNKS = 4
        context_chunks = []
        for i, (text, meta) in enumerate(unique_texts[:MAX_CHUNKS]):
            source_info = meta.get("source", meta.get("url", "N/A"))
            context_chunks.append(f"Source {i+1} ({source_info}):\n{text}")

        context_text = "\n\n---\n\n".join(context_chunks)

        # Fallback to web search if no context
        if not context_text.strip():
            print("[DEBUG] No context from FAISS. Falling back to internet search...")
            context_text = self._fallback_web_search(query, site)

        return context_text
    
    def _format_history(self, chat_history: list) -> str:
        """Format chat history"""
        return "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )
    
    def _generate_response(self, context: Union[str, List[Tuple[str, dict]]], history: str, question: str, detailed: bool = False) -> str:
        """Generate response with optimized context (non-streaming)"""
        template_tokens = self._count_template_tokens()

        # Ensure context is a string
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

        # Optimize context
        optimized_context, stats = self.context_optimizer.optimize_context(
            context, question, history, template_tokens
        )
        print(f"[DEBUG] Optimization stats: {stats}")

        # Ensure optimized_context is a string (after optimization it might still be a list of tuples)
        if isinstance(optimized_context, list):
            context_chunks = []
            for i, chunk in enumerate(optimized_context):
                if isinstance(chunk, tuple):
                    text, meta = chunk
                    source_info = meta.get("source", meta.get("url", "N/A"))
                    context_chunks.append(f"Source {i+1} ({source_info}):\n{text}")
                else:
                    context_chunks.append(str(chunk))
            optimized_context = "\n\n---\n\n".join(context_chunks)

        # Log final optimized context
        print(f"[DEBUG] Final optimized context: {optimized_context[:500]}..." if len(optimized_context) > 500 else f"[DEBUG] Final optimized context: {optimized_context}")

        # Create prompt
        prompt = self._create_optimized_prompt(history, optimized_context, question, detailed)

        # Cache key
        cache_key = f"{question[:50]}_{hash(optimized_context[:100])}_{detailed}"
        if cache_key in self.response_cache:
            print("[DEBUG] Using cached response")
            return self.response_cache[cache_key]

        try:
            raw_answer = self.llm.invoke(prompt)
            response = raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
            self.response_cache[cache_key] = response
            return response
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return self._generate_fallback_response(question, optimized_context)

    def _generate_response_stream(self, context: Union[str, List[Tuple[str, dict]]], history: str, question: str, detailed: bool = False) -> Generator:
        """Generate streaming response with optimized context"""
        template_tokens = self._count_template_tokens()

        # Ensure context is string
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

        # Log final context
        print(f"[DEBUG] Final context for streaming: {context[:500]}..." if len(context) > 500 else f"[DEBUG] Final context for streaming: {context}")

        # Create prompt
        prompt = self._create_optimized_prompt(history, context, question)

        # Cache key
        cache_key = f"{question[:50]}_{hash(context[:100])}"
        if cache_key in self.response_cache:
            print("[DEBUG] Using cached response for streaming")
            # For cached responses, yield in chunks to simulate streaming
            cached_response = self.response_cache[cache_key]
            if isinstance(cached_response, str):
                words = cached_response.split()
                chunk_size = 10
                for i in range(0, len(words), chunk_size):
                    yield ' '.join(words[i:i + chunk_size]) + ' '
                    time.sleep(0.05)  # Small delay for streaming effect
            return

        # Generate streaming response
        try:
            if hasattr(self.llm, 'stream'):
                stream = self.llm.stream(prompt)
                for chunk in stream:
                    content = chunk.content if hasattr(chunk, 'content') else chunk
                    if content:
                        yield content
                        self.response_cache[cache_key] = self.response_cache.get(cache_key, "") + content
            else:
                raw_answer = self.llm.invoke(prompt)
                response = raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
                self.response_cache[cache_key] = response
                words = response.split()
                for i in range(0, len(words), 10):
                    yield ' '.join(words[i:i + 10]) + ' '
                    time.sleep(0.05)
        except Exception as e:
            print(f"[ERROR] LLM streaming call failed: {e}")
            fallback_response = self._generate_fallback_response(question, context)            
            for word in fallback_response.split():
                yield word + ' '
                time.sleep(0.05)
  
    @lru_cache(maxsize=1)
    def _count_template_tokens(self) -> int:
        """Count template tokens"""
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
    
    def _create_optimized_prompt(self, history: str, context: str, question: str, detailed: bool = False) -> str:
        """
        Create optimized prompt for the LLM with concise/detailed control.
        
        Args:
            history: Chat history as string
            context: Optimized context string
            question: User's latest query
            detailed: Whether to generate detailed answer (True) or concise (False)
            
        Returns:
            Prompt string
        """

        # Set length instructions
        if detailed:
            length_rule = "Provide a comprehensive, detailed answer using context and examples."
        else:
            length_rule = "Provide a concise answer within ~600 characters using only relevant context."

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
    3. When relevant, include bullet points or numbered lists to organize information
    4.If the context contains multiple relevant pieces of information, synthesize them into a cohesive response
    5. Do not fabricate information outside the knowledge base or site scope.
    6. If the query is irrelevant (e.g., medical diagnostics, treatments), politely respond that you cannot answer.
    7. If context is insufficient but query seems relevant, fall back to the site-specific web search content.
    """
        return prompt.strip()

    
    def _fallback_web_search(self, query: str, site: str) -> str:
        """Fallback to web search"""
        try:
            # Import here to avoid circular imports
            from backend.search_client import search_site
            from crawler.scraper import scrape_url
            
            search_results = search_site(query, site)
            scraped_texts = []
            
            # Scrape in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for res in search_results:
                    url = res.get("url")
                    if url:
                        future = executor.submit(scrape_url, url)
                        futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result(timeout=10)
                        if result:
                            # Find the original result for this URL
                            for res in search_results:
                                if res.get("url") == url:
                                    title = res.get("title", url)
                                    scraped_texts.append(f"[{title}]({url}): {result}")
                                    break
                    except Exception as e:
                        print(f"[ERROR] Web scraping failed: {e}")
                        continue
            
            return "\n\n".join(scraped_texts[:8])
            
        except Exception as e:
            print(f"[ERROR] Web search failed: {e}")
            return "No additional context available from web search."
    
    def _generate_fallback_response(self, question: str, context: str) -> str:
        """Generate fallback response"""
        return f"""
I apologize, but I'm experiencing technical difficulties. Based on the available information, here's what I can tell you about your question "{question}":
**Available Context:**
{context[:500]}...
**Recommendation:**
Please try rephrasing your question or contact support for more detailed assistance.
Would you like me to try a different approach to answer your question?
"""