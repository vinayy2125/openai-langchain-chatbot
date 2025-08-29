"""
Chatbot Optimization Service

This service provides optimized chatbot response generation with:
- Context length management
- Performance optimization
- Detailed response generation
- Robust fallback handling
"""

import tiktoken
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache
from langchain.schema import AIMessage


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
    Main chatbot service that provides optimized response generation.
    """
    
    def __init__(self, llm, model: str = "gpt-3.5-turbo"):
        self.llm = llm
        self.model = model
        self.context_optimizer = ContextOptimizer(model)
        self.response_cache = {}
        
    def get_detailed_response(self, query: str, chat_history: list, site: str = "ditstek.com") -> Tuple[str, bool]:
        """
        Get detailed response with all optimizations.
        
        Args:
            query: User's question
            chat_history: Conversation history
            site: Website for fallback search
            
        Returns:
            Tuple of (response_text, success_flag)
        """
        start_time = time.time()
        
        try:
            # 1. Retrieve and process context
            context = self._retrieve_context(query, site)
            
            # 2. Format history
            history_text = self._format_history(chat_history)
            
            # 3. Generate detailed response
            response = self._generate_response(context, history_text, query)
            
            processing_time = time.time() - start_time
            print(f"[DEBUG] Total processing time: {processing_time:.2f}s")
            
            return response, True
            
        except Exception as e:
            print(f"[ERROR] Chatbot failed: {e}")
            error_msg = "I apologize, but I'm experiencing technical difficulties. Please try again later."
            return error_msg, False
    
    def _retrieve_context(self, query: str, site: str) -> str:
        """Retrieve and optimize context"""
        # Import here to avoid circular imports
        from  backend.chat_logic import _maybe_expand_queries, _dedupe_chunks
        from backend.retriever import retriever  # Adjust import path as needed
        
        # Get variant queries
        variant_queries = _maybe_expand_queries(query)
        
        # Retrieve documents in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            all_docs = []
            for q in variant_queries:
                docs = list(executor.map(retriever.get_relevant_documents, [q]))[0]
                all_docs.extend(docs)
        
        # De-duplicate
        unique_texts = _dedupe_chunks(all_docs)
        
        # Format context with source labels
        MAX_CHUNKS = 12
        context_text = "\n\n---\n\n".join([
            f"Source {i+1}:\n{chunk}" 
            for i, chunk in enumerate(unique_texts[:MAX_CHUNKS])
        ])
        
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
    
    def _generate_response(self, context: str, history: str, question: str) -> str:
        """Generate response with optimized context"""
        # Get template token count
        template_tokens = self._count_template_tokens()
        
        # Optimize context
        optimized_context, stats = self.context_optimizer.optimize_context(
            context, question, history, template_tokens
        )
        
        # Create optimized prompt
        prompt = self._create_optimized_prompt(history, optimized_context, question)
        
        # Check cache first
        cache_key = f"{question[:50]}_{hash(optimized_context[:100])}"
        if cache_key in self.response_cache:
            print("[DEBUG] Using cached response")
            return self.response_cache[cache_key]
        
        # Generate response
        try:
            raw_answer = self.llm.invoke(prompt)
            response = raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
            
            # Cache successful responses
            self.response_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return self._generate_fallback_response(question, optimized_context)
    
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
    
    def _create_optimized_prompt(self, history: str, context: str, question: str) -> str:
        """Create optimized prompt"""
        return f"""
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