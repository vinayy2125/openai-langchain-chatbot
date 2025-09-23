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
from langchain.schema import AIMessage
import re
import time
from langchain.schema import AIMessage
import random
from datetime import datetime

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
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.model_limits = {
            "gpt-4o": 128000
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
        self.query_llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # Dedicated model for query processing
            temperature=0.3,  # Lower temperature for more focused key generation
            max_tokens=150,   # Limit tokens for key generation
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        logger.info("Initialized OptimizedChatbot with GPT-4o-mini for query processing")
        # Ordered discovery categories for requirement elicitation (10 criteria)
        self.requirement_categories = [
            {
                'key': 'goal',
                'name': 'Project Goal / Primary Objective',
                'question': 'What is the primary goal or outcome you want to achieve?',
                'patterns': ['goal', 'objective', 'aim', 'purpose']
            },
            {
                'key': 'users',
                'name': 'Target Users / Audience',
                'question': 'Who are the primary users or audience for this solution?',
                'patterns': ['user', 'audience', 'customer', 'client', 'end user']
            },
            {
                'key': 'pain_points',
                'name': 'Pain Points / Challenges',
                'question': 'What key pain points or challenges are you trying to solve?',
                'patterns': ['pain', 'challenge', 'problem', 'issue', 'bottleneck']
            },
            {
                'key': 'features',
                'name': 'Desired Features / Functionality',
                'question': 'What core features or functionality do you definitely need?',
                'patterns': ['feature', 'functionality', 'module', 'capability']
            },
            {
                'key': 'success_metrics',
                'name': 'Success Metrics / KPIs',
                'question': 'How will success be measured (KPIs or outcomes)?',
                'patterns': ['kpi', 'success', 'metric', 'measure', 'roi']
            },
            {
                'key': 'constraints',
                'name': 'Budget / Resource Constraints',
                'question': 'Do you have budget or resource constraints we should respect?',
                'patterns': ['budget', 'cost', 'constraint', 'resource', 'limit']
            },
            {
                'key': 'timeline',
                'name': 'Timeline / Urgency',
                'question': 'What is the desired timeline or deadline?',
                'patterns': ['timeline', 'deadline', 'schedule', 'date', 'milestone']
            },
            {
                'key': 'tech_stack',
                'name': 'Technology / Platform Preferences',
                'question': 'Any preferred technologies, platforms, or tools?',
                'patterns': ['tech', 'technology', 'stack', 'platform', 'framework']
            },
            {
                'key': 'integrations',
                'name': 'Data / Integrations',
                'question': 'What external systems or data sources need integration?',
                'patterns': ['integration', 'api', 'data source', 'crm', 'erp']
            },
            {
                'key': 'compliance',
                'name': 'Security / Compliance / Privacy',
                'question': 'Are there security, compliance, or privacy requirements?',
                'patterns': ['security', 'privacy', 'compliance', 'gdpr', 'hipaa', 'pci']
            }
        ]
        # Track collected category answers per session
        self.collected_requirements: dict[str, dict] = {}

        # Keep requirement_categories as reference topics but don't rigidly follow them
        self.requirement_categories = [
            {
                'key': 'goal',
                'name': 'Project Goal / Primary Objective',
                'question': 'What is the primary goal or outcome you want to achieve?',
                'patterns': ['goal', 'objective', 'aim', 'purpose']
            },
            {
                'key': 'users',
                'name': 'Target Users / Audience',
                'question': 'Who are the primary users or audience for this solution?',
                'patterns': ['user', 'audience', 'customer', 'client', 'end user']
            },
            {
                'key': 'pain_points',
                'name': 'Pain Points / Challenges',
                'question': 'What key pain points or challenges are you trying to solve?',
                'patterns': ['pain', 'challenge', 'problem', 'issue', 'bottleneck']
            },
            {
                'key': 'features',
                'name': 'Desired Features / Functionality',
                'question': 'What core features or functionality do you definitely need?',
                'patterns': ['feature', 'functionality', 'module', 'capability']
            },
            {
                'key': 'success_metrics',
                'name': 'Success Metrics / KPIs',
                'question': 'How will success be measured (KPIs or outcomes)?',
                'patterns': ['kpi', 'success', 'metric', 'measure', 'roi']
            },
            {
                'key': 'constraints',
                'name': 'Budget / Resource Constraints',
                'question': 'Do you have budget or resource constraints we should respect?',
                'patterns': ['budget', 'cost', 'constraint', 'resource', 'limit']
            },
            {
                'key': 'timeline',
                'name': 'Timeline / Urgency',
                'question': 'What is the desired timeline or deadline?',
                'patterns': ['timeline', 'deadline', 'schedule', 'date', 'milestone']
            },
            {
                'key': 'tech_stack',
                'name': 'Technology / Platform Preferences',
                'question': 'Any preferred technologies, platforms, or tools?',
                'patterns': ['tech', 'technology', 'stack', 'platform', 'framework']
            },
            {
                'key': 'integrations',
                'name': 'Data / Integrations',
                'question': 'What external systems or data sources need integration?',
                'patterns': ['integration', 'api', 'data source', 'crm', 'erp']
            },
            {
                'key': 'compliance',
                'name': 'Security / Compliance / Privacy',
                'question': 'Are there security, compliance, or privacy requirements?',
                'patterns': ['security', 'privacy', 'compliance', 'gdpr', 'hipaa', 'pci']
            }
        ]
        
        # Track conversation state differently - more flexible
        self.conversation_state = {}  # session_id -> state object
        
    def _init_conversation_state(self, session_id: str):
        """Initialize a flexible conversation state tracker."""
        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {
                'topics_covered': set(),        # Topics we've discussed 
                'topics_to_explore': set(),     # Dynamically discovered topics to ask about
                'user_context': {},             # Key insights about user/project
                'follow_up_strategy': 'explore',  # explore, deepen, clarify, challenge, summarize
                'follow_up_count': 0,           # How many follow-ups we've asked
                'last_generated': None          # Timestamp of last generation
            }
        return self.conversation_state[session_id]
        
    def reset_follow_up_count(self, session_id: str):
        """Reset the follow-up count for a session, useful when switching prompts."""
        if session_id in self.conversation_state:
            self.conversation_state[session_id]['follow_up_count'] = 0
            return True
        return False

    # ---------------- Requirement Collection Helpers -----------------
    def _init_requirement_state(self, session_id: str):
        if session_id not in self.collected_requirements:
            self.collected_requirements[session_id] = {
                'answers': {},   # key -> {'question':..., 'answer':...}
                'asked': set()   # keys already asked
            }

    def record_user_message(self, session_id: str, content: str):
        """Attempt to associate latest user reply with the most recently asked unanswered category."""
        self._init_requirement_state(session_id)
        state = self.collected_requirements[session_id]
        # Find last assistant category question not yet answered
        history = self.get_conversation_history(session_id)
        last_category_key = None
        for msg in reversed(history):
            if msg['role'] == 'assistant' and msg.get('meta_category_key'):
                key = msg['meta_category_key']
                if key not in state['answers']:
                    last_category_key = key
                    break
        if last_category_key:
            state['answers'][last_category_key] = {
                'question': next(c['question'] for c in self.requirement_categories if c['key']==last_category_key),
                'answer': content.strip()
            }

    def _next_missing_category(self, session_id: str) -> dict | None:
        self._init_requirement_state(session_id)
        state = self.collected_requirements[session_id]
        for cat in self.requirement_categories:
            if cat['key'] not in state['answers'] and cat['key'] not in state['asked']:
                return cat
        # If all asked but some unanswered (user skipped), re-ask first unanswered
        for cat in self.requirement_categories:
            if cat['key'] not in state['answers']:
                return cat
        return None

    def _synthesize_requirement_summary(self, session_id: str) -> tuple[str, list[str]]:
        """Return (markdown_summary, missing_keys)."""
        self._init_requirement_state(session_id)
        state = self.collected_requirements[session_id]
        lines = []
        missing = []
        for cat in self.requirement_categories:
            key = cat['key']
            if key in state['answers']:
                ans = state['answers'][key]['answer'] or 'Not provided'
                lines.append(f"- **{cat['name']}:** {ans}")
            else:
                lines.append(f"- **{cat['name']}:** _pending_")
                missing.append(key)
        return "\n".join(lines), missing

    def add_follow_up(self, session_id: str, follow_up: str) -> None:
        """Add a follow-up question for a session."""
        if session_id not in self.follow_ups:
            self.follow_ups[session_id] = []
        self.follow_ups[session_id].append(follow_up)

    # --- Follow-up Streaming Support -------------------------------------------------
    def _extract_requirement_answers_from_history(self, history: list[dict]) -> dict:
        """Reconstruct requirement answers from conversation history without relying on stored state.
        Strategy:
          For each assistant message matching a requirement question, take the next user message (before next assistant) as the answer.
        Returns: key -> {'question': str, 'answer': str}
        """
        q_lookup = {c['question']: c for c in self.requirement_categories}
        answers: dict[str, dict] = {}
        for i, msg in enumerate(history):
            if msg.get('role') == 'assistant':
                content = (msg.get('content') or '').strip()
                if content in q_lookup:
                    cat = q_lookup[content]
                    # find next user response
                    answer_text = ''
                    for j in range(i+1, len(history)):
                        nm = history[j]
                        if nm.get('role') == 'assistant':
                            break  # unanswered / skipped
                        if nm.get('role') == 'user':
                            answer_text = (nm.get('content') or '').strip()
                            break
                    if answer_text:
                        answers[cat['key']] = {'question': content, 'answer': answer_text}
        return answers

    def stream_follow_up_generation(
            self,
            conversation_history: list[dict],
            latest_query: str,
            prompt_context: str,
            combined: bool = False,
            followup_count: int = 2
        ):
        """
        Generate follow-up questions or combined answer+follow-ups.
        
        Key changes:
        - Yields raw text chunks only (no `data:` or extra JSON inside).
        - Follow-ups + suggestions handled via prompt.
        - Context-switch options included when unrelated query detected.
        """

        history = conversation_history or []
        session_id = next((msg.get('session_id') for msg in history 
                        if msg.get('session_id')), 'generic')

        # Initialize or get conversation state
        state = self._init_conversation_state(session_id)

        # Build transcript (last 10 messages)
        transcript = "\n".join([
            f"{'USER' if msg.get('role') == 'user' else 'ASSISTANT'}: {msg.get('content', '')}"
            for msg in history[-10:]
        ])

        category_names = ", ".join([cat['name'] for cat in self.requirement_categories])

        if combined:
            # Prompt for answer + follow-ups + suggestions
            prompt = f"""
    You are an expert requirements consultant having a conversation with a client.

    ### Transcript so far:
    {transcript}

    ### Initial context:
    {prompt_context}

    ### Latest user message:
    {latest_query}

    ### Your task:
    1. **Consider the entire conversation context, not just the latest message:**
       - Provide a helpful response (around 100 words) that addresses the latest message while maintaining continuity.
       - Generate {followup_count} natural follow-up questions that help explore different aspects of the topic.
       - If the conversation has shifted, acknowledge the shift and provide options that either:
         a) Connect the new direction back to the original context.
         b) Continue exploring the new direction if it seems more relevant to the user.

    2. **Follow-up questions:**
       - Each follow-up should include 1-2 suggested answers to help guide the conversation.

    3. **Suggestions:**
       - Include a "suggestions" field with practical next steps that consider the full conversation history.

    4. **Exploration areas:**
       - When appropriate, explore these areas: {category_names}

    ### STRICT OUTPUT FORMAT (JSON only):
    {{
    "answer": "<helpful response that maintains conversation continuity>",
    "follow_ups": [
        {{
        "question": "<thoughtful follow-up question based on conversation context>",
        "options": ["<option1>", "<option2>"]
        }}
    ],
    "suggestions": ["<suggestion1 based on full context>", "<suggestion2 based on full context>"]
    }}
    """
        else:
            # Prompt for follow-ups only
            prompt = f"""
    You are an expert requirements consultant having a conversation with a client.

    ### Recent conversation transcript:
    {transcript}

    ### Initial context:
    {prompt_context}

    ### Latest user message:
    {latest_query}

    ### Your task:
    1. **Consider the entire conversation history, not just the latest message:**
       - Generate ONE natural follow-up question that helps advance the conversation toward a more complete understanding.
       - Focus on exploring details that haven't been discussed yet but are relevant to the overall project/topic.
       - If the conversation seems to have changed topic, offer a question that either:
         a) Bridges the new topic back to the original context in a natural way.
         b) Acknowledges the new direction and helps explore it properly.

    2. **Follow-up question:**
       - Each follow-up may include 1-2 suggested answer options (using hyphens `-`) to help guide the user.

    3. **Question requirements:**
       - Keep the question concise and engaging. No intros or explanations.
       - Ensure your question feels like a natural continuation of the conversation, not an abrupt change.
    """

        # Track generation state
        state['follow_up_count'] += 1
        state['last_generated'] = datetime.now()

        try:
            # Streaming output from LLM
            if hasattr(self.llm, "stream"):
                for chunk in self.llm.stream(prompt):
                    content = getattr(chunk, 'content', str(chunk))
                    if content:
                        # YIELD RAW CHUNK ONLY
                        yield content
            else:
                # Fallback for non-streaming LLMs
                response = self.llm.invoke(prompt)
                text = getattr(response, 'content', str(response))
                for line in text.split('\n'):
                    yield line

        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}", exc_info=True)
            # Simple fallback
            fallback = "Could you tell me more about your goals for this project?\n- Business growth\n- Process improvement"
            for line in fallback.split('\n'):
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
        
    def add_to_conversation_history(self, session_id: str, role: str, content: str) -> None:
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
        
    def get_detailed_response(self, query: str, chat_history: list, site: str = "ditstek.com", stream: bool = True) -> Generator:
        """
        Enhanced response generation with explicit flow:
        1. Query → GPT-4o-mini → Search Keys
        2. Search Keys → Vector DB → Retrieved Context  
        3. Query + Context + History + Instructions → Main LLM → Response
        """
        try:
            # STEP 1: Log the enhanced flow start
            logger.info(f"Starting enhanced response generation for query: {query[:100]}...")
            
            # STEP 2: Retrieve context using enhanced key generation
            context = self._retrieve_context(query, site)
            
            # STEP 3: Format chat history
            history = self._format_history(chat_history)
            
            # STEP 4: Log the complete flow for debugging
            logger.info(f"Enhanced Flow Summary:")
            logger.info(f"- Original Query: {query[:100]}")
            logger.info(f"- Context Retrieved: {len(context)} characters")
            logger.info(f"- Chat History: {len(chat_history)} messages")
            logger.info(f"- Model for keys: gpt-4o-mini")
            logger.info(f"- Model for response: {getattr(self.llm, 'model_name', 'Unknown')}")
            
            # Debug logs for verification
            logger.debug(">>> Enhanced Query Processing: %s", query)
            logger.debug(">>> Enhanced Context Retrieved: %s", context[:500] + "..." if len(context) > 500 else context)
            logger.debug(">>> Enhanced Chat History: %s", history)
            
            # STEP 5: Generate response using enhanced context
            return self._generate_response_stream(query, context, history)
            
        except Exception as e:
            logger.error(f"Enhanced response generation failed: {e}")
            return self._fallback_response_stream(query)
    
    def _fallback_response_stream(self, query: str) -> Generator[str, None, None]:
        """Fallback response when enhanced flow fails"""
        fallback_message = f"I understand you're asking about: {query}. Let me provide a general response based on my knowledge."
        yield fallback_message

    def _clean_response_formatting(self, text: str) -> str:
        """Clean response formatting to ensure consistency and remove unwanted elements"""
        if not text:
            return ""
        
        # Remove unwanted headers that appear at the beginning
        unwanted_headers = [
            r'^#{1,6}\s*Quick Overview.*?\n',
            r'^#{1,6}\s*Overview of.*?\n',
            r'^#{1,6}\s*About.*?\n',
            r'^#{1,6}\s*Introduction.*?\n',
            r'^#{1,6}\s*Implementation\s+Approach.*?\n',
            r'^#{1,6}\s*Next\s+Steps.*?\n'
        ]
        
        for pattern in unwanted_headers:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Fix spacing issues while preserving markdown formatting
        # Remove multiple spaces but preserve markdown formatting
        text = re.sub(r'(?<!\*) {2,}(?!\*)', ' ', text)  # Multiple spaces to single, preserve **bold**
        text = re.sub(r' +\n', '\n', text)  # Remove spaces before newlines
        text = re.sub(r'\n +(?![#*-])', '\n', text)  # Remove spaces after newlines (except markdown)
        
        # Ensure single space after punctuation
        text = re.sub(r'([.!?:;,])\s+', r'\1 ', text)
        
        # Clean up header formatting
        text = re.sub(r'(\n)(#{1,3})\s+', r'\1\2 ', text)  # Ensure single space after headers
        
        # Clean up bullet points
        text = re.sub(r'(\n)([-*•])\s+', r'\1\2 ', text)  # Ensure single space after bullets
        
        # Clean up numbered lists
        text = re.sub(r'(\n)(\d+\.)\s+', r'\1\2 ', text)  # Ensure single space after numbers
        
        # Remove multiple newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure proper spacing around bold text
        text = re.sub(r'\*\*\s+', '**', text)  # Remove space after opening **
        text = re.sub(r'\s+\*\*', '**', text)  # Remove space before closing **
        
        return text.strip()

    def _generate_response_stream(self, question: str, context: str, history: str) -> Generator[str, None, None]:
        logger.debug("Entered _generate_response_stream")

        # Format context with separators and metadata
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

        # Log the formatted context
        logger.debug("Formatted Context: %s", context)

        template_tokens = self._count_template_tokens()
        optimized_context, stats = self.context_optimizer.optimize_context(context, question, history, template_tokens)
        logger.debug("Optimization stats: %s", stats)

        prompt = self._create_optimized_prompt(history, optimized_context, question)
        logger.debug("Final Prompt: %s", prompt)

        cache_key = f"{question[:50]}_{hash(optimized_context[:100])}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            yield cached_response
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
                    
                    full_response += content
                    buffer += content
                    
                    # Process buffer for word-level streaming with proper formatting preservation
                    while True:
                        # Check for complete words to stream
                        # Look for word boundaries while preserving markdown formatting
                        word_match = re.search(r'(\S+(?:\s+|$))', buffer)
                        if word_match:
                            word = word_match.group(1)
                            buffer = buffer[word_match.end():]
                            
                            # Clean up spacing but preserve markdown formatting
                            if word.strip():
                                # Handle newlines and paragraph breaks properly
                                if '\n\n' in word:
                                    # This is a paragraph break
                                    parts = word.split('\n\n')
                                    for i, part in enumerate(parts):
                                        if part.strip():
                                            yield part.strip() + ' '
                                        if i < len(parts) - 1:  # Not the last part
                                            yield '\n\n'
                                elif '\n' in word and not word.strip().startswith('#'):
                                    # This is a line break (not a header)
                                    parts = word.split('\n')
                                    for i, part in enumerate(parts):
                                        if part.strip():
                                            yield part.strip() + ' '
                                        if i < len(parts) - 1:  # Not the last part
                                            yield '\n'
                                else:
                                    # Regular word handling
                                    if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                                        cleaned_word = word.strip() + ' '
                                    elif word.endswith('\n'):
                                        cleaned_word = word.strip() + '\n'
                                    else:
                                        cleaned_word = word.strip() + ' '
                                    
                                    # Handle special markdown cases
                                    if '**' in cleaned_word or '###' in cleaned_word or cleaned_word.startswith('-') or cleaned_word.startswith('*'):
                                        # Preserve formatting for markdown elements
                                        yield cleaned_word
                                    else:
                                        # Regular word with proper spacing
                                        yield cleaned_word
                        else:
                            # No more complete words in buffer
                            break
                
                # Send remaining content word by word with proper cleanup
                if buffer.strip():
                    # Apply comprehensive response cleaning first
                    buffer = self._clean_response_formatting(buffer)
                    
                    # Handle remaining content with proper line break preservation
                    # Split by double newlines first (paragraph breaks)
                    paragraphs = buffer.split('\n\n')
                    for p_idx, paragraph in enumerate(paragraphs):
                        if paragraph.strip():
                            # Split paragraph into lines
                            lines = paragraph.split('\n')
                            for l_idx, line in enumerate(lines):
                                if line.strip():
                                    # Split line into words
                                    words = line.split()
                                    for word in words:
                                        if word.strip():
                                            # Add proper spacing after each word
                                            if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                                                yield word + ' '
                                            else:
                                                yield word + ' '
                                # Add line break after each line (except last)
                                if l_idx < len(lines) - 1:
                                    yield '\n'
                        # Add paragraph break after each paragraph (except last)
                        if p_idx < len(paragraphs) - 1:
                            yield '\n\n'
                    
                # Cache the cleaned response
                self.response_cache[cache_key] = self._clean_response_formatting(full_response)
            else:
                raw_answer = self.llm.invoke(prompt)
                response = raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
                # Apply response cleaning
                response = self._clean_response_formatting(response)
                self.response_cache[cache_key] = response
                
                # Stream the response word by word with proper line break handling
                paragraphs = response.split('\n\n')
                for p_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        lines = paragraph.split('\n')
                        for l_idx, line in enumerate(lines):
                            if line.strip():
                                words = line.split()
                                for word in words:
                                    if word.strip():
                                        # Add proper spacing after each word
                                        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                                            yield word + ' '
                                        else:
                                            yield word + ' '
                            # Add line break after each line (except last)
                            if l_idx < len(lines) - 1:
                                yield '\n'
                    # Add paragraph break after each paragraph (except last)
                    if p_idx < len(paragraphs) - 1:
                        yield '\n\n'
        except Exception:
            logger.error("LLM streaming call failed", exc_info=True)
            fallback_response = self._generate_fallback_response(question, context)
            yield fallback_response

    def _would_break_markdown(self, text: str) -> bool:
        """Check if breaking at this point would damage Markdown formatting"""
        # Count unclosed bold markers
        bold_count = text.count('**')
        if bold_count % 2 != 0:
            return True
        
        # Check if we're in the middle of a header
        lines = text.split('\n')
        if lines and lines[-1].strip().startswith('#') and not lines[-1].strip().endswith(' '):
            return True
        
        # Check if we're breaking a word that might be part of markdown
        if text.endswith('**') or text.endswith('*') or text.endswith('#'):
            return True
        
        # Check for incomplete list items
        if text.strip().endswith('-') and not text.strip().endswith(' -'):
            return True
        
        # Check for incomplete numbered lists
        if re.search(r'\d+\.$', text.strip()):
            return True
            
        return False

    def _generate_search_keys(self, query: str) -> List[str]:
        """Generate search keys using GPT-4o-mini for better vector DB retrieval"""
        
        key_generation_prompt = f"""Break down this user query into 3-5 specific search keys/terms that would help find relevant information in a knowledge base about software development, web development, and technology services.

User Query: {query}

Instructions:
1. Extract the main technical concepts, technologies, or services mentioned
2. Include related/synonym terms that might be used in documentation
3. Focus on actionable keywords that would match knowledge base content
4. Avoid overly broad terms like "help" or "information"
5. Include both specific and general terms related to the query

Return ONLY the search keys, one per line, without numbers or bullets.

Example for "How to build a React app with authentication":
React application development
authentication implementation
user login system
frontend security
React authentication libraries"""

        try:
            # Use GPT-4o-mini for key generation
            messages = [
                {"role": "system", "content": "You are an expert at extracting search keywords from user queries for knowledge base retrieval."},
                {"role": "user", "content": key_generation_prompt}
            ]
            
            response = self.query_llm.invoke(messages)
            search_keys = [key.strip() for key in response.content.split('\n') if key.strip()]
            
            # Fallback to original query if no keys generated
            if not search_keys:
                search_keys = [query]
                
            logger.info(f"Generated {len(search_keys)} search keys from query: {query}")
            return search_keys
            
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            return [query]  # Fallback to original query

    def _retrieve_context(self, query: str, site: str) -> str:
        """Enhanced context retrieval with dedicated key generation"""
        from backend.chat_logic import _maybe_expand_queries, _dedupe_chunks
        from backend.retriever import retriever

        try:
            # STEP 1: Use GPT-4o-mini to generate search keys from query context
            search_keys = self._generate_search_keys(query)
            logger.info(f"Using {len(search_keys)} search keys for context retrieval")
            
            # STEP 2: Perform vector search with generated keys
            with ThreadPoolExecutor(max_workers=4) as executor:
                all_docs = []
                for key in search_keys:
                    docs = list(executor.map(retriever.get_relevant_documents, [key]))[0]
                    all_docs.extend(docs)
                
                # Also search with original query for completeness
                original_docs = list(executor.map(retriever.get_relevant_documents, [query]))[0]
                all_docs.extend(original_docs)
            
            # STEP 3: Deduplicate and limit context
            unique_texts = _dedupe_chunks(all_docs)
            MAX_CHUNKS = 8  # Increased from 4 for better context
            context_chunks = []
            
            for i, (text, meta) in enumerate(unique_texts[:MAX_CHUNKS]):
                source = meta.get('source', 'Unknown') if meta else 'Unknown'
                context_chunks.append(f"Source: {source}\n{text}")
            
            context_text = "\n\n---\n\n".join(context_chunks)
            
            # STEP 4: Fallback if no context found
            if not context_text.strip():
                logger.warning("No context found from vector search, using fallback")
                context_text = self._fallback_web_search(query, site)
            
            logger.info(f"Retrieved context from {len(context_chunks)} documents using enhanced key generation")
            return context_text
            
        except Exception as e:
            logger.error(f"Enhanced context retrieval failed: {e}")
            # Fallback to original method
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
            "Provide comprehensive, well-structured responses with proper Markdown formatting. "
            "Use **bold text extensively** for key terms, technologies, and important concepts. "
            "Use ### headings only when truly needed for organization (avoid generic titles like 'Quick Overview'). "
            "Include specific recommendations with **bold highlights**. "
            "Address all key points discussed in the conversation. "
            "Make responses knowledge-base oriented and informative. "
            "Use natural paragraph breaks for readability. "
            "Emphasize important frameworks, tools, or concepts with **bold**. "
            "Structure content logically with proper spacing. "
            "Use Markdown formatting naturally (lists, bold, italics, etc.). "
            "Ensure consistent single spacing between words. "
        )


        prompt = f"""
You are a helpful assistant providing well-formatted, comprehensive responses using proper Markdown structure with a conversational, engaging tone like ChatGPT.
If the user's question is outside the website’s domain, politely decline or use fallback web search context.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

CONVERSATIONAL TONE REQUIREMENTS:
1. Start naturally and engaging:
   - Use "Absolutely!" or "Great question!" when appropriate
   - Show understanding: "Here's what I can help you with..."
   - Be friendly, professional, and engaging like ChatGPT
   - Use natural conversational flow

2. CRITICAL FORMATTING REQUIREMENTS:
1. {length_rule}
2. Use proper Markdown structure:
   - Use **bold text extensively** for key terms, names, technologies, and important concepts
   - Use ### headings only when truly needed for organization (avoid generic titles like "Quick Overview")
   - Ensure proper single spacing between words - no multiple spaces
   - Use bullet points with **bold** key terms for clarity
   - Proper line breaks between sections

3. Content Guidelines:
   - Use the knowledge base context as your primary information source
   - Be comprehensive but focused (50-150 words)
   - Include specific details and examples with **bold formatting**
   - Maintain professional, conversational tone
   - Address the user's question directly and thoroughly

4. Structure Requirements:
   - Start with the most relevant information immediately
   - Organize content logically with proper spacing
   - Use natural paragraph breaks for readability
   - Include actionable information when relevant to the question

5. Professional Standards:
   - Use engaging, professional tone while staying informative
   - Include relevant context and background from the knowledge base
   - Ensure consistent spacing and formatting throughout
   - Do not fabricate information outside the knowledge base scope

Important: Provide direct, informative responses based on the knowledge base context. Avoid generic introductory headers and focus on delivering valuable information with extensive **bold formatting** for key concepts.
"""
        return prompt.strip()

    def get_flow_debug_info(self, query: str, site: str = "ditstek.com") -> Dict[str, Any]:
        """Get detailed flow information for debugging the enhanced process"""
        try:
            # Generate search keys
            search_keys = self._generate_search_keys(query)
            
            # Get context
            context = self._retrieve_context(query, site)
            
            return {
                "original_query": query,
                "generated_keys": search_keys,
                "context_length": len(context),
                "context_preview": context[:300] + "..." if len(context) > 300 else context,
                "model_used_for_keys": "gpt-4o-mini",
                "model_used_for_response": getattr(self.llm, 'model_name', 'Unknown'),
                "enhancement_status": "Enhanced flow active with GPT-4o-mini key generation",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "enhancement_status": "Enhanced flow failed, using fallback",
                "timestamp": datetime.now().isoformat()
            }

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

# Follow-ups are now handled internally by the OptimizedChatbot class
