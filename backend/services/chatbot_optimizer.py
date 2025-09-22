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
from typing import List, Tuple, Dict, Generator, Union, AsyncGenerator
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
        self.follow_ups = {}
        self.session_data = {}
        self.conversation_history = {}
        # Initialize optimization + caching utilities (previously misplaced)
        self.context_optimizer = ContextOptimizer(model)
        self.response_cache = {}
        self.generated_followups = []  # Store generated follow-ups internally
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

    Transcript so far:
    {transcript}

    Initial context: {prompt_context}
    Latest user message: {latest_query}

    Your task:
    1. Consider the entire conversation context, not just the latest message:
    - Provide a helpful response (around 100 words) that addresses the latest message while maintaining continuity
    - Generate {followup_count} natural follow-up questions that help explore different aspects of the topic
    - If the conversation has shifted, acknowledge the shift and provide options that either:
        a) Connect the new direction back to the original context
        b) Continue exploring the new direction if it seems more relevant to the user

    2. Each follow-up should include 1-2 suggested answers to help guide the conversation.
    3. Include a "suggestions" field with practical next steps that consider the full conversation history.
    4. When appropriate, explore these areas: {category_names}

    STRICT OUTPUT FORMAT (JSON only):
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

    Recent conversation transcript:
    {transcript}

    Initial context: {prompt_context}
    Latest user message: {latest_query}

    Your task:
    1. Consider the entire conversation history, not just the latest message:
    - Generate ONE natural follow-up question that helps advance the conversation toward a more complete understanding
    - Focus on exploring details that haven't been discussed yet but are relevant to the overall project/topic
    - If the conversation seems to have changed topic, offer a question that either:
        a) Bridges the new topic back to the original context in a natural way
        b) Acknowledges the new direction and helps explore it properly

    2. Each follow-up may include 1-2 suggested answer options (using hyphens `-`) to help guide the user.
    3. Keep the question concise and engaging. No intros or explanations.
    4. Ensure your question feels like a natural continuation of the conversation, not an abrupt change.
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



    async def generate_next_follow_up(self, session_id: str):
        """Generate the next follow-up question with streaming."""
        history = self.get_conversation_history(session_id)
        prompt = f"Based on this conversation:\n{self.format_conversation_history(history)}\nWhat should I ask next?"
        
        # Use astream for token-by-token streaming
        async for chunk in self.llm.astream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)


    # async def generate_complete_response(self, session_id: str, query: str) -> str:
    #     """Generate a comprehensive final solution using reconstructed answers from history (stateless)."""
    #     history = self.get_conversation_history(session_id)
    #     answers = self._extract_requirement_answers_from_history(history)
    #     # Build markdown table-like summary
    #     lines = []
    #     missing = []
    #     for cat in self.requirement_categories:
    #         if cat['key'] in answers:
    #             ans = answers[cat['key']]['answer'] or 'Not provided'
    #             lines.append(f"- **{cat['name']}:** {ans}")
    #         else:
    #             lines.append(f"- **{cat['name']}:** _pending_")
    #             missing.append(cat['key'])
    #     summary_md = "\n".join(lines)
    #     base_query = query or next((m['content'] for m in history if m['role']=='user'), '')
    #     synthesis_instructions = (
    #         "You are an expert solutions architect. Using the collected requirements summary below, produce a comprehensive proposal that: \n"
    #         "1. Maps each requirement to specific solution components.\n"
    #         "2. Provides a concise architecture overview (text).\n"
    #         "3. Details recommended features grouped logically.\n"
    #         "4. Lists risks with mitigations.\n"
    #         "5. Highlights assumptions for any pending items (" + str(len(missing)) + ").\n"
    #         "6. Ends with next 3 actionable steps.\n"
    #         "Use only Markdown level 6 headings (######). Keep paragraphs short."
    #     )
    #     prompt = (
    #         f"Conversation transcript (chronological):\n{self.format_conversation_history(history)}\n\n" \
    #         f"Original user goal / latest query:\n{base_query}\n\n" \
    #         f"Collected requirements summary (markdown):\n{summary_md}\n\n" \
    #         f"{synthesis_instructions}\n\nFINAL ANSWER:"
    #     )
    #     try:
    #         response = await self.llm.ainvoke(prompt)
    #         answer = response.content if isinstance(response, AIMessage) else str(response)
    #     except Exception:
    #         logger.error("Final synthesis LLM call failed", exc_info=True)
    #         answer = (
    #             "Model synthesis failed. Here is the structured summary instead:\n\n" + summary_md
    #         )
    #     return answer

    async def generate_suggestions(self, session_id: str) -> List[str]:
        """Generate contextual action suggestions based on the conversation."""
        history = self.get_conversation_history(session_id)
        
        # For early conversations (few messages), suggest discovery actions
        if len(history) < 6:
            return [
                "Schedule a discovery call",
                "Share project documentation",
                "Invite key stakeholders to discussion",
                "Review similar case studies"
            ]
        
        # Extract key topics/entities from conversation
        transcript = '\n'.join([
            f"{msg.get('role','')}: {msg.get('content','')}" 
            for msg in history[-12:] if msg.get('content')
        ])
        
        # Generate contextual suggestions
        prompt = f"""Based on this conversation excerpt, suggest 4 specific next actions the user should take.
Make each suggestion actionable, concrete and specific to their context.
Keep each suggestion under 8 words if possible.

Conversation:
{transcript}

Format: Return exactly 4 suggestions, each on its own line with no numbering or bullets."""
        
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Process into clean suggestions list
            suggestions = []
            for line in content.strip().split('\n'):
                clean_line = re.sub(r'^[-•*\d.)\s]+', '', line).strip()
                if clean_line and len(suggestions) < 4:
                    # Keep suggestions concise
                    words = clean_line.split()
                    if len(words) > 8:
                        clean_line = ' '.join(words[:8])
                    suggestions.append(clean_line)
            
            # Ensure we have 4 suggestions
            while len(suggestions) < 4:
                generic = [
                    "Schedule a discovery workshop",
                    "Share your project timeline",
                    "Define success metrics",
                    "Review proposed solution with team",
                    "Request cost estimate",
                    "Book a follow-up consultation"
                ]
                for g in generic:
                    if g not in suggestions and len(suggestions) < 4:
                        suggestions.append(g)
                        
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return [
                "Schedule a discovery workshop",
                "Share project documentation",
                "Define key success metrics",
                "Review next steps with team"
            ]

    async def generate_followups(self, followup_prompt: str, num: int = 5) -> List[str]:
        """Generate follow-up questions and track them."""
        logger.debug("Generating follow-up questions for prompt: %s", followup_prompt)
        try:
            response = await self.llm.ainvoke(followup_prompt)
            if isinstance(response, AIMessage):
                followups = response.content.split("\n")
            else:
                followups = str(response).split("\n")

            # Filter and track follow-ups
            unique_followups = list(set(
                re.sub(r"^\d+\.\s*", "", followup.strip())
                for followup in followups
                if followup.strip()
            ))

            # Add follow-ups to manager
            for follow_up in unique_followups[:num]:
                self.add_follow_up("session_id_placeholder", follow_up)

            return unique_followups[:num]
        except Exception as e:
            logger.error("Failed to generate follow-up questions: %s", e)
            return ["Follow-up generation failed. Please try again."]
        
    def get_detailed_response(self, query: str, chat_history: list, site: str = "ditstek.com", stream: bool = True) -> Generator:
        context = self._retrieve_context(query, site)
        history = self._format_history(chat_history)
        logger.debug(">>> Query: %s", query)
        logger.debug(">>> Context Retrieved: %s", context)
        logger.debug(">>> Chat History: %s", history)
        return self._generate_response_stream(query, context, history)
    
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
                    
                    # Process buffer to yield complete sections with proper formatting
                    while True:
                        # Look for complete sections (headers with content)
                        header_match = re.search(r'\n(#{1,3}\s+[^\n]+)\n([^#]*?)(?=\n#{1,3}|\n\n|\Z)', buffer, re.DOTALL)
                        if header_match and len(header_match.group(0)) > 40:
                            section = header_match.group(0)
                            buffer = buffer[header_match.end():]
                            
                            # Clean up the section - ensure single spaces between words and proper line breaks
                            section = section.strip() + "\n\n"
                            section = re.sub(r'\s+', ' ', section)  # Convert any whitespace to single space
                            section = re.sub(r' +\n', '\n', section)  # Remove spaces before newlines
                            section = re.sub(r'\n ', '\n', section)  # Remove spaces after newlines
                            section = re.sub(r'(\n#{1,3}) ', r'\1 ', section)  # Ensure space after headers
                            
                            yield section
                            continue
                        
                        # Look for complete paragraphs (double newline separated)
                        paragraph_match = re.search(r'(?:^|\n\n)([^\n]+(?:\n[^\n#]+)*?)(?=\n\n|\n#{1,3}|\Z)', buffer, re.MULTILINE)
                        if paragraph_match and len(paragraph_match.group(1)) > 30:
                            paragraph = paragraph_match.group(0)
                            buffer = buffer[paragraph_match.end():]
                            
                            # Clean formatting - ensure proper spacing and line breaks
                            paragraph = re.sub(r'\s+', ' ', paragraph)  # Convert any whitespace to single space
                            paragraph = re.sub(r' +\n', '\n', paragraph)  # Remove spaces before newlines
                            paragraph = re.sub(r'\n ', '\n', paragraph)  # Remove spaces after newlines
                            if not paragraph.endswith('\n\n'):
                                paragraph += '\n\n'
                            yield paragraph
                            continue
                        
                        # Look for complete bullet point lists
                        list_match = re.search(r'(\n(?:[-*•]\s+[^\n]+\n)+)', buffer)
                        if list_match and len(list_match.group(0)) > 20:
                            list_section = list_match.group(0) + "\n"
                            buffer = buffer[list_match.end():]
                            
                            # Clean formatting - ensure proper spacing
                            list_section = re.sub(r'\s+', ' ', list_section)  # Single spaces
                            list_section = re.sub(r' +\n', '\n', list_section)  # No spaces before newlines
                            list_section = re.sub(r'\n ', '\n', list_section)  # No spaces after newlines
                            list_section = re.sub(r'(\n[-*•]) ', r'\1 ', list_section)  # Ensure space after bullet
                            yield list_section
                            continue
                        
                        # Look for complete numbered lists
                        num_list_match = re.search(r'(\n(?:\d+\.\s+[^\n]+\n)+)', buffer)
                        if num_list_match and len(num_list_match.group(0)) > 20:
                            num_list = num_list_match.group(0) + "\n"
                            buffer = buffer[num_list_match.end():]
                            
                            # Clean formatting - ensure proper spacing
                            num_list = re.sub(r'\s+', ' ', num_list)  # Single spaces
                            num_list = re.sub(r' +\n', '\n', num_list)  # No spaces before newlines
                            num_list = re.sub(r'\n ', '\n', num_list)  # No spaces after newlines
                            num_list = re.sub(r'(\n\d+\.) ', r'\1 ', num_list)  # Ensure space after number
                            yield num_list
                            continue
                        
                        # Look for sentence boundaries
                        sentence_match = re.search(r'([^.!?]*[.!?]\s+)', buffer)
                        if sentence_match and len(sentence_match.group(0)) > 50:
                            sentence = sentence_match.group(0)
                            buffer = buffer[sentence_match.end():]
                            
                            # Don't break markdown formatting
                            if not self._would_break_markdown(sentence):
                                sentence = re.sub(r'\s+', ' ', sentence)  # Single spaces between words
                                sentence = re.sub(r' +([.!?])', r'\1', sentence)  # No space before punctuation
                                yield sentence
                                continue
                        
                        # Fallback: word boundary chunking
                        words = buffer.split()
                        if len(words) >= 15:
                            # Find safe breaking point
                            for i in range(10, min(len(words), 20)):
                                word_chunk = ' '.join(words[:i])  # Remove the extra space
                                remaining = ' '.join(words[i:])
                                
                                if not self._would_break_markdown(word_chunk):
                                    buffer = remaining
                                    # Ensure single spaces between words
                                    word_chunk = re.sub(r'\s+', ' ', word_chunk).strip() + ' '
                                    yield word_chunk
                                    break
                            else:
                                # Ultimate fallback
                                word_chunk = ' '.join(words[:10])  # Remove the extra space
                                buffer = ' '.join(words[10:])
                                word_chunk = re.sub(r'\s+', ' ', word_chunk).strip() + ' '
                                yield word_chunk
                        else:
                            break
                
                # Send remaining content with proper cleanup
                if buffer.strip():
                    buffer = re.sub(r'\s+', ' ', buffer.strip())  # Single spaces between words
                    buffer = re.sub(r' +\n', '\n', buffer)  # Remove spaces before newlines
                    buffer = re.sub(r'\n ', '\n', buffer)  # Remove spaces after newlines
                    
                    # Ensure proper line breaks for structured content
                    if '###' in buffer or '-' in buffer or any(char.isdigit() and '.' in buffer for char in buffer):
                        # This looks like structured content, add proper spacing
                        buffer = re.sub(r'(\n)([-*•]\s+)', r'\1\n\2', buffer)  # Add space before bullet points
                        buffer = re.sub(r'(\n)(\d+\.\s+)', r'\1\n\2', buffer)  # Add space before numbered lists
                        buffer = re.sub(r'(\n)(#{1,3}\s+)', r'\1\n\2', buffer)  # Add space before headers
                    
                    if not buffer.endswith('\n'):
                        buffer += '\n'
                    yield buffer
                    
                self.response_cache[cache_key] = full_response
            else:
                raw_answer = self.llm.invoke(prompt)
                response = raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
                self.response_cache[cache_key] = response
                yield response
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
            "Provide comprehensive, well-structured responses with proper Markdown formatting. "
            "Use clear headings, proper spacing, bold text for emphasis, and organized sections. "
            "Ensure all formatting is clean and readable with proper line breaks and structure."
        )


        prompt = f"""
You are a helpful assistant providing well-formatted, comprehensive responses using proper Markdown structure.
If the user's question is outside the website’s domain, politely decline or use fallback web search context.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

CRITICAL FORMATTING REQUIREMENTS:
1. {length_rule}
2. Use proper Markdown structure:
   - ### for main headings (use engaging titles like "### Quick Overview", "### Key Points", "### Implementation Guide")
   - **Bold text** for important terms, technologies, and key concepts
   - Proper line breaks between sections
   - Bullet points or numbered lists for clarity
   - Code snippets with proper backticks when relevant

3. Structure your response clearly:
   - Start with a brief overview paragraph
   - Use sections with descriptive headings
   - Include specific details and examples
   - End with actionable recommendations when appropriate

4. Formatting best practices:
   - Ensure proper spacing around headings and sections
   - Use **bold** for technologies, key terms, and important concepts
   - Include bullet points for lists and features
   - Add line breaks for readability
   - Use engaging, professional tone

5. Content guidelines:
   - Synthesize information from multiple context sources when available
   - Provide comprehensive but focused answers
   - Include relevant examples and specifics
   - If context is limited, use general knowledge while staying on topic
   - Do not fabricate information outside the knowledge base scope

Important: Do NOT include example text, placeholders, or template content in your response. Provide only actual, relevant information.
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

# Follow-ups are now handled internally by the OptimizedChatbot class
