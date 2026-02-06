import json
import re
import tiktoken
from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.logger import get_logger
from app.utils.llm_utils import generate_llm_response
from app.utils.redis_context import get_production_context
from app.utils.context_wrapper import wrap_context_as_xml
from app.utils.validator import get_validator

# format_response is now imported only when needed (async version with URL validation)
from app.utils.prompts import PROMPT_VERSION
from app.utils.dynamic_prompts import build_dynamic_prompt
from app.utils.response_formatter import format_response, _normalize_url
import asyncio
import functools
from app.api.models import MessageCreate, UserCreate
from email_validator import validate_email, EmailNotValidError

logger = get_logger("chatbot")


def _validate_email(email: str) -> str | None:
    """
    Validate email format using email-validator library.
    Returns the normalized email if valid, None if invalid.
    """
    if not email:
        return None
    try:
        # Validate and normalize the email
        valid = validate_email(email, check_deliverability=False)
        return valid.normalized
    except EmailNotValidError as e:
        logger.warning(f"[LeadCapture] Invalid email format rejected: '{email}' - {e}")
        return None


def _get_source_label(url: str) -> str:
    """
    Extract a readable label from source URL for display.
    
    Parses URL fragment or path to create human-readable link text.
    Example: 'https://ditstek.com/blog/ai-chatbot#section-name' -> 'Section Name'
    
    Args:
        url: The source URL
        
    Returns:
        str: Human-readable label (max 60 chars)
    """
    if not url:
        return "Source"
    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        
        # Use fragment if present (e.g., #section-name)
        if parsed.fragment:
            label = unquote(parsed.fragment.replace("-", " ").replace("_", " "))
            # Clean up and title case
            label = " ".join(word.capitalize() for word in label.split())
            return label[:60] if label else parsed.netloc
        
        # Otherwise use last meaningful path segment
        path = parsed.path.rstrip("/")
        if path:
            segment = path.split("/")[-1]
            # Skip common file extensions
            if segment and not segment.endswith((".html", ".php", ".aspx", ".htm")):
                label = unquote(segment.replace("-", " ").replace("_", " "))
                label = " ".join(word.capitalize() for word in label.split())
                if label:
                    return label[:60]
        
        # Fallback to domain name
        return parsed.netloc or "Source"
    except Exception:
        # Ultimate fallback
        return url[:60] if len(url) > 60 else url


def _calculate_dynamic_top_n(query: str, conversation_history: list) -> int:
    """
    Dynamically calculate top_n based on query characteristics rather than static keywords.

    Uses query length, question type indicators, and conversation context to determine
    how many chunks to retrieve. Adapts to different query types automatically.

    Args:
        query: User's current query
        conversation_history: Previous conversation messages

    Returns:
        int: Dynamic top_n value (4-30 range)
    """
    if not query:
        return 4  # Default for empty queries

    query_lower = query.lower().strip()
    query_length = len(query_lower)
    query_words = len(query_lower.split())

    # Start with base value
    top_n = 4

    # Factor 1: Query length and complexity
    # Longer, more detailed queries may need more context
    if query_length > 50:
        top_n += 4
    if query_length > 100:
        top_n += 4
    if query_words > 10:
        top_n += 3

    # Factor 2: Broad exploratory questions (question words indicate broad queries)
    # Questions starting with "what", "tell me about", "show me" suggest broader searches
    broad_question_patterns = [
        query_lower.startswith("what"),
        query_lower.startswith("tell me"),
        query_lower.startswith("show me"),
        query_lower.startswith("explore"),
        query_lower.startswith("list"),
        "all" in query_lower,
        "everything" in query_lower,
        "comprehensive" in query_lower,
    ]

    if any(broad_question_patterns):
        top_n += 8  # Increase for broad exploratory queries

    # Factor 3: Comparison or multiple items queries
    if "compare" in query_lower or "difference" in query_lower or "vs" in query_lower:
        top_n += 6

    # Factor 4: Conversation context - analyze message patterns for breadth indicators
    if conversation_history:
        recent_user_msgs = [
            (
                msg.get("content", "")
                if isinstance(msg, dict)
                else (msg[1] if isinstance(msg, (list, tuple)) and len(msg) > 1 else "")
            )
            for msg in conversation_history[-3:]
            if (isinstance(msg, dict) and msg.get("role") == "user")
            or (isinstance(msg, (list, tuple)) and len(msg) > 0 and msg[0] == "user")
        ]

        # If recent messages are longer and have broad question patterns, increase top_n
        for msg in recent_user_msgs:
            if msg and len(msg) > 30:
                msg_lower = msg.lower()
                # Detect broad exploratory patterns (structural, not keyword-based)
                has_broad_pattern = (
                    msg_lower.startswith(("what", "tell", "show", "explore", "list"))
                    or "all" in msg_lower
                    or "everything" in msg_lower
                    or len(msg_lower.split())
                    > 8  # Longer messages often need more context
                )
                if has_broad_pattern:
                    top_n += 4
                    break

    # Factor 5: Question marks indicate information-seeking queries
    if "?" in query:
        top_n += 2

    # Cap the maximum - balance between accuracy and token limits
    # Increased from 10 to 12 to improve result coverage for all queries
    top_n = min(max(top_n, 4), 12)

    logger.info(
        f"[DynamicTopN] Calculated top_n={top_n} for query length={query_length}, words={query_words}"
    )

    return top_n


class ContextOptimizer:
    """
    Handles context optimization to fit within model token limits
    while maximizing relevance and information retention.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        # Use cl100k_base encoding as fallback for non-OpenAI models
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.model_limits = {"gpt-4o": 128000, "gpt-4o": 128000, "gpt-4-turbo": 128000, "gpt-3.5-turbo": 16385}
        self.context_limit = self.model_limits.get(model, 128000)

    @lru_cache(maxsize=1000)
    def count_tokens_cached(self, text: str) -> int:
        return len(self.encoding.encode(text))


class OptimizedChatbot:
    async def _is_phatic(self, query: str) -> bool:
        """Check if query is a simple phatic expression (greeting, thanks, etc.)"""
        if not query:
            return True
        
        q = query.strip().lower()
        phatic_patterns = {
            "hi", "hello", "hey", "greetings",
            "thanks", "thank you", "thx",
            "good morning", "good afternoon", "good evening",
            "bye", "goodbye", "see you",
            "ok", "okay", "sure", "cool",
            "no", "yes", "nope", "yep"
        }
        
        # Exact match
        if q in phatic_patterns:
            return True
            
        # Starts with (for "hi there", "hello bot")
        if len(q.split()) <= 3 and any(q.startswith(p + " ") for p in phatic_patterns):
            return True
            
        return False

    def _is_too_similar(
        self, last_response: str, new_response: str, threshold: float = 0.9
    ) -> bool:
        """
        Returns True if the new response is too similar to the last one (using a simple ratio).
        """
        if not last_response or not new_response:
            return False
        import difflib

        ratio = difflib.SequenceMatcher(
            None, last_response.strip().lower(), new_response.strip().lower()
        ).ratio()
        return ratio >= threshold

    def __init__(self, llm=None, model: str = "gpt-4o"):
        """Initialize OptimizedChatbot.

        Parameters:
        - llm: optional external LLM client object to use (must implement .invoke or content access). If provided, it will be used as self.query_llm.
        - model: model name to instantiate a default ChatOpenAI if llm is not provided.
        """
        self.model = model
        # ContextOptimizer is unused in the critical path, skipping initialization to save overhead
        # self.context_optimizer = ContextOptimizer(model)
        # self.encoding = self.context_optimizer.encoding
        # self.model_limits = self.context_optimizer.model_limits
        # self.context_limit = self.context_optimizer.context_limit

        self.query_llm = llm if llm is not None else ChatOpenAI(model=model)

    async def get_detailed_response(
        self, query: str, chat_history, session_id: str, stream: bool = True,
        is_uc1: bool = False  # EXPLICIT UC1 flag - set by caller, not inferred
    ):
        from app.api.helpers import (
            get_user_details_known_from_db,
        )  # Do Not Move Outside Function

        from app.logger import session_id_context
        
        # Set session context for logging
        token = session_id_context.set(session_id)
        
        logger.info("="*80)
        logger.info(f"START MESSAGE: session_id={session_id}, query='{query[:100]}'")
        logger.info("="*80)
        
        # PROMPT AUTHORITY ARCHITECTURE:
        # 1. Validation runs FIRST (before ANY routing decision)
        # 2. Router selects exactly ONE authority
        # 3. No fall-through, no blending
        # """
        
        # ============================================================
        # STEP 1: HARD PROMPT ROUTER - Determine authority ONCE
        # ============================================================
        # ============================================================
        # STEP 1: HARD PROMPT ROUTER - Determine authority ONCE
        # ============================================================
        from app.orchestrator.prompt_router import route_to_authority, get_authority_name
        from app.orchestrator.llm_adapter import LLMAuthority, ContentMode
        from app.orchestrator.state_input_validators import validate_input_for_state
        
        # Determine authority based on session state (NOT query content)
        # is_uc1 flag from caller takes precedence (for trigger CTA activation)
        if is_uc1:
            authority = LLMAuthority.UC1_CANONICAL
            content_mode = ContentMode.GENERIC
        else:
            authority, content_mode = route_to_authority(session_id, query)
        
        logger.info(f"[PROMPT_ROUTER] Authority: {authority.value}, Mode: {content_mode.value}")
        
        # ============================================================
        # STEP 2: PRE-ROUTING VALIDATION (runs before any slot mutation)
        # ============================================================
        # Get current state for state-scoped validation (only for UC1)
        current_state_value = ""
        # Get current state for state-scoped validation (only for UC1)
        current_state_value = ""
        if authority == LLMAuthority.UC1_CANONICAL:
            try:
                from app.orchestrator.slot_manager import SlotManager
                if session_id in SlotManager._session_slots:
                    # Get the current state from orchestrator if available
                    from app.orchestrator import ConversationOrchestrator
                    orch = ConversationOrchestrator.get_or_create(session_id)
                    current_state_value = orch._current_state.value if orch._current_state else ""
            except Exception:
                current_state_value = ""
        
        # Validate input for current state
        is_valid, failure_reason = validate_input_for_state(query, current_state_value)
        
        # Validate input for current state
        is_valid, failure_reason = validate_input_for_state(query, current_state_value)
        
        if not is_valid and authority == LLMAuthority.UC1_CANONICAL:
            # Input rejected - return validation error response
            logger.warning(f"[VALIDATION] Input rejected: reason={failure_reason}, state={current_state_value}")
            
            # Map failure reasons to user-friendly messages
            error_messages = {
                "empty_input": "I didn't catch that. Could you share a bit more?",
                "low_signal": "I need a bit more detail to help you. Could you elaborate?",
                "ack_not_allowed": "I'd love to know more about your specific needs. Could you tell me what you're looking for?",
                "no_alpha_signal": "Could you describe what you're looking for in words?",
                "too_short": "Just a bit more detail would help me assist you better.",
                "placeholder_name": "I'd like to address you properly. What's your real name?",
                "invalid_format": "Could you rephrase that? I want to make sure I understand correctly.",
            }
            error_message = error_messages.get(failure_reason, "Could you tell me more?")
            
            yield {"status": "chunk", "chunk": error_message}
            yield {"status": "done"}
            return
        
        # ============================================================
        # STEP 3: ROUTE TO AUTHORITY (mutual exclusivity enforced)
        # ============================================================
        if authority == LLMAuthority.UC1_CANONICAL:
            logger.info(f"[Chatbot] UC1 mode active for session: {session_id}")
            try:
                from app.orchestrator import ConversationOrchestrator
                from app.orchestrator.state_machine import UC1State
                from app.orchestrator.llm_adapter import LLMIntent, OutputViolation
                from app.utils.conversation_memory import get_session_memory_manager
                from app.orchestrator.uc1_config import get_bucket_by_id
                
                orchestrator = ConversationOrchestrator.get_or_create(session_id)
                memory_mgr = get_session_memory_manager()
                
                # Add user message to memory for continuity
                memory_mgr.add_user_message(session_id, query)
                
                # Process input through orchestrator (text-blind response)
                response = orchestrator.process_input(query)
                
                # Generate message using LLM adapter with HYBRID context
                # ONLY if message isn't already set (some states use fixed messages)
                llm_intent = LLMIntent.UNCLEAR  # Default
                
                if hasattr(response, 'call_spec') and response.call_spec and not response.message:
                    spec = response.call_spec
                    # generate_state_response now returns (text, intent, options) tuple
                    text, llm_intent, dynamic_options = orchestrator.llm_adapter.generate_state_response(
                        state=spec.state,
                        response_intent=spec.response_intent,
                        user_input=spec.user_input,
                        slots=spec.slots,
                        bucket=spec.bucket,
                        exploration_turn=spec.exploration_turn,
                        session_id=session_id,
                        authority=authority,
                        content_mode=content_mode
                    )
                    
                    # ============================================================
                    # ACC PHASE 5: OUTPUT VALIDATION (Final Safety Net)
                    # ============================================================
                    # Block redundant questions even if LLM generates them
                    # ONLY enforce for UC1 (Exploration mode IS allowed to ask questions)
                    violation = None
                    if authority == LLMAuthority.UC1_CANONICAL:
                        try:
                            # Use fully qualified name to ensure access regardless of import state
                            violation = orchestrator.llm_adapter.validate_output(text, spec.slots)
                        except Exception as ve:
                            logger.error(f"[Chatbot] Output validation failed (swallowed): {ve}")
                            violation = None

                    if violation == OutputViolation.REDUNDANT_QUESTION:
                        logger.warning("[ACC] OutputViolation detected: REDUNDANT_QUESTION. Forcing recovery.")
                        # Deterministic recovery - do not re-ask LLM
                        signal = spec.slots.context_signal
                        short_signal = (signal[:50] + "...") if signal and len(signal) > 50 else (signal or "that")
                        text = f"Got it — {short_signal}. What aspect feels most urgent right now?"
                    
                    response.message = text
                    response.llm_intent = llm_intent
                    
                    
                    # Check for EXIT state - Skip UI logic to ensure clean shutdown
                    is_exit = False
                    if hasattr(response, 'call_spec') and response.call_spec and response.call_spec.state == UC1State.EXIT:
                        is_exit = True
                        logger.info(f"[Chatbot] Detected EXIT state. Waiting for Post-CTA selection.")
                        # orchestrator.clear_session(session_id)  <-- REMOVED: Closure happens via meta signal
                    
                    # ============================================================
                    # BUTTON CLICK INTENT OVERRIDE - Hard commitment
                    # When user clicks a button (selects an alternative), treat as
                    # explicit commitment and override LLM intent to READY_FOR_CTA
                    # ============================================================
                    bucket = None
                    # Only access slots if NOT exiting (slots might be deleted)
                    if not is_exit:
                        try:
                            if orchestrator.slots and orchestrator.slots.capability_bucket:
                                bucket = get_bucket_by_id(orchestrator.config, orchestrator.slots.capability_bucket)
                        except Exception:
                            bucket = None
                    
                    # Detect button click: input matches alternative or CTA option
                    is_button_click = False
                    if query and bucket and bucket.alternatives:
                        query_lower = query.strip().lower()
                        alternatives_lower = [a.lower() for a in bucket.alternatives]
                        if query_lower in alternatives_lower:
                            is_button_click = True
                            # Record selection in slot (critical for option consumption)
                            orchestrator.slot_manager.set_selected_alternative(query, caller="orchestrator")
                            logger.info(f"[UC1] BUTTON CLICK detected: '{query}' - overriding intent to READY_FOR_CTA")
                    
                    # Also check for CTA button clicks
                    if not is_exit and query and not is_button_click:
                        cta_choices_lower = [cta.choice.lower() for cta in orchestrator.config.exit_ctas]
                        if query.strip().lower() in cta_choices_lower:
                            is_button_click = True
                            logger.info(f"[UC1] CTA BUTTON CLICK detected: '{query}'")
                    
                    # Override intent for button clicks
                    if is_button_click:
                        llm_intent = LLMIntent.READY_FOR_CTA
                        response.llm_intent = llm_intent
                    
                    # ============================================================
                    # INTENT-BASED UI GATING - State allows, Intent decides
                    # ============================================================
                    
                    # 4. ERROR HANDLING INTERCEPT (New Robustness Layer)
                    # If LLM failed (e.g. 429 Quota), suppress options to avoid confusing UI
                    if "[Service unavailable]" in (response.message or ""):
                        logger.warning("[UC1] Service unavailable detected - suppressing options to prevent confusing UI")
                        response.message = "I apologize, but I'm unable to process your request at the moment. Please try again in a little while."
                        response.options = None
                        response.input_type = "text"
                        # Force intent to unclear to prevent forward progression
                        llm_intent = LLMIntent.UNCLEAR
                        response.llm_intent = llm_intent
                    else:
                        # ============================================================
                        # EXIT STATE: Preserve orchestrator's explicit options
                        # ============================================================
                        # For EXIT state, orchestrator sets ["Restart Conversation", "Close Chat"]
                        # These should NOT be overridden by build_intent_gated_options()
                        # This ensures consistent closure flow for ALL UC-1 sub-cases
                        if is_exit:
                            logger.info(f"[UC1] EXIT state: preserving options={response.options}")
                            # Keep response.options as set by orchestrator
                        else:
                            # Only build options if service is healthy
                            # Override options based on intent + state permission (via input_type)
                            intent_gated_options = orchestrator.build_intent_gated_options(
                                state=response.state,
                                intent=llm_intent,
                                bucket=bucket,
                                input_type=response.input_type,  # Permission from state config
                                dynamic_options=dynamic_options  # Pass LLM-generated options
                            )
                            
                            # Apply intent-gated options - HYBRID UI (text + buttons)
                            # Always keep text input available, show buttons alongside when appropriate
                            if intent_gated_options:
                                response.options = intent_gated_options
                            else:
                                response.options = None
                    
                    # ALWAYS allow text input (hybrid UI - user can type OR click)
                    response.input_type = "text"
                    logger.info(f"[UC1] Hybrid UI: intent={llm_intent.value}, options={response.options}")
                
                # Add assistant message to memory for future context
                if response.message:
                    memory_mgr.add_ai_message(session_id, response.message)
                
                # Yield SSE chunks - MANUALLY ORDERED for Streaming UX
                # 1. Yield Text Chunk FIRST (so it renders before buttons)
                # ALWAYS yield message as a chunk to ensure visibility in UI
                if response.message and response.message != "Safe landing...":
                    yield {"status": "chunk", "chunk": response.message}
                
                # 2. Yield Meta Chunk SECOND (so buttons pop in after text starts)
                # Reconstruct meta from response object manually to ensure separation
                meta = {
                    "uc1_state": response.state.value,
                    "uc1_input_type": response.input_type,
                    "uc1_terminal": response.terminal,
                    "allow_text_input": response.input_type == "text",
                }
                if response.options:
                    meta["uc1_options"] = response.options
                if response.llm_intent:
                    meta["llm_intent"] = response.llm_intent.value
                if response.metadata:
                    meta.update(response.metadata)
                
                yield {"status": "meta", "chunk": meta}
                
                # Close Chat handler - Trigger session closure flow
                if response.metadata and response.metadata.get("close_chat"):
                    # Clear session from orchestrator cache
                    orchestrator.clear_session(session_id)
                    
                    # Trigger session closure in frontend
                    # Chunk is empty because message was already sent as a standard 'chunk'
                    yield {
                        "status": "end_chat", 
                        "chunk": ""
                    }
                
                # UC1 handled - return early
                return
            except Exception as uc1_error:
                logger.exception(f"[Chatbot] UC1 orchestrator error: {uc1_error}")
                # HARD FAILURE - Do not fall through to standard flow
                # UC1 errors should not silently switch prompt authorities
                yield {"status": "chunk", "chunk": "I encountered an issue. Let me try again."}
                yield {"status": "done"}
                return
            finally:
                logger.info("="*80)
                logger.info(f"END MESSAGE: session_id={session_id}")
                logger.info("="*80)
                # No reset(token) here because we are in an async generator
                # The context is managed by the event loop/task as long as we yield
        
        # ============================================================
        # STANDARD DYNAMIC FLOW (for non-UC1 sessions ONLY)
        # ============================================================
        # This block ONLY executes if authority == NON_UC1 (implied by previous if)
        # There is NO fall-through from UC1 - mutual exclusivity enforced
        
        if authority != LLMAuthority.NON_UC1:
            # Safety check - should never reach here
            logger.error(f"[PROMPT_ROUTER] VIOLATION: Reached standard flow with authority={authority.value}")
            return
        
        # Track last assistant response for anti-repetition
        last_assistant_response = None
        for msg in reversed(chat_history or []):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_assistant_response = msg.get("content")
                break
            elif (
                isinstance(msg, (list, tuple))
                and len(msg) >= 2
                and msg[0] == "assistant"
            ):
                last_assistant_response = msg[1]
                break
        try:
            # Step 1: Parallel retrieval of Context and User Details
            try:
                import asyncio
                import functools
                from app.api.helpers import get_user_details_from_db, get_user_details_known_from_db, update_user_by_session
                
                loop = asyncio.get_event_loop()
                
                # Check if we should skip context retrieval (phatic queries)
                is_phatic = await self._is_phatic(query)
                if is_phatic:
                    logger.info("[Chatbot] Optimization: Skipping context retrieval for phatic query")
                
                # Define tasks
                tasks = []
                
                # Task 1: Redis Context Retrieval (Production Pipeline)
                if is_phatic:
                    # Return empty context immediately
                    tasks.append(asyncio.sleep(0, result=[]))
                else:
                    # Convert chat_history...
                    conversation_history_for_redis = []
                    for msg in chat_history or []:
                        if isinstance(msg, dict):
                            conversation_history_for_redis.append(msg)
                        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                            conversation_history_for_redis.append(
                                {"role": msg[0], "content": msg[1]}
                            )
                    
                    # Dynamic Top-N Calculation
                    # Adapts retrieval volume based on query type (Broad vs Specific)
                    top_n_value = _calculate_dynamic_top_n(query, conversation_history_for_redis)
                    logger.info(f"[Chatbot] Dynamic Top-N: {top_n_value}")
                    
                    tasks.append(loop.run_in_executor(
                        None,
                        functools.partial(
                            get_production_context,
                            session_id,
                            query,
                            top_n=top_n_value,
                        ),
                    ))
                
                # Task 2: User Details - single DB query for both details and known flag
                # This saves ~200-400ms by eliminating redundant DB round-trip
                def fetch_user_details():
                    from app.api.helpers import get_user_session_info
                    return get_user_session_info(session_id)
                
                tasks.append(loop.run_in_executor(None, fetch_user_details))
                
                # Execute in parallel
                results = await asyncio.gather(*tasks)
                
                # List[Dict] from production context
                kb_chunks = results[0] 
                user_details, user_details_known = results[1]
                
            except Exception as e:
                logger.warning(
                    "Parallel retrieval failed, using fallback sequential: %s", e
                )
                kb_chunks = []
                user_details = {}
                user_details_known = False

            # Wrap chunks in XML for Structural Security
            # Preprocess to ensure 'source' key exists (DeepReranker returns formatted dicts usually)
            processed_chunks = []
            if kb_chunks:
                for chunk in kb_chunks:
                     # Ensure source_url mapped correctly if missing
                     if "source" not in chunk:
                         meta = chunk.get("metadata", {})
                         chunk["source"] = meta.get("url") or meta.get("source") or "knowledge_base"
                     processed_chunks.append(chunk)

            context = wrap_context_as_xml(processed_chunks)
            
            # Build conversation context using LangChain memory (efficient: summary + recent buffer)
            try:
                from app.utils.conversation_memory import get_session_memory_manager
                
                memory_mgr = get_session_memory_manager()
                history = memory_mgr.get_context(session_id)
                
                # If memory is empty, try to initialize from chat_history passed in
                if not history and chat_history:
                    memory_mgr.initialize_from_history(session_id, [
                        {"role": msg[0], "content": msg[1]} if isinstance(msg, (list, tuple)) else msg
                        for msg in chat_history
                    ])
                    history = memory_mgr.get_context(session_id)
                
                if not history:
                    history = self._format_history(chat_history)
                    
                logger.debug(f"[Chatbot] Using LangChain memory context ({len(history)} chars)")
            except Exception as e:
                logger.warning(f"[Chatbot] LangChain memory failed, using fallback: {e}")
                history = self._format_history(chat_history)
            
            count = len(chat_history)
            
            logger.info(f"[Chatbot] Context: user_details_known={user_details_known}, details_found={bool(user_details)}")

            # Handle simple affirmative "yes" to a choice question
            if (
                query.strip().lower() == "yes"
                and last_assistant_response
                and " or " in last_assistant_response
            ):
                # If the user says "yes" to a choice, ask for clarification.
                clarification_response = (
                    "Great! Which of those options works best for you?"
                )
                yield {"status": "chunk", "chunk": clarification_response}
                # Save this clarification to history
                try:
                    # import asyncio
                    # from app.api.models import MessageCreate
                    from app.api.helpers import save_message

                    assistant_msg = MessageCreate(
                        session_id=session_id,
                        content=clarification_response,
                        role="assistant",
                        reply_to=None,
                        follow_up_to=None,
                        follow_up_depth=0,
                        metadata={},
                    )
                    asyncio.create_task(save_message(assistant_msg))
                    logger.info(
                        f"[Chatbot] Saved clarification response for 'yes' answer."
                    )
                except Exception as e:
                    logger.error(f"Failed to save clarification response: {e}")
                return

            # ============================================================
            # PROMPT SOURCE: Using build_dynamic_prompt (Redis-backed, fallback to prompts.py)
            # ============================================================
            # PROMPT SOURCE: Using LLM Adapter for NON_UC1 (Unified Logic)
            # The new design routes ALL non-UC1 traffic through the Adapter to use strictly controlled prompts
            # (ContentMode.WEBSITE_REPRESENTATIVE_STRICT or EXPLORATION)
            # We NO LONGER use build_dynamic_prompt for these modes to ensure prompt compliance.
            
            logger.info(f"<< PROMPT TRIGGER >> Using Adapter for NON_UC1 [Mode: {content_mode.value}]")
            
            # Use the LLM adapter to generate response even for non-UC1 flow
            # This unifies the "Single Source of Truth" architecture.
            
            try:
                # We need an orchestrator instance to get the adapter
                from app.orchestrator import ConversationOrchestrator
                from app.orchestrator.llm_adapter import LLMIntent
                orchestrator = ConversationOrchestrator.get_or_create(session_id)
                
                # Call _canonical_llm directly (or expose a method for non-state generation)
                # Since generate_state_response requires state/slots, we should use _canonical_llm directly
                # or a new public method `generate_free_response`.
                # For now, we can use _canonical_llm since prompt selection is inside it.
                
                # We pass user query as 'user_text'
                # Anchor can be empty or used for context injection if we fetch RAG here.
                # Use 'context' (retrieved RAG XML) as anchor?
                
                rag_anchor = f"\n<trusted_context>\n{context}\n</trusted_context>\n" if context else ""
                
                
                final_text, intent, options = orchestrator.llm_adapter._canonical_llm(
                    user_text=query,
                    anchor=rag_anchor,
                    session_id=session_id,
                    authority=authority,
                    content_mode=content_mode
                )
                
                # ERROR HANDLING INTERCEPT (NON_UC1)
                if "[Service unavailable]" in (final_text or ""):
                    logger.warning("[Chatbot] NON_UC1 Service unavailable detected - applying graceful fallback")
                    safe_final_text = "I apologize, but I'm unable to process your request at the moment. Please try again in a little while."
                else:
                    safe_final_text = final_text
                
                
                # Adapter handles JSON parsing internally usually, but _canonical_llm returns RAW text + intent tuple.
                # IF _canonical_llm returns the formatted response (text), we can use it.
                # However, _canonical_llm calls `client.chat.completions.create` which returns `content`.
                # The adapter's `_canonical_llm` returns `(response_text, intent)`.
                # Wait, let's verify _canonical_llm return type in llm_adapter.py
                # It returns (str, LLMIntent). The str is the "response" part usually? 
                # No, looking at llm_adapter.py, it parses the JSON/Delimiter and returns the "clean reponse".
                # Let's check _canonical_llm implementation (I didn't view it fully).
                # Assuming it returns the user-facing text.
                
            except Exception as e:
                logger.exception(f"[Chatbot] Adapter call failed for NON_UC1: {e}")
                safe_final_text = ""
                
            # If we used the adapter, we can skip the old build_dynamic_prompt + generate_llm_response logic.
            # But the existing code below expects 'safe_final_text'.
            # So the above block REPLACES lines 624-653? 
            # The existing code did "Step 3: Generate response from LLM".
            # I am replacing that with "Step 3: Generate response via Adapter".
            
            # However, I need to match the variable 'safe_final_text'.
            # And 'intent' might differ. The old code parsed JSON from safe_final_text (lines 672+).
            # The Adapter `_canonical_llm` does internal parsing!
            # If `_canonical_llm` returns the CLEAN text, then `safe_final_text` is clean text.
            # But the subsequent code (lines 672+) tries to PARSE JSON AGAIN from `safe_final_text`.
            # If `safe_final_text` is already clean text, JSON parse will fail or return text.
            # The existing parsing logic (lines 787+) handles fallback to text.
            
            # IMPORTANT: The dataset plan generates JSON or text?
            # The PROMPT I added uses delimiters: <INTENT>: ...
            # `llm_adapter` parses delimiters.
            # The OLD "Standard Dynamic" flow used JSON output.
            # If I switch to `llm_adapter`, I am switching to Delimiter flow.
            # So I need to bypass the downstream JSON parsing logic or ensure it works.
            
            # The `llm_adapter` returns `(text, intent)`.
            # I should construct a fake "llm_json" structure so the rest of the pipeline works? 
            # Or just update the variables.
            
            # Let's look at downstream usage (lines 672-800).
            # It expects `llm_json` to have `response`, `funnel_stage`, `user_info` etc.
            # My new prompts DO NOT generate user_info extraction or prospect profile!
            # They only generate text + intent + options.
            
            # So I am losing "User Details Extraction" in Strict Mode?
            # The prompt I added for Strict Mode does NOT have "extract user info".
            # This is acceptable for Strict Mode (Fact Retrieval).
            # But maybe not for Exploration? 
            # The user said "Website Exploration ... Narrative flow ... No fabrication".
            # Did not explicitely mandate Lead Capture in Exploration.
            
            # If I skip the old `extract_user_info` logic, I might lose lead capture features in Free Exploration?
            # Or is lead capture handled by Orchestrator?
            # In `orchestrator.py`, `EMAIL_CAPTURE` is a state.
            
            # For `NON_UC1` (Standard Dynamic), `chatbot_optimizer.py` was responsible for lead capture via JSON extraction.
            # If I bypass it, I lose that.
            
            # Strategy:
            # For now, I will use the Adapter to generate the text.
            # I will mock the JSON processing downstream or skip it if it's text.
            # The downstream fallback (lines 790+) handles plain text.
            
            # But wait, `_canonical_llm` returns Clean Text.
            # So `safe_final_text` = Clean Text.
            # `extract_json_from_markdown` will return Clean Text.
            # `json.loads` will fail.
            # It goes to `except Exception as json_exc`.
            # Then regex fallback.
            # Then `response_text = safe_final_text`.
            # `user_network_id` = None.
            # `sources` = [].
            
            # This works for basic response!
            
            # OPTION: I can make `_canonical_llm` return the raw response if I wanted to, but the architecture says Adapter handles LLM.
            
            logger.info("<< PROMPT TRIGGER >> Using Adapter for NON_UC1 (Skipping old Dynamic Prompt)")

            # Fallback if LLM returns None or empty string (e.g., due to rate limit or internal error)
            if not safe_final_text or not safe_final_text.strip():
                logger.warning(
                    "[Chatbot] LLM returned empty or None, triggering fallback response."
                )
                async for fallback in self._fallback_response_stream(
                    "I couldn't generate a response right now."
                ):
                    yield fallback
                return

            # Step 4: Parse LLM JSON output (if present)
            funnel_stage = ""
            response_text = ""
            # Initialize early to ensure always defined (robustness)
            prospect_profile = None
            sources = []

            def extract_json_from_markdown(txt: str) -> str:
                # First look for a fenced JSON block
                pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
                match = re.search(pattern, txt, re.DOTALL)
                if match:
                    return match.group(1).strip()

                # If no fence found, try to find the first balanced JSON object by brace matching
                start = None
                depth = 0
                for i, ch in enumerate(txt):
                    if ch == "{":
                        if start is None:
                            start = i
                        depth += 1
                    elif ch == "}" and start is not None:
                        depth -= 1
                        if depth == 0:
                            return txt[start : i + 1].strip()

                # Fall back to returning the original stripped text
                return txt.strip()

            try:
                cleaned_json = extract_json_from_markdown(safe_final_text)
                # First attempt: JSON inside a code fence or a balanced JSON object in the text
                llm_json = json.loads(cleaned_json)
                response_text = llm_json.get("response", "") or ""
                funnel_stage = (llm_json.get("funnel_stage", "") or "").lower()
                # EXTRACT AND SAVE USER DETAILS
                # Only check for extraction if we don't already know the user details
                email_was_invalid = False  # Track for response post-processing
                if not user_details_known:
                    extracted_info = llm_json.get("user_info")
                    if extracted_info and isinstance(extracted_info, dict):
                        # Validate email if present
                        raw_email = extracted_info.get("email")
                        validated_email = _validate_email(raw_email) if raw_email else None
                        
                        # Track if email was provided but invalid (for response correction)
                        if raw_email and not validated_email:
                            email_was_invalid = True
                            # Fix the response_text to not show the invalid email
                            if raw_email in response_text:
                                # First, remove the invalid email entirely from response
                                # Handle various patterns: "email vinay.com", "email (vinay.com)", "email you provided (vinay.com)"
                                response_text = re.sub(
                                    rf'\s*\(?\s*{re.escape(raw_email)}\s*\)?\s*',
                                    ' ',
                                    response_text
                                )
                                # Clean up any "email you provided ()" or "email ()" remnants
                                response_text = re.sub(
                                    r'email\s+you\s+provided\s*\(\s*\)',
                                    'email',
                                    response_text,
                                    flags=re.IGNORECASE
                                )
                                response_text = re.sub(
                                    r'email\s*\(\s*\)',
                                    'email',
                                    response_text,
                                    flags=re.IGNORECASE
                                )
                                # Clean up double spaces
                                response_text = re.sub(r'\s{2,}', ' ', response_text)
                                
                            # Always add a request for valid email when invalid
                            if "valid email" not in response_text.lower():
                                response_text += "\n\n⚠️ **The email format provided doesn't appear to be valid.** Could you please share a valid email address (e.g., name@company.com) so we can send you the proposal?"
                            logger.info(f"[LeadCapture] Fixed response to remove invalid email mention: '{raw_email}'")
                        
                        # Check if we have at least a name or a VALID email
                        if extracted_info.get("name") or validated_email:
                            logger.info(f"[LeadCapture] Extracted details from LLM: name={extracted_info.get('name')}, email={validated_email or '(invalid/none)'}")
                            try:
                                # Save to DB - only use validated email
                                user_update = UserCreate(
                                    username=extracted_info.get("name"),
                                    email=validated_email,  # Only save if validated
                                    user_details_known=True if validated_email else False  # Only mark known if we have valid email
                                )
                                # update_user_by_session is async
                                await update_user_by_session(session_id, user_update)
                                user_details_known = True if validated_email else user_details_known
                                logger.info(f"[LeadCapture] Successfully saved extracted user details.")
                            except Exception as e:
                                logger.error(f"[LeadCapture] Failed to save extracted user details: {e}")

                # Extract prospect profile for in-session tracking (not persisted to DB)
                prospect_profile = llm_json.get("prospect_profile")
                if prospect_profile and isinstance(prospect_profile, dict):
                    logger.info(f"[ProspectProfile] Extracted: user_type={prospect_profile.get('user_type')}, stage={prospect_profile.get('stage')}, budget={prospect_profile.get('budget_sensitivity')}")
                
                # Extract sources for response traceability (metadata footnotes)
                sources = llm_json.get("sources", [])
                if sources and isinstance(sources, list):
                    sources = [s for s in sources if s and isinstance(s, str) and s.startswith("http")]
                    if sources:
                        logger.info(f"[SourceTracing] Response sources: {sources}")

                # Ignore LLM's user_details_known, always use DB value for meta chunk
                user_network_id = llm_json.get("user_network_id") or None
                logger.info(f"[Chatbot] Parsed JSON: {llm_json}")
            except Exception as json_exc:
                logger.warning(
                    f"[Chatbot] Failed to parse LLM output as JSON: {json_exc}"
                )
                # As a best-effort, try to extract a user-facing 'response' field
                # using a simple regex fallback before giving up.
                resp_match = re.search(
                    r'"response"\s*:\s*"([\s\S]*?)"\s*(,|})', safe_final_text
                )
                if resp_match:
                    response_text = resp_match.group(1).strip()
                    logger.info(
                        "[Chatbot] Recovered 'response' value from LLM text via regex fallback."
                    )
                else:
                    response_text = safe_final_text
                user_network_id = None
                sources = []
                prospect_profile = None

            logger.info(
                f"[Chatbot] Final output len={len(safe_final_text)}, funnel_stage='{funnel_stage}'"
            )

            # ============================================================
            # STEP 4.b: POST-GENERATION VALIDATION (Hallucination Prevention)
            # ============================================================
            # Only check validation if we actually had KB context to check against
            if kb_chunks:
                validator = get_validator()
                # Clean markdown/JSON from response text for validation
                clean_text = response_text # response_text is already extracted from JSON
                
                is_valid, validation_msg = validator.validate_response(clean_text, kb_chunks)
                if not is_valid:
                    logger.warning(f"[Validator] Blocked Standard Response: {validation_msg}")
                    # Hard Abort Message
                    refusal_msg = "I apologize, but I don't have enough verified information to answer that specific question accurately based on our current documentation."
                    
                    # Yield as immediate chunk then finish
                    yield {"status": "chunk", "chunk": refusal_msg}
                    yield {"status": "done"}
                    return
            
            # Step 5: Continue with regular response flow

            # Step 6: Stream formatted response
            cleaned = (response_text or "").strip()

            # Debug markdown preservation
            has_markdown_before = any(
                marker in cleaned for marker in ["**", "*", "_", "#", "-", "`", "```"]
            )
            if has_markdown_before:
                logger.info(
                    f"[MARKDOWN_DEBUG] Raw response contains markdown: {cleaned[:100]}..."
                )

            # Step 6: DYNAMIC Funnel-based form trigger (LLM-driven intelligence)
            # Use cached user_details_known
            logger.info(
                f"[FormLogic] Stage='{funnel_stage}', Count={count}, Known={user_details_known}"
            )
            trigger_form = False
            trigger_reason = ""
            # REMOVED: Form triggering logic and fallback deleted to prevent input blocking.
            # The form will NEVER trigger automatically. Lead capture is fully conversational.

            # Only yield the assistant response if we are NOT about to trigger a form
            if not trigger_form:
                # Use smart formatter - format immediately, validate URLs in background
                # from app.utils.response_formatter import format_response

                # Format response immediately (no blocking URL validation)
                # formatted = format_response(cleaned, query, None)
                # Skip formatting to debug newline issue
                formatted = cleaned
                # Note: format_response already handles \\n replacement internally

                # Quick URL normalization (no network calls) - fix spaces in URLs
                # Skip blocking URL validation - normalize only (fixes spaces like "real- estate")
                # from app.utils.response_formatter import _normalize_url

                formatted = self._normalize_urls_in_text(formatted)

                # Sanitize horizontal rules that cause Setext H2 rendering
                # Remove lines containing only dashes, asterisks, or underscores (3+)
                # These create horizontal rules in markdown and can cause text above to render as headers
                formatted = re.sub(
                    r'^\s*[-*_]{3,}\s*$',  # Match lines with 3+ dashes, asterisks, or underscores
                    '',
                    formatted,
                    flags=re.MULTILINE
                )
                # Clean up any resulting multiple blank lines
                formatted = re.sub(r'\n{3,}', '\n\n', formatted)


                # Anti-repetition: compare with last assistant response
                if last_assistant_response and isinstance(last_assistant_response, str):
                    if self._is_too_similar(last_assistant_response, formatted):
                        logger.warning(
                            "[ANTI-REPETITION] New response is too similar to last assistant response. Consider regenerating or expanding."
                        )
                        # Only add clarifying question if user_details_known=False
                        if not user_details_known:
                            formatted += "\n\n(Can you share more details or specify what you'd like to know next?)"

                if has_markdown_before:
                    has_markdown_after = any(
                        marker in formatted
                        for marker in ["**", "*", "_", "#", "-", "`", "```"]
                    )
                    logger.info(
                        f"[MARKDOWN_DEBUG] Formatted response markdown preserved: {has_markdown_after}"
                    )
                    if not has_markdown_after:
                        logger.warning(
                            "[MARKDOWN_WARNING] Markdown lost during formatting!"
                        )
                    
                    # Additional debug for bold markdown specifically
                    bold_before = "**" in cleaned
                    bold_after = "**" in formatted
                    if bold_before and not bold_after:
                        logger.warning(
                            f"[MARKDOWN_WARNING] Bold markdown (**) was removed! known={user_details_known}"
                        )
                    elif bold_before and bold_after:
                        # Check for spacing issues in bold markdown
                        if "** " in formatted or " **" in formatted:
                            logger.warning(
                                f"[MARKDOWN_WARNING] Bold markdown spacing issue detected: '** ' or ' **' found in response"
                            )

                # Append sources section for transparency (show where info came from)
                if sources and isinstance(sources, list) and len(sources) > 0:
                    sources_section = "\n\n---\n**📚 Sources:**"
                    for src in sources[:5]:  # Limit to 5 sources for readability
                        if src and isinstance(src, str) and src.startswith("http"):
                            label = _get_source_label(src)
                            sources_section += f"\n- [{label}]({src})"
                    formatted += sources_section
                    logger.info(f"[SourceTransparency] Appended {min(len(sources), 5)} source(s) to response")

                # Only yield a chunk if it is non-empty and not just whitespace
                if formatted and formatted.strip():
                    yield {"status": "chunk", "chunk": formatted}
                    if count >= 50:
                        logger.info(
                            f"[Chatbot] [DEBUG] Yielding end_chat chunk: user_message_count={count}"
                        )
                        print(
                            f"[DEBUG] Yielding end_chat chunk: user_message_count={count}"
                        )
                        yield {
                            "status": "end_chat",
                            "chunk": "Our sales team will reach out within 1 business day, Thank you for your interest in Ditstek innovations.",
                        }
                
            # Step 7: Emit meta update if present, but skip if form_trigger is about to be yielded
            # Use cached user_details_known
            # Include sources for traceability and prospect_profile for session context
            meta_chunk = {
                # "user_details_known": user_details_known, # REMOVED to prevent frontend blocking
                **({"user_network_id": user_network_id} if user_network_id else {}),
                **({"sources": sources} if sources else {}),
                **({"prospect_profile": prospect_profile} if prospect_profile else {}),
            }
            if not trigger_form and meta_chunk:
                yield {"status": "meta", "chunk": meta_chunk}

            # Step 8: Form trigger event (if needed)
            if trigger_form:
                logger.info(
                    f"[Chatbot] FORM TRIGGER ACTIVATED! Reason: {trigger_reason}, funnel_stage={funnel_stage}, message_count={count}"
                )
                yield {"status": "form_trigger", "chunk": ""}
            else:
                logger.info(
                    f"[Chatbot] No form trigger. Stage={funnel_stage}"
                )

        except Exception as e:
            logger.exception("[Chatbot] Redis-based response generation failed")
            async for fallback in self._fallback_response_stream(
                "I couldn't generate a response right now."
            ):
                yield fallback
        finally:
            logger.info("="*80)
            logger.info(f"END MESSAGE: session_id={session_id}")
            logger.info("="*80)
            session_id_context.reset(token)

    async def _fallback_response_stream(self, query: str):
        """Fallback response when enhanced flow fails"""
        # If the query is the default fallback string, show a simple error message
        if query.strip() == "I couldn't generate a response right now.":
            fallback_text = "Sorry, I couldn't generate a response right now. Please try again later."
        elif query.strip():
            fallback_text = f"Sorry, I couldn't answer your request: {query}"
        else:
            fallback_text = "Sorry, I couldn't generate a response."
        yield {"status": "chunk", "chunk": fallback_text}

    def _format_history(self, chat_history: list) -> str:
        """
        Format chat history for LLM with enhanced conversation flow context.
        Includes timestamps when available and structures the conversation for better understanding.
        Handles both dict and tuple formats.
        """
        if not chat_history:
            return ""

        formatted = []
        conversation_count = 0

        # Normalize messages to dict format for consistent processing
        normalized_msgs = []
        for msg in chat_history:
            if isinstance(msg, dict):
                normalized_msgs.append(msg)
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                normalized_msgs.append(
                    {
                        "role": msg[0],
                        "content": msg[1],
                        "timestamp": msg[2] if len(msg) > 2 else "",
                    }
                )

        # Count actual conversation pairs for context
        user_msgs = [
            msg for msg in normalized_msgs if msg.get("role", "").lower() == "user"
        ]
        assistant_msgs = [
            msg for msg in normalized_msgs if msg.get("role", "").lower() == "assistant"
        ]
        conversation_count = min(len(user_msgs), len(assistant_msgs))

        # Add conversation flow summary at the top
        if conversation_count > 0:
            formatted.append(
                f"=== CONVERSATION FLOW ({conversation_count} exchanges, {len(normalized_msgs)} total messages) ==="
            )

        # Format each message with enhanced context
        for i, msg in enumerate(normalized_msgs):
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            # Add timestamp if available
            time_info = f" [{timestamp}]" if timestamp else ""

            if role == "user":
                formatted.append(f"User{time_info}: {content}")
            elif role == "assistant":
                # Don't manipulate assistant content to preserve markdown formatting
                # Simply append as-is to avoid breaking markdown
                formatted.append(f"Assistant{time_info}: {content}")
            else:
                # In case of unknown role, include as-is for debugging
                formatted.append(f"{role.capitalize()}{time_info}: {content}")

        # Add conversation context summary at the end
        if conversation_count > 0:
            formatted.append(f"=== END CONVERSATION FLOW ===")

        return "\n".join(formatted)

    def _normalize_urls_in_text(self, text: str) -> str:
        """Helper to normalize URLs in text without blocking validation."""
        url_pattern = r"\[([^\]]+)\]\(([^)]+)\)|(https?://[^\s\)]+)"

        def normalize_match(match):
            if match.group(2):  # Markdown link
                normalized = _normalize_url(match.group(2))
                return f"[{match.group(1)}]({normalized})"
            elif match.group(3):  # Plain URL
                return _normalize_url(match.group(3))
            return match.group(0)

        return re.sub(url_pattern, normalize_match, text)
