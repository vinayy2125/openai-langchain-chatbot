# UC1 Conversation Orchestrator - Central Controller (LLM-First Architecture)
#
# ARCHITECTURE INVARIANT:
#   The fine-tuned LLM generates 100% of user-visible language.
#   The orchestrator generates 0%.
#
# The orchestrator is TEXT-BLIND. It:
# 1. Controls state transitions (deterministic)
# 2. Manages slots (deterministic)
# 3. Returns AdapterCallSpec (WHY to speak, not WHAT)
# 4. NEVER emits user-visible strings
#
# Language generation happens ONLY in llm_adapter.generate_state_response()

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, AsyncGenerator
import re
from app.orchestrator.uc1_config import (
    UC1Config, load_uc1_config, get_bucket_by_id, 
    get_bucket_by_trigger, CapabilityBucket
)
from app.orchestrator.state_machine import UC1State, UC1StateMachine, ResponseIntent
from app.orchestrator.slot_manager import UC1Slots, SlotManager, EngagementEvent
from app.orchestrator.llm_adapter import ConstrainedLLMAdapter, LLMIntent
from app.orchestrator.output_sanitizer import LLMOutputSanitizer
from app.orchestrator.policy_validator import UC1PolicyValidator
from app.orchestrator.policy_engine import (
    ConversationPolicy, PolicyDecision, PolicyContext,
    UserIntent, create_policy_context
)
from app.orchestrator.input_classifier import classify_input, InputClass
from app.orchestrator.state_input_validators import validate_context_answer, validate_name, validate_email
from app.orchestrator.button_manager import ButtonManager
from app.logger import get_logger

logger = get_logger("orchestrator")


# Type for input types expected by orchestrator
InputType = Literal["buttons", "text", "none"]


@dataclass
class AdapterCallSpec:
    """
    Text-blind specification for LLM adapter.
    
    This is what the orchestrator outputs - it specifies WHY something
    must be said, not WHAT. The LLM adapter translates this to language.
    
    INVARIANT: Orchestrator returns only AdapterCallSpec, never strings.
    """
    state: UC1State
    response_intent: ResponseIntent
    user_input: Optional[str] = None
    slots: Optional[UC1Slots] = None
    bucket: Optional[CapabilityBucket] = None
    exploration_turn: int = 0  # For exploration layer


@dataclass 
class OrchestratorResponse:
    """
    Output schema for frontend enforcement.
    
    The 'message' field is populated by the LLM adapter AFTER
    the orchestrator returns, not by the orchestrator itself.
    
    STATE ALLOWS. INTENT DECIDES.
    - options are gated by llm_intent, not state
    """
    state: UC1State
    call_spec: AdapterCallSpec  # What the LLM adapter needs
    input_type: InputType
    options: Optional[List[str]] = None
    terminal: bool = False
    metadata: Optional[Dict[str, Any]] = None
    llm_intent: Optional[LLMIntent] = None  # Intent inferred by LLM for UI gating
    
    # Message is set by caller after LLM generation, not by orchestrator
    message: str = ""
    
    def __post_init__(self):
        """
        ARCHITECTURAL ASSERTION (Phase 7 - micro-correction #7):
        
        Either the orchestrator speaks (message is set) OR the LLM speaks
        (call_spec.state not in fixed-prompt states). Never both.
        
        This prevents future regressions by new developers.
        """
        from app.orchestrator.state_machine import UC1State
        FIXED_PROMPT_STATES = {UC1State.ENTRY, UC1State.CONTEXT_QUESTION, UC1State.NAME_CAPTURE, UC1State.EXIT}
        
        # If message is set by orchestrator AND state requires LLM generation, that's a violation
        # (Fixed-prompt states should have message set, others should not)
        is_fixed_prompt_state = self.call_spec.state in FIXED_PROMPT_STATES
        
        # The invariant: if message is populated AND this is NOT a fixed-prompt state,
        # LLM would overwrite it anyway, so it's wasteful/confusing
        # For now we allow message in non-fixed states (robustness handlers set messages)
        # The key invariant is: LLM must NOT be called for fixed-prompt states (enforced in llm_adapter)
    
    def to_sse_chunks(self) -> List[Dict[str, Any]]:
        """Convert response to SSE-compatible chunks."""
        chunks = []
        
        if self.message:
            chunks.append({"status": "chunk", "chunk": self.message})
        
        meta = {
            "uc1_state": self.state.value,
            "uc1_input_type": self.input_type,
            "uc1_terminal": self.terminal,
            "allow_text_input": self.input_type == "text",  # Explicit UI control
        }
        if self.options:
            meta["uc1_options"] = self.options
        if self.llm_intent:
            meta["llm_intent"] = self.llm_intent.value
        if self.metadata:
            meta.update(self.metadata)
        
        chunks.append({"status": "meta", "chunk": meta})
        return chunks


class ConversationOrchestrator:
    """
    Central conversation controller for UC1 flow (TEXT-BLIND).
    
    ARCHITECTURE INVARIANT:
        Orchestrator controls FLOW, not LANGUAGE.
        LLM adapter controls LANGUAGE, not FLOW.
    
    The orchestrator:
    1. Validates input against current state
    2. Updates slots (deterministic)
    3. Determines next state (deterministic)
    4. Returns AdapterCallSpec (WHY to speak)
    
    The orchestrator NEVER:
    - Emits user-visible strings
    - Decides what words to use
    """
    
    # Class-level cache of orchestrator instances per session
    _instances: Dict[str, "ConversationOrchestrator"] = {}
    
    # Class-level singleton for config and validator (loaded once)
    _config: Optional[UC1Config] = None
    _config_validated: bool = False
    
    @classmethod
    def get_or_create(cls, session_id: str) -> "ConversationOrchestrator":
        """
        Get or create orchestrator for a session.
        
        This ensures one orchestrator per session.
        """
        if session_id not in cls._instances:
            cls._instances[session_id] = cls(session_id)
        return cls._instances[session_id]
    
    @classmethod
    def clear_session(cls, session_id: str) -> None:
        """Clear orchestrator instance for a session."""
        if session_id in cls._instances:
            del cls._instances[session_id]
            SlotManager.clear_session(session_id)
            logger.info(f"[Orchestrator] Cleared session: {session_id}")
    
    @classmethod
    def get_config(cls) -> UC1Config:
        """
        Get the UC1 config (loads and validates once).
        
        This is called at first use, not at import time.
        """
        if cls._config is None:
            cls._config = load_uc1_config()
            if not cls._config_validated:
                validator = UC1PolicyValidator()
                validator.validate(cls._config)
                cls._config_validated = True
        return cls._config
    
    def __init__(self, session_id: str):
        """
        Initialize orchestrator for a session.
        
        Args:
            session_id: The session identifier
        """
        self.session_id = session_id
        self.config = self.get_config()
        
        # Initialize components
        self.state_machine = UC1StateMachine()
        self.slot_manager = SlotManager(session_id)
        self.llm_adapter = ConstrainedLLMAdapter(self.config)
        self.sanitizer = LLMOutputSanitizer(self.config)
        self.policy = ConversationPolicy(session_id)
        self.button_manager = ButtonManager(self.config)  # Centralized button logic
        
        # Current state (default to ENTRY for new sessions)
        self._current_state = UC1State.ENTRY
        
        logger.info(f"[Orchestrator] Initialized for session: {session_id}")
    
    @property
    def current_state(self) -> UC1State:
        """Get current conversation state."""
        return self._current_state
    
    @property
    def slots(self) -> UC1Slots:
        """Get current slot values."""
        return self.slot_manager.slots
    
    def build_intent_gated_options(
        self, 
        state: UC1State, 
        intent: LLMIntent,
        bucket: CapabilityBucket = None,
        input_type: str = "text",
        dynamic_options: Optional[List[str]] = None
    ) -> List[str]:
        """
        DEPRECATED: Wrapper for ButtonManager.get_buttons_for_state().
        
        Kept for backward compatibility with chatbot_optimizer.py.
        All logic is now in button_manager.py.
        """
        return self.button_manager.get_buttons_for_state(
            state=state,
            slots=self.slots,
            bucket=bucket,
            dynamic_options=dynamic_options
        )
    
    def _get_exploration_buttons(self, bucket: CapabilityBucket = None) -> List[str]:
        """DEPRECATED: Wrapper for ButtonManager._get_exploration_buttons()."""
        return self.button_manager._get_exploration_buttons(bucket, self.slots)
    

    def process_input(self, user_input: str) -> OrchestratorResponse:
        """
        Process user input and return structured response.
        
        POLICY-GOVERNED: User intent overrides UC1 compliance.
        
        INPUT CLASSIFICATION GATE (Per UC1 Robustness Fixes):
            1. Classify input (GIBBERISH, NEGATION, ACK, QUESTION, STATEMENT)
            2. Block gibberish before it pollutes slots
            3. Force intents for control signals
        
        Args:
            user_input: The user's message (empty string for initial entry)
        
        Returns:
            OrchestratorResponse: The structured response for frontend
        """
        logger.info(f"[Orchestrator] Processing input in state {self._current_state.value}: '{user_input[:50] if user_input else '(empty)'}...'")
        
        # ============================================================
        # FROZEN CHECK - No mutations after bailout
        # ============================================================
        if self.slots.frozen:
            logger.warning("[Orchestrator] Slots frozen - checking for restart command")
            if user_input and user_input.strip().lower() == "restart":
                return self._handle_restart()
            return self._handle_frozen_state()
        
        # ============================================================
        # DB SYNC - Load user details already extracted by LLM
        # ============================================================
        # LLM extracts user_info → saves to DB via update_user_by_session
        # Sync from DB into UC1 slots to prevent repetitive name/email asks
        if not self.slots.user_name or not self.slots.user_email:
            try:
                from app.api.helpers import get_user_details_from_db
                db_details = get_user_details_from_db(self.session_id)
                if db_details:
                    if db_details.get("username") and not self.slots.user_name:
                        self.slot_manager.set_user_name(db_details["username"], caller="db_sync")
                        logger.info(f"[Orchestrator] Synced user_name from DB: {db_details['username']}")
                    if db_details.get("email") and not self.slots.user_email:
                        self.slot_manager.set_user_email(db_details["email"], caller="db_sync")
                        logger.info(f"[Orchestrator] Synced user_email from DB: {db_details['email']}")
            except Exception as e:
                logger.warning(f"[Orchestrator] DB sync failed (non-critical): {e}")
        
        # Track exchange count for policy decisions
        if user_input:
            self.slot_manager.increment_exchange()
        
        # ============================================================
        # INPUT CLASSIFICATION GATE (Pre-State-Machine)
        # ============================================================
        input_class = classify_input(user_input) if user_input else None
        
        # GIBBERISH GATE - Block garbage before it pollutes slots
        if input_class == InputClass.GIBBERISH:
            logger.info(f"[Orchestrator] GIBBERISH detected: '{user_input[:20]}...'")
            return self._handle_gibberish_input()
        
        # MAX-RETRY BAILOUT - Hard ceiling on retries
        if self.slots.retry_count >= 3:
            logger.info(f"[Orchestrator] Max retries ({self.slots.retry_count}) exceeded - bailout")
            return self._handle_max_retries()
        
        # ============================================================
        # NEGATION HANDLING - Scoped (clear only selected_alternative)
        # ============================================================
        if input_class == InputClass.NEGATION:
            logger.info(f"[Orchestrator] NEGATION detected: '{user_input}'")
            return self._handle_negation()
        
        # ============================================================
        # ACC PHASE 4: ACK BYPASS - Control signal, NOT language
        # ============================================================
        # ACK must never trigger clarification or probing.
        # ACK = noop or advance. LLM is NOT called for ACK.
        # ============================================================
        # ACC PHASE 4: ACK BYPASS - Control signal, NOT language
        # ============================================================
        # ACK must never trigger clarification or probing.
        # ACK = noop or advance. LLM is NOT called for ACK.
        #
        # MODIFICATION (2026-01-15): Removed strict ACK bypass to allow LLM to
        # handle contextual affirmatives (e.g., "Yes" to "Ready to explore?").
        # Simple "ok" will now flow to LLM, which can handle it naturally.
        #
        # if input_class == InputClass.ACK:
        #    logger.info(f"[ACC] ACK detected: '{user_input}' - bypassing LLM, control signal only")
        #    return self._handle_ack_bypass()

        
        # ============================================================
        # POLICY ENGINE CONSULTATION (User intent first)
        # ============================================================
        # Only consult policy when in UC1 flow (not ENTRY, not EXIT)
        is_in_uc1_flow = self._current_state not in (UC1State.ENTRY, UC1State.EXIT)
        
        if is_in_uc1_flow and user_input:
            # Create policy context
            session_context = {
                "exchange_count": self.slots.exchange_count,
                "is_in_uc1": True,
                "is_in_free_exploration": self._current_state == UC1State.FREE_EXPLORATION,
                "uc1_paused": self.slots.uc1_paused,
                "state_history": [],  # Could track this if needed
                "recent_inputs": [],
            }
            policy_context = create_policy_context(
                current_state=self._current_state.value,
                user_input=user_input,
                slots=self.slots.to_dict(),
                session_context=session_context,
            )
            
            # Evaluate policy
            decision = self.policy.evaluate(policy_context)
            logger.info(f"[Orchestrator] Policy decision: {decision.value}")
            
            # Act on policy decision
            if decision == PolicyDecision.PIVOT:
                # User wants to break out of funnel - enter FREE_EXPLORATION
                return self._enter_free_exploration(user_input)
            
            elif decision == PolicyDecision.SKIP:
                # User is resisting - skip current state if possible
                next_state = self.state_machine.get_default_next_state(self._current_state)
                if next_state and next_state != UC1State.EXIT:
                    logger.info(f"[Orchestrator] Skipping {self._current_state.value} -> {next_state.value}")
                    self._current_state = next_state
                    # Fall through to handle new state
            
            elif decision == PolicyDecision.RESUME_UC1:
                # User wants to resume UC1 from FREE_EXPLORATION
                if self._current_state == UC1State.FREE_EXPLORATION:
                    paused_state = self.slot_manager.resume_uc1()
                    self._current_state = UC1State(paused_state) if paused_state else UC1State.CAPABILITY_SELECTION
                    logger.info(f"[Orchestrator] Resuming UC1 at {self._current_state.value}")
            
            elif decision == PolicyDecision.CAPTURE_LEAD:
                # Signal-based lead capture (handled elsewhere, just log)
                logger.info("[Orchestrator] Lead capture signal detected")
            
            # PolicyDecision.PROCEED and SUPPRESS_ASK fall through to normal handling
        
        # ============================================================
        # HANDLE FREE_EXPLORATION MODE
        # ============================================================
        if self._current_state == UC1State.FREE_EXPLORATION:
            return self._handle_free_exploration(user_input)
        
        # ============================================================
        # NORMAL STATE HANDLING
        # ============================================================
        if self._current_state == UC1State.ENTRY:
            return self._handle_entry()
        elif self._current_state == UC1State.CAPABILITY_SELECTION:
            return self._handle_capability_selection(user_input)
        elif self._current_state == UC1State.CONTEXT_QUESTION:
            return self._handle_context_question(user_input)
        elif self._current_state == UC1State.NAME_CAPTURE:
            return self._handle_name_capture(user_input)
        elif self._current_state == UC1State.EXPLORATION_LAYER:
            return self._handle_exploration_layer(user_input)
        elif self._current_state == UC1State.AI_SYNTHESIS:
            return self._handle_ai_synthesis()
        elif self._current_state == UC1State.CONSULTATIVE_ALTERNATIVES:
            return self._handle_consultative_alternatives(user_input)
        elif self._current_state == UC1State.RECOMMENDATION:
            return self._handle_recommendation(user_input)
        elif self._current_state == UC1State.EMAIL_CAPTURE:
            return self._handle_email_capture(user_input)
        elif self._current_state == UC1State.EXIT:
            return self._handle_exit()
        else:
            logger.error(f"[Orchestrator] Unknown state: {self._current_state}")
            bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.EXIT,
                    response_intent=ResponseIntent.EXIT,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                terminal=True
            )
    
    def _enter_free_exploration(self, user_input: str) -> OrchestratorResponse:
        """
        Enter FREE_EXPLORATION mode - user breaks out of UC1 funnel.
        
        UC1 is PAUSED, not abandoned. User can resume via explicit intent.
        
        NEW (2026-01-15): Delegates to _handle_free_exploration which uses
        the ExplorerAgent for smart, lead-generative responses.
        """
        # Pause UC1 state (can resume later)
        if self._current_state != UC1State.FREE_EXPLORATION:
            self.slot_manager.pause_uc1(self._current_state.value)
        
        self._current_state = UC1State.FREE_EXPLORATION
        logger.info(f"[Orchestrator] Entered FREE_EXPLORATION mode, paused at {self.slots.paused_state}")
        
        # Delegate to _handle_free_exploration which has agent logic
        return self._handle_free_exploration(user_input)
    
    def _handle_free_exploration(self, user_input: str) -> OrchestratorResponse:
        """
        Handle FREE_EXPLORATION state - unstructured user-driven conversation.
        
        NEW (2026-01-15): Uses ExplorerAgent for smart, lead-generative responses
        when USE_AGENT_EXPLORATION=true.
        
        NEW (2026-01-20): CTA INTERCEPTION - Route high-intent CTAs to EMAIL_CAPTURE.
        
        RULES:
        1. INTERCEPT CTAs before agent processing
        2. Agent answers questions with KB search
        3. Agent gathers slots naturally (name, email)
        4. Resume UC1 only on EXPLICIT user intent (detected by policy engine)
        
        STABILIZER:
        After 2+ unclear/gibberish inputs, stabilize with anchor-derived response.
        """
        import os
        use_agent = os.getenv("USE_AGENT_EXPLORATION", "false").lower() == "true"
        
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # =========================================================================
        # CTA INTERCEPTION (Fix for "Talk to expert" going to agent as text)
        # =========================================================================
        # Check if user clicked a CTA button - route to EMAIL_CAPTURE if high-intent
        if user_input:
            # First check for exact button click match
            button_result = self.button_manager.is_button_click(user_input, bucket)
            selected_cta = None
            
            if button_result and button_result[0] == "cta":
                cta_action = button_result[1]
                logger.info(f"[Orchestrator] FREE_EXPLORATION: CTA button clicked: {user_input} -> {cta_action}")
                
                # Find matching CTA by choice text
                for cta in self.config.exit_ctas:
                    if cta.choice.lower() == user_input.strip().lower():
                        selected_cta = cta
                        break
            
            # Also check for phrase-based CTA intent (typed requests)
            if not selected_cta:
                input_lower = user_input.strip().lower()
                # High-intent phrases that should trigger CTA flow
                HIGH_INTENT_PHRASES = [
                    ("talk to expert", "calendar"),
                    ("talk to an expert", "calendar"),
                    ("speak to someone", "calendar"),
                    ("schedule a call", "calendar"),
                    ("schedule call", "calendar"),
                    ("book a call", "calendar"),
                    ("meet the team", "calendar"),
                    ("discuss my requirement", "UC2"),
                    ("discuss requirement", "UC2"),
                    ("get a consultation", "UC2"),
                    ("talk to consultant", "calendar"),
                    ("talk to architect", "calendar"),
                    ("talk to devops team", "calendar"),
                ]
                
                for phrase, outcome in HIGH_INTENT_PHRASES:
                    if phrase in input_lower:
                        logger.info(f"[Orchestrator] FREE_EXPLORATION: CTA phrase detected: '{phrase}' -> {outcome}")
                        # Find matching CTA by outcome
                        for cta in self.config.exit_ctas:
                            if cta.outcome == outcome:
                                selected_cta = cta
                                break
                        break
            
            if selected_cta:
                # Store the CTA outcome
                self.slot_manager.set_selected_cta_outcome(selected_cta.outcome, caller="orchestrator")
                
                # High-intent CTAs (UC2, calendar) -> EMAIL_CAPTURE first
                if selected_cta.outcome in ("UC2", "calendar"):
                    if not self.slots.user_email:
                        logger.info(f"[Orchestrator] High-intent CTA without email -> EMAIL_CAPTURE")
                        self._current_state = UC1State.EMAIL_CAPTURE
                        return self._handle_email_capture("")
                    else:
                        # Email already captured -> EXIT
                        logger.info(f"[Orchestrator] High-intent CTA with email -> EXIT")
                        self._current_state = UC1State.EXIT
                        return self._handle_exit()
                elif selected_cta.outcome == "loop":
                    # Continue exploring -> back to CAPABILITY_SELECTION
                    self._current_state = UC1State.CAPABILITY_SELECTION
                    return self._handle_capability_selection("")
                else:
                    # Exit outcome -> EXIT
                    self._current_state = UC1State.EXIT
                    return self._handle_exit()
        
        # Classify input for stabilizer logic
        input_class = classify_input(user_input) if user_input else None
        
        # Track unclear inputs for stabilizer
        if input_class in (InputClass.GIBBERISH, InputClass.ACK):
            unclear_count = self.slot_manager.increment_free_exploration_unclear()
            if unclear_count >= 2:
                logger.info(f"[Orchestrator] FREE_EXPLORATION stabilizer triggered after {unclear_count} unclear inputs")
                return self._stabilize_free_exploration()
        else:
            # Valid input - reset unclear count
            self.slot_manager.reset_free_exploration_unclear()
        
        # Track user message
        if user_input:
            self.slot_manager.set_last_user_message(user_input, caller="orchestrator")
        
        # Get dynamic exploration buttons for this topic
        exploration_options = self._get_exploration_buttons(bucket)
        
        # =========================================================================
        # AGENT MODE: Use ExplorerAgent for smart, lead-generative responses
        # =========================================================================
        if use_agent:
            return self._handle_free_exploration_with_agent(user_input, bucket, exploration_options)
        
        # Traditional mode: Use LLM adapter directly
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.FREE_EXPLORATION,
                response_intent=ResponseIntent.PROMPT,  # Answer freely, no qualifying
                user_input=user_input,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            options=exploration_options,
            metadata={"mode": "free_exploration", "paused_state": self.slots.paused_state}
        )
    
    def _handle_free_exploration_with_agent(
        self, 
        user_input: str, 
        bucket: CapabilityBucket,
        exploration_options: List[str]
    ) -> OrchestratorResponse:
        """
        Handle FREE_EXPLORATION using the ReAct ExplorerAgent.
        
        The agent:
        - Uses search_knowledge_base for domain-specific answers
        - Captures slots naturally (name, email, context)
        - Calculates lead score for prioritization
        """
        try:
            from app.agents.explorer_agent import get_explorer_agent
            
            agent = get_explorer_agent()
            
            # Invoke agent with current context (including shared URLs to filter)
            result = agent.invoke(
                user_input=user_input,
                session_id=self.session_id,
                initial_slots={
                    "user_name": self.slots.user_name,
                    "user_email": self.slots.user_email,
                    "context_signal": self.slots.context_signal,
                    "capability_bucket": self.slots.capability_bucket,
                    "shared_urls": self.slot_manager.get_shared_urls(),  # Pass for filtering
                }
            )
            
            # Extract response
            response_text = agent.get_response_text(result)
            is_ready = result.get("is_ready", False)
            lead_score = result.get("lead_score", 0)
            
            # ============================================================
            # TRACK URLS IN RESPONSE (Prevents repetitive sharing)
            # ============================================================
            from app.agents.tools.rag_search import URL_PATTERN
            found_urls = URL_PATTERN.findall(response_text.lower())
            for url in found_urls:
                self.slot_manager.add_shared_url(url)
            
            # Sync any slots captured by agent
            if result.get("slots"):
                if result["slots"].get("user_name") and not self.slots.user_name:
                    self.slot_manager.set_user_name(result["slots"]["user_name"], caller="orchestrator")
                if result["slots"].get("user_email") and not self.slots.user_email:
                    self.slot_manager.set_user_email(result["slots"]["user_email"], caller="orchestrator")
                if result["slots"].get("context_signal") and not self.slots.context_signal:
                    self.slot_manager.set_context_signal(result["slots"]["context_signal"], caller="orchestrator")
            
            logger.info(f"[Orchestrator] Agent exploration: ready={is_ready}, score={lead_score}")
            
            # If user is ready, show CTAs
            if is_ready:
                self.slot_manager.mark_exploration_complete()
                exploration_options = [cta.choice for cta in self.config.exit_ctas]
            
            # Check if we should ask for email (progressive lead capture)
            email_prompt = ""
            if self.config.email_capture and not self.slots.user_email:
                min_turns = self.config.email_capture.min_turns_before_ask
                if self.slot_manager.should_ask_for_email(min_turns=min_turns):
                    email_prompt = f"\n\n{self.config.email_capture.soft_prompt}"
                    self.slot_manager.mark_email_asked()
                    logger.info("[Orchestrator] Appending email capture prompt")
            
            # ============================================================
            # GENERATE DYNAMIC OPTIONS FROM RESPONSE (Not static config!)
            # ============================================================
            # Use LLM adapter's fallback generator to extract next steps from response
            dynamic_options = self.llm_adapter._generate_fallback_options(response_text, user_input)
            logger.info(f"[Orchestrator] Generated dynamic options from agent response: {dynamic_options}")
            
            # Return agent response with DYNAMIC options
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.FREE_EXPLORATION,
                    response_intent=ResponseIntent.PROMPT,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message=response_text + email_prompt,  # Agent already generated the text
                options=dynamic_options,  # DYNAMIC options from response, not static config!
                metadata={
                    "mode": "agent_exploration", 
                    "lead_score": lead_score,
                    "paused_state": self.slots.paused_state
                }
            )
            
        except Exception as e:
            logger.error(f"[Orchestrator] Agent exploration failed: {e}, falling back to traditional")
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.FREE_EXPLORATION,
                    response_intent=ResponseIntent.PROMPT,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                options=exploration_options,
                metadata={"mode": "free_exploration", "paused_state": self.slots.paused_state}
            )

    
    def _handle_entry(self) -> OrchestratorResponse:
        """Handle ENTRY state - transition to capability selection."""
        # Transition to CAPABILITY_SELECTION
        self._current_state = UC1State.CAPABILITY_SELECTION
        button_options = [bucket.trigger for bucket in self.config.capability_buckets]
        
        # Use FIXED entry_message from config - no LLM generation
        # Per spec: "Great — happy to guide you. Pick the closest area and I'll narrow it down from there."
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.ENTRY,
                response_intent=ResponseIntent.PROMPT,
                slots=self.slots,
            ),
            input_type="buttons",
            options=button_options,
            message=self.config.entry_message.strip(),  # Fixed message from config
        )
    
    def _handle_capability_selection(self, user_input: str) -> OrchestratorResponse:
        """Handle CAPABILITY_SELECTION state - user selects a capability bucket."""
        bucket = get_bucket_by_trigger(self.config, user_input)
        
        if not bucket:
            # Invalid selection - retry
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            button_options = [b.trigger for b in self.config.capability_buckets]
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.CAPABILITY_SELECTION,
                    response_intent=ResponseIntent.RETRY,
                    user_input=user_input,
                    slots=self.slots,
                ),
                input_type="buttons",
                options=button_options,
            )
        
        # Valid selection
        self.slot_manager.set_capability_bucket(bucket.id, caller="orchestrator")
        self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
        # Transition to CONTEXT_QUESTION
        self._current_state = UC1State.CONTEXT_QUESTION
        
        # FIXED PROMPT: Emit context_question directly, NO LLM
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.CONTEXT_QUESTION,
                response_intent=ResponseIntent.PROMPT,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message=bucket.context_question,  # Fixed from config
        )
    
    def _handle_context_question(self, user_input: str) -> OrchestratorResponse:
        """
        Handle CONTEXT_QUESTION state - user provides context answer.
        
        VALIDATION: Must pass BEFORE slot mutation.
        FIXED PROMPT: Emits exact config question on retry (no LLM).
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # ============================================================
        # VALIDATE BEFORE SLOT MUTATION (per micro-correction #3)
        # ============================================================
        is_valid, reason = validate_context_answer(user_input)
        
        if not is_valid:
            # Invalid input - retry with FIXED prompt
            self.slot_manager.increment_retry()
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            logger.info(f"[Orchestrator] Context answer validation failed: {reason}")
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.CONTEXT_QUESTION,
                    response_intent=ResponseIntent.RETRY,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message=bucket.context_question if bucket else "Could you tell me more about what you're looking for?",  # FIXED prompt
            )
        
        # VALID: Now safe to mutate slots
        self.slot_manager.set_context_signal(user_input, caller="orchestrator")
        self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        self.slot_manager.reset_retry_count()
        
        # Transition to NAME_CAPTURE
        self._current_state = UC1State.NAME_CAPTURE
        
        # FIXED PROMPT: Emit name_capture_prompt directly, NO LLM
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.NAME_CAPTURE,
                response_intent=ResponseIntent.TRANSITION,
                user_input=user_input,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message=self.config.name_capture_prompt,  # Fixed from config
        )
    
    def _handle_name_capture(self, user_input: str) -> OrchestratorResponse:
        """
        Handle NAME_CAPTURE state - user provides their name.
        
        VALIDATION: Must pass BEFORE slot mutation (per micro-correction #3).
        FIXED PROMPT: Emits exact config prompt on retry (no LLM).
        
        Rules:
            1. Write-once: Skip if name already captured
            2. Validate BEFORE slot mutation
            3. Max 3 retries before bailout (handled at process_input level)
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # ============================================================
        # WRITE-ONCE GUARD - Skip if name already set
        # ============================================================
        if self.slots.user_name:
            logger.info(f"[Orchestrator] Name already captured: {self.slots.user_name} - skipping")
            self._current_state = UC1State.AI_SYNTHESIS
            return self._handle_ai_synthesis()
        
        # ============================================================
        # VALIDATE BEFORE SLOT MUTATION (per micro-correction #3)
        # ============================================================
        is_valid, reason = validate_name(user_input)
        
        # Check if user provided EMAIL instead of NAME (Validation Fix)
        if not is_valid:
            is_email, _ = validate_email(user_input)
            if is_email:
                # User provided email. Accept it.
                self.slot_manager.set_user_email(user_input, caller="orchestrator")
                # Infer name from email local part
                inferred_name = user_input.split("@")[0]
                # Filter special chars from name
                inferred_name = ''.join(c for c in inferred_name if c.isalpha())
                if inferred_name:
                    inferred_name = inferred_name.capitalize()
                else:
                    inferred_name = "There" # Fallback if email is like 123@...
                
                # We do NOT set name here to let the standard flow below handle it with the inferred name
                # But wait, logic below sets user_input. So let's override user_input
                user_input = inferred_name
                is_valid = True 
                reason = ""
                logger.info(f"[Orchestrator] Captured email in name state: {self.slots.user_email} -> Name: {user_input}")

        if not is_valid:
            # Invalid input - retry with FIXED prompt
            self.slot_manager.increment_retry()
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            logger.info(f"[Orchestrator] Name validation failed: {reason} for input '{user_input}'")
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.NAME_CAPTURE,
                    response_intent=ResponseIntent.RETRY,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message="Just your name is enough — for example, Vinay.",  # FIXED prompt
            )
        
        # VALID: Now safe to mutate slots
        self.slot_manager.set_user_name(user_input.strip(), caller="orchestrator")
        self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        self.slot_manager.reset_retry_count()  # Successful capture
        
        # Transition to AI_SYNTHESIS (S4) - synthesis comes BEFORE exploration per spec
        self._current_state = UC1State.AI_SYNTHESIS
        
        # Return spec for synthesis + auto-advance to exploration
        return self._handle_ai_synthesis()

    
    def _handle_exploration_layer(self, user_input: str) -> OrchestratorResponse:
        """
        Handle EXPLORATION_LAYER state - dynamic Q&A with optional Agent delegation.
        
        FEATURE FLAG: If USE_AGENT_EXPLORATION=true, delegates to ExplorerAgent.
        Otherwise uses the traditional fixed-turn exploration.
        """
        import os
        use_agent = os.getenv("USE_AGENT_EXPLORATION", "false").lower() == "true"
        
        if use_agent:
            return self._handle_exploration_with_agent(user_input)
        
        return self._handle_exploration_traditional(user_input)
    
    def _handle_exploration_with_agent(self, user_input: str) -> OrchestratorResponse:
        """
        Delegate exploration to the ReAct Explorer Agent.
        
        The agent handles dynamic Q&A, slot gathering, and readiness detection.
        """
        try:
            from app.agents.explorer_agent import get_explorer_agent
            
            agent = get_explorer_agent()
            bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
            
            # Invoke agent with current context (including shared URLs to filter)
            result = agent.invoke(
                user_input=user_input,
                session_id=self.session_id,
                initial_slots={
                    "user_name": self.slots.user_name,
                    "context_signal": self.slots.context_signal,
                    "capability_bucket": self.slots.capability_bucket,
                    "shared_urls": self.slot_manager.get_shared_urls(),  # Pass for filtering
                }
            )
            
            # Extract response
            response_text = agent.get_response_text(result)
            is_ready = result.get("is_ready", False)
            lead_score = result.get("lead_score", 0)
            
            # ============================================================
            # TRACK URLS IN RESPONSE (Prevents repetitive sharing)
            # ============================================================
            from app.agents.tools.rag_search import URL_PATTERN
            found_urls = URL_PATTERN.findall(response_text.lower())
            for url in found_urls:
                self.slot_manager.add_shared_url(url)
            
            # Sync any slots captured by agent
            if result.get("slots"):
                if result["slots"].get("user_name"):
                    self.slot_manager.set_user_name(result["slots"]["user_name"], caller="orchestrator")
                if result["slots"].get("user_email"):
                    # Validate email from agent before syncing
                    is_email_valid, _ = validate_email(result["slots"]["user_email"])
                    if is_email_valid:
                        self.slot_manager.set_user_email(result["slots"]["user_email"], caller="orchestrator")
                if result["slots"].get("context_signal"):
                    self.slot_manager.set_context_signal(result["slots"]["context_signal"], caller="orchestrator")
            
            logger.info(f"[Orchestrator] Agent exploration: ready={is_ready}, score={lead_score}")
            
            # Check if ready for options
            if is_ready:
                self.slot_manager.mark_exploration_complete()
                self._current_state = UC1State.CONSULTATIVE_ALTERNATIVES
                return self._handle_consultative_alternatives("")
            
            # Return agent response
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.EXPLORATION_LAYER,
                    response_intent=ResponseIntent.REFLECT,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message=response_text,  # Agent already generated the text
                metadata={"agent_mode": True, "lead_score": lead_score}
            )
            
        except Exception as e:
            logger.error(f"[Orchestrator] Agent exploration failed: {e}, falling back to traditional")
            return self._handle_exploration_traditional(user_input)
    
    def _handle_exploration_traditional(self, user_input: str) -> OrchestratorResponse:
        """
        Traditional fixed-turn exploration (fallback when agent disabled).
        
        AUTHORITATIVE FLAG: If exploration_complete is True, route to alternatives.
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        if not bucket:
            logger.error(f"[Orchestrator] Bucket not found in exploration")
            self._current_state = UC1State.AI_SYNTHESIS
            return self._handle_ai_synthesis()
        
        # ============================================================
        # EXPLORATION AUTHORITY - Never re-enter if complete
        # ============================================================
        if self.slots.exploration_complete:
            logger.info("[Orchestrator] Exploration already complete - routing to alternatives")
            if not self.slots.alternatives_consumed:
                self._current_state = UC1State.CONSULTATIVE_ALTERNATIVES
                return self._handle_consultative_alternatives("")
            else:
                self._current_state = UC1State.RECOMMENDATION
                return self._handle_recommendation("")
        
        current_turn = self.slots.exploration_turn
        max_turns = 2
        
        # User provided a response - process it
        if user_input:
            self.slot_manager.add_exploration_response(user_input)
            self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
            self.slot_manager.reset_retry_count()  # Successful progression
        
        # Check if we've completed enough exploration turns
        if current_turn >= max_turns and user_input:
            # Mark exploration as COMPLETE (irreversible)
            self.slot_manager.mark_exploration_complete()
            logger.info(f"[Orchestrator] Exploration complete after {current_turn} turns")
            self._current_state = UC1State.CONSULTATIVE_ALTERNATIVES
            return self._handle_consultative_alternatives("")
        
        # Generate next exploration prompt/reflect
        next_turn = current_turn + 1 if user_input else current_turn
        self.slot_manager.set_exploration_turn(next_turn, caller="orchestrator")
        
        # Return spec - LLM adapter will generate appropriate question/reflection
        intent = ResponseIntent.REFLECT if user_input else ResponseIntent.PROMPT
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.EXPLORATION_LAYER,
                response_intent=intent,
                user_input=user_input,
                slots=self.slots,
                bucket=bucket,
                exploration_turn=next_turn,
            ),
            input_type="text",
        )


    
    def _handle_ai_synthesis(self) -> OrchestratorResponse:
        """Handle AI_SYNTHESIS state (S4) - generate synthesis, then go to Exploration."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        if not bucket:
            logger.error(f"[Orchestrator] Bucket not found in synthesis")
            return OrchestratorResponse(
                state=UC1State.EXIT,
                call_spec=AdapterCallSpec(
                    state=UC1State.EXIT,
                    response_intent=ResponseIntent.EXIT,
                    slots=self.slots,
                ),
                input_type="text",  # Allow re-engagement even in error cases
                terminal=True
            )
        
        # AI Synthesis (S4) leads to Exploration Layer (S5)
        # The LLM generates synthesis, then we transition to exploration
        self._current_state = UC1State.EXPLORATION_LAYER
        self.slot_manager.set_exploration_turn(1, caller="orchestrator")
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.AI_SYNTHESIS,
                response_intent=ResponseIntent.ACKNOWLEDGE,  # Synthesis + acknowledge user
                user_input=self.slots.user_name,  # Pass name for personalized synthesis
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",  # User will provide exploration input
        )
    
    def _handle_consultative_alternatives(self, user_input: str) -> OrchestratorResponse:
        """Handle CONSULTATIVE_ALTERNATIVES state (S6) - present 3 alternatives."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # If called after exploration complete (empty input), present alternatives
        if not user_input or not user_input.strip():
            # Clear any previous selection to avoid "Stuck CTA" loop (fixes "buttons not working" bug)
            self.slot_manager.set_selected_alternative(None, caller="orchestrator")
            
            # NOTE: Do NOT set options here - let chatbot_optimizer.py call ButtonManager
            # with dynamic_options from LLM. ButtonManager will use dynamic options if available,
            # falling back to bucket.alternatives only if LLM didn't generate any.
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.CONSULTATIVE_ALTERNATIVES,
                    response_intent=ResponseIntent.PRESENT,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",  # Allow typing, buttons added by chatbot_optimizer
                options=None,  # Let ButtonManager decide with dynamic options priority
            )
        
        # Track user selection and mark alternatives consumed
        if bucket:
            alternatives_lower = [a.lower() for a in bucket.alternatives]
            input_lower = user_input.strip().lower()
            
            if input_lower in alternatives_lower:
                self.slot_manager.set_selected_alternative(user_input, caller="orchestrator")
                self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
            else:
                self.slot_manager.set_selected_alternative(user_input, caller="orchestrator")
                self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
            
            # Mark alternatives as consumed (authoritative - prevents re-show)
            self.slot_manager.mark_alternatives_consumed()
        
        # Transition to RECOMMENDATION
        self._current_state = UC1State.RECOMMENDATION
        button_options = [cta.choice for cta in self.config.exit_ctas]
        outcome_map = {cta.choice: cta.outcome for cta in self.config.exit_ctas}
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.RECOMMENDATION,
                response_intent=ResponseIntent.PRESENT,
                user_input=user_input,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="buttons",
            options=button_options,
            metadata={"cta_outcomes": outcome_map}
        )
    
    def _handle_recommendation(self, user_input: str) -> OrchestratorResponse:
        """Handle RECOMMENDATION state - user selects a CTA."""
        # Find matching CTA
        selected_cta = None
        for cta in self.config.exit_ctas:
            if cta.choice.lower() == user_input.strip().lower():
                selected_cta = cta
                break
        
        if not selected_cta:
            # Invalid CTA - retry
            bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            button_options = [cta.choice for cta in self.config.exit_ctas]
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.RECOMMENDATION,
                    response_intent=ResponseIntent.RETRY,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="buttons",
                options=button_options,
            )
        
        # Valid CTA
        self.slot_manager.set_selected_cta_outcome(selected_cta.outcome, caller="orchestrator")
        self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
        # Get bucket for both branches (was missing - caused undefined bucket in loop-back)
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Determine next state based on CTA outcome
        next_state = self.state_machine.get_next_state_for_cta(
            self._current_state, selected_cta.outcome
        )
        
        if next_state == UC1State.CAPABILITY_SELECTION:
            # Loop back - user wants to continue exploring
            self._current_state = UC1State.CAPABILITY_SELECTION
            button_options = [b.trigger for b in self.config.capability_buckets]
            
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.CAPABILITY_SELECTION,
                    response_intent=ResponseIntent.TRANSITION,
                    user_input=user_input,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="buttons",
                options=button_options,
            )
        elif next_state == UC1State.EMAIL_CAPTURE:
            # Move to EMAIL_CAPTURE
            self._current_state = UC1State.EMAIL_CAPTURE
            return self._handle_email_capture("")  # Trigger prompt with empty input
        else:
            # Exit flow
            self._current_state = UC1State.EXIT
            return self._handle_exit()

    def _handle_email_capture(self, user_input: str) -> OrchestratorResponse:
        """
        Handle EMAIL_CAPTURE state.
        
        Triggered when high-intent CTA is selected but email is missing.
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # 1. Skip if already captured (write-once)
        if self.slots.user_email:
            logger.info(f"[Orchestrator] Email already captured ({self.slots.user_email}), skipping capture")
            self._current_state = UC1State.EXIT
            return self._handle_exit()

        # 2. If prompt (empty input or first entry), return prompt
        if not user_input:
            prompt = self.config.email_capture.prompt if self.config.email_capture else "What's the best email to reach you?"
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.EMAIL_CAPTURE,
                    response_intent=ResponseIntent.PROMPT,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message=prompt
            )
            
        # 3. Validate input
        is_valid, reason = validate_email(user_input)
        
        if not is_valid:
            # Retry logic
            self.slot_manager.increment_retry()
            # Construct retry logic
            prompt = self.config.email_capture.prompt if self.config.email_capture else "What's the best email to reach you?"
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.EMAIL_CAPTURE,
                    response_intent=ResponseIntent.RETRY,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message=f"Please provide a valid email format (e.g., name@company.com). {prompt}"
            )
        
        # 4. Valid -> Save -> Exit
        self.slot_manager.set_user_email(user_input, caller="orchestrator")
        # Infer name if missing 
        if not self.slots.user_name:
            inferred = user_input.split("@")[0].capitalize()
            self.slot_manager.set_user_name(inferred, caller="orchestrator")
        
        logger.info(f"[Orchestrator] Email captured: {user_input}")
        
        self._current_state = UC1State.EXIT
        return self._handle_exit()

    def _handle_exit(self) -> OrchestratorResponse:
        """Handle EXIT state (TEXT-BLIND).
        
        EXIT is a flow-completion state, NOT a session termination.
        After sending the exit message:
        1. terminal=False keeps the chat input enabled
        2. State resets to ENTRY so the next message starts fresh
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Generate message DETERMINISTICALLY (no LLM)
        exit_message = self.llm_adapter.generate_exit_summary(self.slots, bucket)
        
        # Build response BEFORE resetting state (response references current EXIT state)
        response = OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.EXIT,
                response_intent=ResponseIntent.GRACEFUL_EXIT,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",  # Allow re-engagement after goodbye
            terminal=False,  # EXIT != TERMINATION - keep chat open
            message=exit_message
        )
        
        # LIFECYCLE RESET: Prepare for next user input
        # This ensures the NEXT call to process_input starts from ENTRY
        # (Does NOT affect current turn - we return immediately after this)
        self._current_state = UC1State.ENTRY
        self.slot_manager.clear()  # Clear UC1 context for clean restart
        
        logger.info(f"[Orchestrator] EXIT complete, state reset to ENTRY for re-engagement")
        
        return response
    
    # ============================================================
    # ROBUSTNESS HANDLERS (Per UC1 Fixes - 2026-01-12)
    # ============================================================
    
    def _handle_gibberish_input(self) -> OrchestratorResponse:
        """
        Handle gibberish input (pre-state-machine rejection).
        
        Does NOT count as retry. Does NOT advance state.
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=self._current_state,
                response_intent=ResponseIntent.RETRY,
                user_input="gibberish_detected",
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message="I might be missing that — could you rephrase in a sentence?",
        )
    
    def _handle_max_retries(self) -> OrchestratorResponse:
        """
        Handle max retries exceeded (bailout).
        
        Freezes slots. User must type 'restart' to begin again.
        """
        self.slot_manager.freeze_slots()
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=self._current_state,
                response_intent=ResponseIntent.GRACEFUL_EXIT,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message="Looks like we're not aligned right now. You can type 'restart' anytime to begin fresh.",
            terminal=False,  # Keep chat open for restart
        )
    
    def _handle_negation(self) -> OrchestratorResponse:
        """
        Handle negation (state-aware).
        
        HYBRID APPROACH (2026-01-19):
        - NAME_CAPTURE: Skip to AI_SYNTHESIS, mark name_declined (re-offer at CTA)
        - Other states: Clear selected_alternative, continue exploration
        
        Preserves context_signal, capability_bucket.
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # ============================================================
        # STATE-AWARE NEGATION: NAME_CAPTURE gets skipped, not looped
        # ============================================================
        if self._current_state == UC1State.NAME_CAPTURE:
            # User declined to give name - respect that, continue funnel
            self.slot_manager.mark_name_declined()
            self._current_state = UC1State.AI_SYNTHESIS
            logger.info("[Orchestrator] Name declined - skipping to AI_SYNTHESIS")
            
            # Build user-friendly message
            if bucket:
                message = f"No problem! Let's continue exploring {bucket.trigger}."
            else:
                message = "No problem! Let's continue."
            
            # Proceed to AI synthesis (which will auto-advance to exploration)
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.AI_SYNTHESIS,
                    response_intent=ResponseIntent.TRANSITION,
                    user_input="user_declined_name",
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
                message=message,
            )
        
        # ============================================================
        # OTHER STATES: Clear selection, continue exploration
        # ============================================================
        # Clear only the selection (scoped)
        self.slot_manager.slots.selected_alternative = None
        self.slot_manager._safe_persist()
        
        # Build user-friendly response (NOT using internal IDs)
        if bucket:
            # Use the bucket's trigger text which is what the user clicked on
            message = f"No worries. You're exploring {bucket.trigger}. What aspect would you like to know more about?"
        elif self.slots.context_signal:
            # Fallback to context signal if available
            context_preview = self.slots.context_signal[:40] + "..." if len(self.slots.context_signal) > 40 else self.slots.context_signal
            message = f"Alright. Let's refocus on your goal: \"{context_preview}\". What would you like to explore?"
        else:
            message = "Alright. What would you like to explore instead?"
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=self._current_state,
                response_intent=ResponseIntent.PROMPT,
                user_input="user_declined",
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message=message,
        )
    
    def _handle_frozen_state(self) -> OrchestratorResponse:
        """Handle input when slots are frozen (after bailout)."""
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=self._current_state,
                response_intent=ResponseIntent.RETRY,
                slots=self.slots,
            ),
            input_type="text",
            message="Type 'restart' to begin a fresh conversation.",
            terminal=False,
        )
    
    def _handle_ack_bypass(self) -> OrchestratorResponse:
        """
        Handle ACK bypass (control signal only).
        
        ACC INVARIANT: ACK never triggers LLM.
        It advances state if appropriate, or waits.
        """

        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Build user-friendly message (NOT using internal anchor with IDs)
        if self._current_state in (UC1State.EXPLORATION_LAYER, UC1State.FREE_EXPLORATION):
            # Use bucket trigger for user-friendly context
            if bucket:
                message = f"Got it. You're exploring {bucket.trigger}. What else would you like to know?"
            elif self.slots.user_name:
                message = f"Got it, {self.slots.user_name}. What else is on your mind?"
            else:
                message = "Got it. What else is on your mind?"
                
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=self._current_state,
                    response_intent=ResponseIntent.ACKNOWLEDGE,
                    user_input="ack_bypass",
                    slots=self.slots, # Pass slots even if no LLM call
                    bucket=bucket,
                ),
                input_type="text",
                message=message,
                terminal=False,
            )
            
        # Default: just a simple acknowledgment
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=self._current_state,
                response_intent=ResponseIntent.ACKNOWLEDGE,
                user_input="ack_bypass",
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message="Got it.",
            terminal=False,
        )

    def _handle_restart(self) -> OrchestratorResponse:
        """Handle restart command (clears all state)."""
        logger.info("[Orchestrator] RESTART command received - clearing session")
        self.slot_manager.clear()
        self._current_state = UC1State.ENTRY
        
        # Reinitialize slot manager
        self.slot_manager = SlotManager(self.session_id)
        
        return self._handle_entry()
    
    def _stabilize_free_exploration(self) -> OrchestratorResponse:
        """
        Stabilize FREE_EXPLORATION after 2+ unclear inputs.
        
        Uses user-friendly bucket trigger, not internal anchor IDs.
        """
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Build user-friendly reset message
        if bucket:
            message = f"Let's refocus. You were exploring {bucket.trigger}. What specific aspect should we look at first?"
        elif self.slots.user_name:
            message = f"Let's reset, {self.slots.user_name}. What are you looking to accomplish?"
        else:
            message = "Let's reset. What are you looking to accomplish?"
        
        # Reset unclear count
        self.slot_manager.reset_free_exploration_unclear()
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.FREE_EXPLORATION,
                response_intent=ResponseIntent.PROMPT,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
            message=message,
        )
    
    async def process_input_stream(
        self,
        user_input: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process input and yield SSE-compatible chunks.
        
        This is the streaming version for integration with
        the existing chatbot_optimizer streaming infrastructure.
        
        Args:
            user_input: The user's message
        
        Yields:
            Dict: SSE-compatible event chunks
        """
        response = self.process_input(user_input)
        
        for chunk in response.to_sse_chunks():
            yield chunk
