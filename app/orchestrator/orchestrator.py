# UC1 Conversation Orchestrator - Central Controller
#
# ARCHITECTURE RULE: This is the SOLE OWNER of UC1 conversation flow.
# - LLM NEVER decides state transitions
# - All flow decisions are made HERE based on state machine
# - Slot management is coordinated HERE
# - LLM adapter is called for paraphrasing ONLY
#
# The orchestrator coordinates:
# 1. State machine transitions
# 2. Slot updates
# 3. LLM paraphrasing
# 4. Output sanitization
# 5. Response generation

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal, AsyncGenerator
from app.orchestrator.uc1_config import UC1Config, load_uc1_config, get_bucket_by_id, get_bucket_by_trigger
from app.orchestrator.state_machine import UC1State, UC1StateMachine
from app.orchestrator.slot_manager import UC1Slots, SlotManager, EngagementEvent
from app.orchestrator.llm_adapter import ConstrainedLLMAdapter
from app.orchestrator.output_sanitizer import LLMOutputSanitizer
from app.orchestrator.policy_validator import UC1PolicyValidator
from app.logger import get_logger

logger = get_logger("orchestrator")


# Type for input types expected by orchestrator
InputType = Literal["buttons", "text", "none"]


@dataclass
class OrchestratorResponse:
    """
    Explicit output schema for frontend enforcement.
    
    This is the ONLY format returned by the orchestrator.
    Frontend should render based on input_type and options.
    """
    state: UC1State
    message: str
    input_type: InputType
    options: Optional[List[str]] = None  # Button labels if input_type=buttons
    terminal: bool = False  # True if EXIT state
    metadata: Optional[Dict[str, Any]] = None  # Additional data (e.g., CTA outcomes)
    
    def to_sse_chunks(self) -> List[Dict[str, Any]]:
        """
        Convert response to SSE-compatible chunks.
        
        This matches the existing streaming format used by chatbot_optimizer.py
        """
        chunks = []
        
        # Main message chunk
        if self.message:
            chunks.append({
                "status": "chunk",
                "chunk": self.message
            })
        
        # Meta chunk with state info
        meta = {
            "uc1_state": self.state.value,
            "uc1_input_type": self.input_type,
            "uc1_terminal": self.terminal,
        }
        if self.options:
            meta["uc1_options"] = self.options
        if self.metadata:
            meta.update(self.metadata)
        
        chunks.append({
            "status": "meta",
            "chunk": meta
        })
        
        return chunks


class ConversationOrchestrator:
    """
    Central conversation controller for UC1 flow.
    
    SOLE OWNER of UC1 conversation flow.
    LLM NEVER decides state transitions.
    
    The orchestrator:
    1. Validates input against current state
    2. Updates slots (with engagement scoring)
    3. Determines next state DETERMINISTICALLY
    4. Selects content DETERMINISTICALLY (alternatives, CTAs)
    5. Passes to LLM adapter for PARAPHRASING ONLY
    6. Sanitizes output
    7. Returns structured response
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
    
    def process_input(self, user_input: str) -> OrchestratorResponse:
        """
        Process user input and return structured response.
        
        This is the main entry point for conversation handling.
        
        Args:
            user_input: The user's message (empty string for initial entry)
        
        Returns:
            OrchestratorResponse: The structured response for frontend
        """
        logger.info(f"[Orchestrator] Processing input in state {self._current_state.value}: '{user_input[:50] if user_input else '(empty)'}...'")
        
        # Handle based on current state
        if self._current_state == UC1State.ENTRY:
            return self._handle_entry()
        elif self._current_state == UC1State.CAPABILITY_SELECTION:
            return self._handle_capability_selection(user_input)
        elif self._current_state == UC1State.CONTEXT_QUESTION:
            return self._handle_context_question(user_input)
        elif self._current_state == UC1State.NAME_CAPTURE:
            return self._handle_name_capture(user_input)
        elif self._current_state == UC1State.AI_SYNTHESIS:
            return self._handle_ai_synthesis()
        elif self._current_state == UC1State.CONSULTATIVE_ALTERNATIVES:
            return self._handle_consultative_alternatives(user_input)
        elif self._current_state == UC1State.RECOMMENDATION:
            return self._handle_recommendation(user_input)
        elif self._current_state == UC1State.EXIT:
            return self._handle_exit()
        else:
            logger.error(f"[Orchestrator] Unknown state: {self._current_state}")
            return OrchestratorResponse(
                state=self._current_state,
                message="I'm not sure what happened. Let's start over.",
                input_type="none",
                terminal=True
            )
    
    def _handle_entry(self) -> OrchestratorResponse:
        """Handle ENTRY state - show welcome message and capability options."""
        # Entry message from config (no bullet list - frontend renders buttons from options)
        entry_msg = self.config.entry_message
        
        # Transition to CAPABILITY_SELECTION
        self._current_state = UC1State.CAPABILITY_SELECTION
        
        # Get button options for frontend
        button_options = [bucket.trigger for bucket in self.config.capability_buckets]
        
        return OrchestratorResponse(
            state=self._current_state,
            message=entry_msg,
            input_type="buttons",
            options=button_options,
        )
    
    def _handle_capability_selection(self, user_input: str) -> OrchestratorResponse:
        """Handle CAPABILITY_SELECTION state - user selects a capability bucket."""
        # Find the matching bucket
        bucket = get_bucket_by_trigger(self.config, user_input)
        
        if not bucket:
            # Invalid selection - increment retry and ask again
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            button_options = [b.trigger for b in self.config.capability_buckets]
            return OrchestratorResponse(
                state=self._current_state,
                message="I didn't catch that. Please select one of the options above:",
                input_type="buttons",
                options=button_options,
            )
        
        # Valid selection
        self.slot_manager.set_capability_bucket(bucket.id)
        self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
        # Transition to CONTEXT_QUESTION
        self._current_state = UC1State.CONTEXT_QUESTION
        
        # Get the context question for this bucket
        question = self.llm_adapter.generate_context_question_prompt(bucket)
        
        return OrchestratorResponse(
            state=self._current_state,
            message=question,
            input_type="text",
        )
    
    def _handle_context_question(self, user_input: str) -> OrchestratorResponse:
        """Handle CONTEXT_QUESTION state - user provides context answer."""
        if not user_input or not user_input.strip():
            # Empty input - ask again
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
            question = bucket.context_question if bucket else "Could you tell me more about what you're looking for?"
            return OrchestratorResponse(
                state=self._current_state,
                message=f"I'd love to hear more. {question}",
                input_type="text",
            )
        
        # Valid input
        self.slot_manager.set_context_signal(user_input)
        self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        # Transition to NAME_CAPTURE
        self._current_state = UC1State.NAME_CAPTURE
        
        # Get name capture prompt
        prompt = self.llm_adapter.generate_name_capture_prompt()
        
        return OrchestratorResponse(
            state=self._current_state,
            message=f"Thanks for sharing! Before I share some thoughts, {prompt}",
            input_type="text",
        )
    
    def _handle_name_capture(self, user_input: str) -> OrchestratorResponse:
        """Handle NAME_CAPTURE state - user provides their name."""
        if not user_input or not user_input.strip():
            # Empty input - ask again
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            return OrchestratorResponse(
                state=self._current_state,
                message="I didn't catch your name. What should I call you?",
                input_type="text",
            )
        
        # Extract name (simple: take first word or full input if short)
        name = user_input.strip()
        if len(name.split()) > 3:
            # Likely a sentence, extract first capitalized word
            words = name.split()
            for word in words:
                if word[0].isupper() and len(word) > 1:
                    name = word
                    break
        
        self.slot_manager.set_user_name(name)
        self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        # Transition to AI_SYNTHESIS (auto-advance)
        self._current_state = UC1State.AI_SYNTHESIS
        
        # Process AI synthesis immediately
        return self._handle_ai_synthesis()
    
    def _handle_ai_synthesis(self) -> OrchestratorResponse:
        """Handle AI_SYNTHESIS state - generate synthesis and show alternatives."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        if not bucket:
            logger.error(f"[Orchestrator] Bucket not found: {self.slots.capability_bucket}")
            return OrchestratorResponse(
                state=UC1State.EXIT,
                message="Something went wrong. Let's start over.",
                input_type="none",
                terminal=True
            )
        
        # Generate synthesis (no alternatives list - frontend renders buttons from options)
        synthesis = self.llm_adapter.paraphrase_synthesis(bucket, self.slots)
        
        # Sanitize output
        sanitized, error = self.sanitizer.safe_sanitize(synthesis, UC1State.AI_SYNTHESIS)
        if error:
            logger.warning(f"[Orchestrator] Sanitization fixed issue: {error}")
        
        # Transition to CONSULTATIVE_ALTERNATIVES
        self._current_state = UC1State.CONSULTATIVE_ALTERNATIVES
        
        # Button options are the 3 alternatives
        button_options = list(bucket.alternatives)
        
        return OrchestratorResponse(
            state=self._current_state,
            message=sanitized,
            input_type="buttons",
            options=button_options,
        )
    
    def _handle_consultative_alternatives(self, user_input: str) -> OrchestratorResponse:
        """Handle CONSULTATIVE_ALTERNATIVES state - user selects an alternative."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Check if input matches one of the alternatives
        if bucket:
            alternatives_lower = [a.lower() for a in bucket.alternatives]
            input_lower = user_input.strip().lower()
            
            if input_lower in alternatives_lower:
                self.slot_manager.set_selected_alternative(user_input)
                self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
            else:
                # User typed something else - that's okay, still proceed
                self.slot_manager.set_selected_alternative(user_input)
                self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        # Transition to RECOMMENDATION
        self._current_state = UC1State.RECOMMENDATION
        
        # Prepare message (no CTA list - frontend renders buttons from options)
        full_message = "That makes sense. Here's what we can do next:"
        
        # Button options are the 4 CTAs
        button_options = [cta.choice for cta in self.config.exit_ctas]
        
        # Include outcome mapping in metadata for frontend
        outcome_map = {cta.choice: cta.outcome for cta in self.config.exit_ctas}
        
        return OrchestratorResponse(
            state=self._current_state,
            message=full_message,
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
            # Invalid CTA - ask again
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            button_options = [cta.choice for cta in self.config.exit_ctas]
            return OrchestratorResponse(
                state=self._current_state,
                message="I didn't quite get that. What would you like to do?",
                input_type="buttons",
                options=button_options,
            )
        
        # Valid CTA
        self.slot_manager.set_selected_cta_outcome(selected_cta.outcome)
        self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
        # Determine next state based on CTA outcome
        next_state = self.state_machine.get_next_state_for_cta(
            self._current_state, selected_cta.outcome
        )
        
        if next_state == UC1State.CAPABILITY_SELECTION:
            # Loop back - user wants to continue exploring (no bullet list - frontend renders buttons)
            self._current_state = UC1State.CAPABILITY_SELECTION
            button_options = [b.trigger for b in self.config.capability_buckets]
            
            return OrchestratorResponse(
                state=self._current_state,
                message="Great! Let's explore more. What else would you like to learn about?",
                input_type="buttons",
                options=button_options,
            )
        else:
            # Exit flow
            self._current_state = UC1State.EXIT
            return self._handle_exit()
    
    def _handle_exit(self) -> OrchestratorResponse:
        """Handle EXIT state - generate exit summary."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Generate exit summary
        summary = self.llm_adapter.generate_exit_summary(self.slots, bucket)
        
        return OrchestratorResponse(
            state=self._current_state,
            message=summary,
            input_type="none",
            terminal=True,
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
