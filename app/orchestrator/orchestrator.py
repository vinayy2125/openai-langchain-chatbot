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
from app.orchestrator.uc1_config import (
    UC1Config, load_uc1_config, get_bucket_by_id, 
    get_bucket_by_trigger, CapabilityBucket
)
from app.orchestrator.state_machine import UC1State, UC1StateMachine, ResponseIntent
from app.orchestrator.slot_manager import UC1Slots, SlotManager, EngagementEvent
from app.orchestrator.llm_adapter import ConstrainedLLMAdapter
from app.orchestrator.output_sanitizer import LLMOutputSanitizer
from app.orchestrator.policy_validator import UC1PolicyValidator
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
    """
    state: UC1State
    call_spec: AdapterCallSpec  # What the LLM adapter needs
    input_type: InputType
    options: Optional[List[str]] = None
    terminal: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    # Message is set by caller after LLM generation, not by orchestrator
    message: str = ""
    
    def to_sse_chunks(self) -> List[Dict[str, Any]]:
        """Convert response to SSE-compatible chunks."""
        chunks = []
        
        if self.message:
            chunks.append({"status": "chunk", "chunk": self.message})
        
        meta = {
            "uc1_state": self.state.value,
            "uc1_input_type": self.input_type,
            "uc1_terminal": self.terminal,
        }
        if self.options:
            meta["uc1_options"] = self.options
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
        elif self._current_state == UC1State.EXPLORATION_LAYER:
            return self._handle_exploration_layer(user_input)
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
                input_type="none",
                terminal=True
            )
    
    def _handle_entry(self) -> OrchestratorResponse:
        """Handle ENTRY state - transition to capability selection."""
        # Transition to CAPABILITY_SELECTION
        self._current_state = UC1State.CAPABILITY_SELECTION
        button_options = [bucket.trigger for bucket in self.config.capability_buckets]
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.ENTRY,
                response_intent=ResponseIntent.PROMPT,
                slots=self.slots,
            ),
            input_type="buttons",
            options=button_options,
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
        self.slot_manager.set_capability_bucket(bucket.id)
        self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
        # Transition to CONTEXT_QUESTION
        self._current_state = UC1State.CONTEXT_QUESTION
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.CONTEXT_QUESTION,
                response_intent=ResponseIntent.PROMPT,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="text",
        )
    
    def _handle_context_question(self, user_input: str) -> OrchestratorResponse:
        """Handle CONTEXT_QUESTION state - user provides context answer."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        if not user_input or not user_input.strip():
            # Empty input - retry
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
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
            )
        
        # Valid input
        self.slot_manager.set_context_signal(user_input)
        self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        # Transition to NAME_CAPTURE
        self._current_state = UC1State.NAME_CAPTURE
        
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
        )
    
    def _handle_name_capture(self, user_input: str) -> OrchestratorResponse:
        """Handle NAME_CAPTURE state - user provides their name."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        if not user_input or not user_input.strip():
            # Empty input - retry
            self.slot_manager.increment_engagement(EngagementEvent.RETRY)
            return OrchestratorResponse(
                state=self._current_state,
                call_spec=AdapterCallSpec(
                    state=UC1State.NAME_CAPTURE,
                    response_intent=ResponseIntent.RETRY,
                    slots=self.slots,
                    bucket=bucket,
                ),
                input_type="text",
            )
        
        # Extract name (take first word or full input if short)
        name = user_input.strip()
        if len(name.split()) > 3:
            words = name.split()
            for word in words:
                if word[0].isupper() and len(word) > 1:
                    name = word
                    break
        
        self.slot_manager.set_user_name(name)
        self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        # Transition to EXPLORATION_LAYER
        self._current_state = UC1State.EXPLORATION_LAYER
        self.slot_manager.set_exploration_turn(1)
        
        # Return spec for acknowledgment + first exploration question
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.EXPLORATION_LAYER,
                response_intent=ResponseIntent.ACKNOWLEDGE,
                user_input=name,  # Pass name for acknowledgment
                slots=self.slots,
                bucket=bucket,
                exploration_turn=1,
            ),
            input_type="text",
        )
    
    def _handle_exploration_layer(self, user_input: str) -> OrchestratorResponse:
        """Handle EXPLORATION_LAYER state - 2 turns of guided Q&A (TEXT-BLIND)."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        if not bucket:
            logger.error(f"[Orchestrator] Bucket not found in exploration")
            self._current_state = UC1State.AI_SYNTHESIS
            return self._handle_ai_synthesis()
        
        current_turn = self.slots.exploration_turn
        max_turns = 2
        
        # User provided a response - process it
        if user_input:
            self.slot_manager.add_exploration_response(user_input)
            self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        # Check if we've completed enough exploration turns
        if current_turn >= max_turns and user_input:
            # Done exploring - transition to AI_SYNTHESIS
            logger.info(f"[Orchestrator] Exploration complete after {current_turn} turns")
            self._current_state = UC1State.AI_SYNTHESIS
            return self._handle_ai_synthesis()
        
        # Generate next exploration prompt/reflect
        next_turn = current_turn + 1 if user_input else current_turn
        self.slot_manager.set_exploration_turn(next_turn)
        
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
        """Handle AI_SYNTHESIS state - present alternatives (TEXT-BLIND)."""
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
                input_type="none",
                terminal=True
            )
        
        # Transition to CONSULTATIVE_ALTERNATIVES
        self._current_state = UC1State.CONSULTATIVE_ALTERNATIVES
        button_options = list(bucket.alternatives)
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.AI_SYNTHESIS,
                response_intent=ResponseIntent.PRESENT,
                slots=self.slots,
                bucket=bucket,
            ),
            input_type="buttons",
            options=button_options,
        )
    
    def _handle_consultative_alternatives(self, user_input: str) -> OrchestratorResponse:
        """Handle CONSULTATIVE_ALTERNATIVES state - user selects alternative (TEXT-BLIND)."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        # Track user selection
        if bucket:
            alternatives_lower = [a.lower() for a in bucket.alternatives]
            input_lower = user_input.strip().lower()
            
            if input_lower in alternatives_lower:
                self.slot_manager.set_selected_alternative(user_input)
                self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
            else:
                self.slot_manager.set_selected_alternative(user_input)
                self.slot_manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
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
        self.slot_manager.set_selected_cta_outcome(selected_cta.outcome)
        self.slot_manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
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
        else:
            # Exit flow
            self._current_state = UC1State.EXIT
            return self._handle_exit()
    
    def _handle_exit(self) -> OrchestratorResponse:
        """Handle EXIT state (TEXT-BLIND)."""
        bucket = get_bucket_by_id(self.config, self.slots.capability_bucket)
        
        return OrchestratorResponse(
            state=self._current_state,
            call_spec=AdapterCallSpec(
                state=UC1State.EXIT,
                response_intent=ResponseIntent.EXIT,
                slots=self.slots,
                bucket=bucket,
            ),
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
