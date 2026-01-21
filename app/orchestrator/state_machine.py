# UC1 State Machine - Deterministic State Transitions
#
# ARCHITECTURE RULE: The state machine owns ALL transition logic.
# LLM never decides state transitions - only the orchestrator through this machine.
#
# State transitions are deterministic based on:
# - Current state
# - User input type (button click vs free text)
# - Required slots being filled

from dataclasses import dataclass, field
from enum import Enum
from typing import Set, List, Optional, Dict, Literal
from app.logger import get_logger

logger = get_logger("state_machine")


class UC1State(Enum):
    """
    The 10 states of the UC1 conversation flow.
    
    Flow: ENTRY → CAPABILITY_SELECTION → CONTEXT_QUESTION → NAME_CAPTURE 
          → AI_SYNTHESIS (S4) → EXPLORATION_LAYER (S5, 2-3 turns) 
          → CONSULTATIVE_ALTERNATIVES (S6) → RECOMMENDATION (S7) → EXIT
    
    FREE_EXPLORATION: User-driven unstructured conversation.
    - User can pivot here from ANY state
    - UC1 state is PAUSED (not abandoned)
    - Resume to UC1 only on EXPLICIT user intent
    
    Note: Some transitions allow looping (e.g., RECOMMENDATION → CAPABILITY_SELECTION
    via "Continue exploring" CTA).
    """
    ENTRY = "entry"
    CAPABILITY_SELECTION = "capability_selection"
    CONTEXT_QUESTION = "context_question"
    NAME_CAPTURE = "name_capture"
    AI_SYNTHESIS = "ai_synthesis"
    EXPLORATION_LAYER = "exploration_layer"
    CONSULTATIVE_ALTERNATIVES = "consultative_alternatives"
    RECOMMENDATION = "recommendation"
    EMAIL_CAPTURE = "email_capture"  # NEW: Force email capture before exit
    EXIT = "exit"
    FREE_EXPLORATION = "free_exploration"  # Unstructured user-driven conversation


class ResponseIntent(Enum):
    """
    WHY something must be said (not WHAT).
    
    The orchestrator specifies intent; the LLM adapter generates language.
    This separation ensures the orchestrator remains text-blind.
    """
    PROMPT = "prompt"           # Initial state prompt (ask appropriate question)
    RETRY = "retry"             # Invalid/empty input, ask again
    TRANSITION = "transition"   # Acknowledge and move to next state
    ACKNOWLEDGE = "acknowledge" # Acknowledge user input (e.g., name)
    REFLECT = "reflect"         # Reflect on user's response
    PRESENT = "present"         # Present options/alternatives
    EXIT = "exit"               # Closing conversation
    GRACEFUL_EXIT = "graceful_exit" # Explicit graceful exit with message


# Input types that the user can provide
InputType = Literal["none", "buttons", "text"]


@dataclass(frozen=True)
class StateConfig:
    """
    Configuration for a single state.
    
    Defines what inputs are valid, what slots are required,
    and what states can follow.
    """
    input_type: InputType  # What kind of input this state expects
    next_states: tuple  # Valid next states (tuple of UC1State)
    required_slots: tuple  # Slots that must be filled to advance (tuple of str)
    retry_limit: int = 3  # Max retries for invalid input before escalation
    system_message_key: Optional[str] = None  # Key to fetch system message for this state


# State configuration - defines the entire state machine
STATE_CONFIGS: Dict[UC1State, StateConfig] = {
    UC1State.ENTRY: StateConfig(
        input_type="none",  # System auto-advances, no user input needed
        next_states=(UC1State.CAPABILITY_SELECTION,),
        required_slots=(),
        system_message_key="entry_message",
    ),
    UC1State.CAPABILITY_SELECTION: StateConfig(
        input_type="buttons",  # User selects from 6 capability buckets
        next_states=(UC1State.CONTEXT_QUESTION,),
        required_slots=("capability_bucket",),
        system_message_key=None,  # Message is derived from entry_message
    ),
    UC1State.CONTEXT_QUESTION: StateConfig(
        input_type="text",  # User provides free-form context answer
        next_states=(UC1State.NAME_CAPTURE,),
        required_slots=("context_signal",),
        system_message_key=None,  # Message comes from bucket.context_question
    ),
    UC1State.NAME_CAPTURE: StateConfig(
        input_type="text",  # User provides their name
        next_states=(UC1State.AI_SYNTHESIS,),  # S4: AI Synthesis comes after name
        required_slots=("user_name",),
        system_message_key="name_capture_prompt",
    ),
    UC1State.EXPLORATION_LAYER: StateConfig(
        input_type="text",  # User provides free-form exploration answers
        next_states=(UC1State.CONSULTATIVE_ALTERNATIVES,),  # S5 → S6: After exploration, show alternatives
        required_slots=(),  # No slots required to advance
        retry_limit=3,
        system_message_key=None,  # Questions come from LLM adapter
    ),
    UC1State.AI_SYNTHESIS: StateConfig(
        input_type="text",  # Allow user to type while synthesis displays (better interactivity)
        next_states=(UC1State.EXPLORATION_LAYER,),  # S4 → S5: After synthesis, do exploration
        required_slots=(),
        system_message_key=None,  # Generated by LLM paraphraser
    ),
    UC1State.CONSULTATIVE_ALTERNATIVES: StateConfig(
        input_type="buttons",  # User selects from 3 alternatives
        next_states=(UC1State.RECOMMENDATION,),
        required_slots=(),  # Selection is informational, not blocking
        system_message_key=None,  # Alternatives come from bucket config
    ),
    UC1State.RECOMMENDATION: StateConfig(
        input_type="buttons",  # User selects from 4 CTAs
        next_states=(UC1State.EXIT, UC1State.CAPABILITY_SELECTION, UC1State.EMAIL_CAPTURE),  # Added EMAIL_CAPTURE
        required_slots=(),
        system_message_key=None,  # CTAs come from config
    ),
    UC1State.EMAIL_CAPTURE: StateConfig(
        input_type="text",
        next_states=(UC1State.EXIT, UC1State.CAPABILITY_SELECTION),
        required_slots=("user_email",),
        system_message_key="email_capture_prompt", # We will need to add this property or handle it
        retry_limit=3,
    ),
    UC1State.EXIT: StateConfig(
        input_type="text",  # Allow re-engagement even after goodbye (better UX)
        next_states=(),  # No next states - conversation ends
        required_slots=(),
        system_message_key=None,  # Exit summary generated by LLM
    ),
    UC1State.FREE_EXPLORATION: StateConfig(
        input_type="text",  # User-driven conversation, always text
        next_states=(UC1State.CAPABILITY_SELECTION,),  # Can only return to UC1 via explicit intent
        required_slots=(),  # No slots required
        retry_limit=99,  # Effectively infinite - no forcing user back
        system_message_key=None,  # LLM answers questions, no qualifying questions allowed
    ),
}


class UC1StateMachine:
    """
    Deterministic state machine for UC1 conversation flow.
    
    OWNERSHIP:
    - This class owns ALL transition logic
    - Orchestrator calls this to determine valid transitions
    - LLM never influences state transitions
    """
    
    def __init__(self):
        self._configs = STATE_CONFIGS
    
    def get_state_config(self, state: UC1State) -> StateConfig:
        """Get configuration for a state."""
        return self._configs[state]
    
    def get_valid_next_states(self, current_state: UC1State) -> tuple:
        """Get valid next states from current state."""
        return self._configs[current_state].next_states
    
    def get_required_slots(self, state: UC1State) -> tuple:
        """Get slots required to advance from this state."""
        return self._configs[state].required_slots
    
    def get_input_type(self, state: UC1State) -> InputType:
        """Get expected input type for a state."""
        return self._configs[state].input_type
    
    def is_terminal(self, state: UC1State) -> bool:
        """Check if state is terminal (no valid next states)."""
        return len(self._configs[state].next_states) == 0
    
    def can_transition(
        self,
        from_state: UC1State,
        to_state: UC1State,
        filled_slots: Set[str],
    ) -> bool:
        """
        Check if transition is valid given current slots.
        
        Args:
            from_state: Current state
            to_state: Proposed next state
            filled_slots: Set of slot names that are currently filled
        
        Returns:
            bool: True if transition is valid
        """
        config = self._configs[from_state]
        
        # Check if to_state is a valid next state
        if to_state not in config.next_states:
            logger.warning(
                f"[StateMachine] Invalid transition: {from_state.value} → {to_state.value} "
                f"(valid: {[s.value for s in config.next_states]})"
            )
            return False
        
        # Check if required slots are filled
        required = set(config.required_slots)
        missing = required - filled_slots
        if missing:
            logger.warning(
                f"[StateMachine] Transition blocked: {from_state.value} → {to_state.value} "
                f"(missing slots: {missing})"
            )
            return False
        
        return True
    
    def get_default_next_state(self, current_state: UC1State) -> Optional[UC1State]:
        """
        Get the default next state (first in the list of valid next states).
        
        Used when there's only one valid next state.
        """
        next_states = self._configs[current_state].next_states
        if next_states:
            return next_states[0]
        return None
    
    def get_next_state_for_cta(
        self,
        current_state: UC1State,
        cta_outcome: str,
    ) -> Optional[UC1State]:
        """
        Determine next state based on CTA outcome.
        
        This is specifically for RECOMMENDATION state where CTAs
        can lead to different next states.
        
        Args:
            current_state: Must be RECOMMENDATION
            cta_outcome: The outcome string from ExitCTA
        
        Returns:
            UC1State: The next state based on CTA, or None if invalid
        """
        if current_state != UC1State.RECOMMENDATION:
            logger.warning(f"[StateMachine] get_next_state_for_cta called from non-RECOMMENDATION state: {current_state.value}")
            return None
        
        # Map CTA outcomes to states
        outcome_map = {
            "UC2": UC1State.EMAIL_CAPTURE,  # Discuss requirement -> Get Email first
            "calendar": UC1State.EMAIL_CAPTURE,  # Schedule call -> Get Email first
            "loop": UC1State.CAPABILITY_SELECTION,  # Continue exploring = loop back
            "exit": UC1State.EXIT,  # Graceful exit
        }
        
        next_state = outcome_map.get(cta_outcome)
        if next_state is None:
            logger.warning(f"[StateMachine] Unknown CTA outcome: {cta_outcome}")
        
        return next_state
    
    def can_pivot_to_free_exploration(self, current_state: UC1State) -> bool:
        """
        Check if user can pivot to FREE_EXPLORATION from current state.
        
        FREE_EXPLORATION is an ESCAPE HATCH from the UC1 funnel.
        User can pivot here from ANY state except EXIT.
        
        Returns:
            bool: True if pivot is allowed
        """
        # Can pivot from any state except terminal states
        if current_state == UC1State.EXIT:
            return False
        if current_state == UC1State.FREE_EXPLORATION:
            return False  # Already there
        return True
    
    def is_free_exploration(self, state: UC1State) -> bool:
        """Check if state is FREE_EXPLORATION."""
        return state == UC1State.FREE_EXPLORATION
    
    def get_paused_uc1_state(self, state_before_pivot: UC1State) -> UC1State:
        """
        Get the state to resume when user exits FREE_EXPLORATION.
        
        UC1 is PAUSED, not abandoned. User resumes from where they left off.
        Only called on EXPLICIT user intent to resume.
        
        Args:
            state_before_pivot: The UC1 state before entering FREE_EXPLORATION
            
        Returns:
            UC1State: The state to resume (typically the paused state)
        """
        # If user was in a completion-oriented state, resume from start
        if state_before_pivot in (UC1State.EXIT, UC1State.RECOMMENDATION):
            return UC1State.CAPABILITY_SELECTION
        return state_before_pivot

