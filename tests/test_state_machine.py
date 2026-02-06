# Tests for UC1 State Machine
#
# These tests verify state transitions and slot requirements.

import pytest
from app.orchestrator.state_machine import (
    UC1State,
    UC1StateMachine,
    StateConfig,
    STATE_CONFIGS,
)


class TestUC1StateMachine:
    """Tests for state machine transitions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.sm = UC1StateMachine()
    
    def test_all_states_have_config(self):
        """Test that all states have configuration."""
        for state in UC1State:
            config = self.sm.get_state_config(state)
            assert config is not None
            assert isinstance(config, StateConfig)
    
    def test_entry_transitions_to_capability_selection(self):
        """Test ENTRY → CAPABILITY_SELECTION transition."""
        next_states = self.sm.get_valid_next_states(UC1State.ENTRY)
        assert UC1State.CAPABILITY_SELECTION in next_states
    
    def test_exit_is_terminal(self):
        """Test that EXIT state is terminal."""
        assert self.sm.is_terminal(UC1State.EXIT)
        next_states = self.sm.get_valid_next_states(UC1State.EXIT)
        assert len(next_states) == 0
    
    def test_capability_selection_requires_bucket_slot(self):
        """Test that CAPABILITY_SELECTION requires capability_bucket slot."""
        required = self.sm.get_required_slots(UC1State.CAPABILITY_SELECTION)
        assert "capability_bucket" in required
    
    def test_context_question_requires_signal_slot(self):
        """Test that CONTEXT_QUESTION requires context_signal slot."""
        required = self.sm.get_required_slots(UC1State.CONTEXT_QUESTION)
        assert "context_signal" in required
    
    def test_name_capture_requires_user_name_slot(self):
        """Test that NAME_CAPTURE requires user_name slot."""
        required = self.sm.get_required_slots(UC1State.NAME_CAPTURE)
        assert "user_name" in required
    
    def test_can_transition_with_filled_slots(self):
        """Test valid transition when slots are filled."""
        can_advance = self.sm.can_transition(
            from_state=UC1State.CAPABILITY_SELECTION,
            to_state=UC1State.CONTEXT_QUESTION,
            filled_slots={"capability_bucket"}
        )
        assert can_advance is True
    
    def test_cannot_transition_with_missing_slots(self):
        """Test transition blocked when required slots missing."""
        can_advance = self.sm.can_transition(
            from_state=UC1State.CAPABILITY_SELECTION,
            to_state=UC1State.CONTEXT_QUESTION,
            filled_slots=set()  # Empty - missing capability_bucket
        )
        assert can_advance is False
    
    def test_recommendation_cta_loop(self):
        """Test that RECOMMENDATION can loop back to CAPABILITY_SELECTION."""
        next_state = self.sm.get_next_state_for_cta(
            UC1State.RECOMMENDATION,
            "loop"
        )
        assert next_state == UC1State.CAPABILITY_SELECTION
    
    def test_recommendation_cta_exit(self):
        """Test that RECOMMENDATION can exit with UC2/calendar/exit CTAs."""
        for outcome in ["UC2", "calendar", "exit"]:
            next_state = self.sm.get_next_state_for_cta(
                UC1State.RECOMMENDATION,
                outcome
            )
            assert next_state == UC1State.EXIT


class TestStateFlow:
    """Tests for complete state flow sequences."""
    
    def test_happy_path_flow(self):
        """Test complete happy path through all states."""
        sm = UC1StateMachine()
        
        # Expected flow (SPEC-COMPLIANT: S4=Synthesis before S5=Exploration)
        flow = [
            UC1State.ENTRY,
            UC1State.CAPABILITY_SELECTION,
            UC1State.CONTEXT_QUESTION,
            UC1State.NAME_CAPTURE,
            UC1State.AI_SYNTHESIS,  # S4 - synthesis BEFORE exploration per spec
            UC1State.EXPLORATION_LAYER,  # S5 - 2 turns of Q&A after synthesis
            UC1State.CONSULTATIVE_ALTERNATIVES,
            UC1State.RECOMMENDATION,
            UC1State.EXIT,
        ]
        
        # Verify each transition is valid
        for i in range(len(flow) - 1):
            current = flow[i]
            next_state = flow[i + 1]
            valid_next = sm.get_valid_next_states(current)
            assert next_state in valid_next, f"Invalid: {current} → {next_state}"
