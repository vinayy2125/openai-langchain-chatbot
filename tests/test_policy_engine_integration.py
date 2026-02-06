# Tests for UC1 Policy Engine
#
# These tests verify intent detection, resistance scoring, and lead capture governance.

import pytest
from app.orchestrator.policy_engine import (
    ConversationPolicy, PolicyDecision, PolicyContext,
    UserIntent, create_policy_context,
    CONTACT_INTENT_PATTERNS, RESISTANCE_PATTERNS
)


class TestUserIntentDetection:
    """Tests for intent detection from user messages."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.policy = ConversationPolicy("test-session")
    
    def test_detect_explicit_contact_request(self):
        """Test detection of explicit contact intent."""
        contact_phrases = [
            "contact me",
            "email me",
            "can we discuss",
            "schedule a call",
            "book a demo",
            "send me info",
        ]
        for phrase in contact_phrases:
            intent = self.policy.detect_intent(phrase)
            assert intent == UserIntent.REQUEST_CONTACT, f"Failed for: {phrase}"
    
    def test_detect_just_browsing(self):
        """Test detection of 'just browsing' intent (permanent suppression)."""
        browsing_phrases = [
            "just browsing",
            "just looking around",
            "no commitment yet",
        ]
        for phrase in browsing_phrases:
            intent = self.policy.detect_intent(phrase)
            assert intent == UserIntent.JUST_BROWSING, f"Failed for: {phrase}"
    
    def test_detect_resistance(self):
        """Test detection of resistance patterns."""
        resistance_phrases = [
            "not now",
            "stop asking",
            "can we skip this",
        ]
        for phrase in resistance_phrases:
            intent = self.policy.detect_intent(phrase)
            assert intent == UserIntent.RESISTANCE, f"Failed for: {phrase}"
    
    def test_detect_user_question(self):
        """Test detection of user asking questions (triggers PIVOT)."""
        questions = [
            "what services do you offer?",
            "how do you handle AI projects?",
            "tell me about your team",
        ]
        for q in questions:
            intent = self.policy.detect_intent(q)
            assert intent == UserIntent.ASK_QUESTION, f"Failed for: {q}"
    
    def test_detect_continue_flow(self):
        """Test normal responses continue flow."""
        normal_responses = [
            "yes",
            "I want to build a new product",
            "mobile app",
        ]
        for resp in normal_responses:
            intent = self.policy.detect_intent(resp)
            assert intent == UserIntent.CONTINUE_FLOW, f"Failed for: {resp}"


class TestPolicyEvaluation:
    """Tests for policy decision evaluation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.policy = ConversationPolicy("test-session")
    
    def test_pivot_on_user_question_during_uc1(self):
        """Test that user questions during UC1 trigger PIVOT to FREE_EXPLORATION."""
        context = PolicyContext(
            current_state="capability_selection",
            user_input="what services do you offer?",
            is_in_uc1=True,
            is_in_free_exploration=False,
        )
        decision = self.policy.evaluate(context)
        assert decision == PolicyDecision.PIVOT
    
    def test_proceed_on_normal_answer(self):
        """Test that normal answers proceed with UC1 flow."""
        context = PolicyContext(
            current_state="capability_selection",
            user_input="I want AI/ML solutions",
            is_in_uc1=True,
            is_in_free_exploration=False,
        )
        decision = self.policy.evaluate(context)
        assert decision == PolicyDecision.PROCEED
    
    def test_explicit_contact_triggers_capture(self):
        """Test that explicit contact request triggers lead capture."""
        context = PolicyContext(
            current_state="exploration_layer",
            user_input="please contact me",
            is_in_uc1=True,
            lead_capture_permanently_suppressed=False,
            lead_capture_asked_count=0,
        )
        decision = self.policy.evaluate(context)
        assert decision == PolicyDecision.CAPTURE_LEAD
    
    def test_suppress_when_permanent(self):
        """Test lead capture suppressed when permanently suppressed."""
        context = PolicyContext(
            current_state="exploration_layer",
            user_input="contact me",  # Would normally trigger
            is_in_uc1=True,
            lead_capture_permanently_suppressed=True,
        )
        # Should NOT return CAPTURE_LEAD due to suppression
        decision = self.policy.evaluate(context)
        assert decision != PolicyDecision.CAPTURE_LEAD


class TestResistanceDetection:
    """Tests for resistance scoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.policy = ConversationPolicy("test-session")
    
    def test_high_resistance_on_explicit_pushback(self):
        """Test high resistance detected on explicit pushback."""
        resistance = self.policy.detect_resistance("skip this", [])
        assert resistance >= 0.3
    
    def test_low_resistance_on_normal_input(self):
        """Test low resistance on normal cooperative input."""
        resistance = self.policy.detect_resistance("yes, I'm interested in AI", [])
        assert resistance < 0.3


class TestArbitrationPriority:
    """Tests for policy arbitration priority order."""
    
    def test_priority_order(self):
        """Test that arbitration priority is correctly ordered."""
        priority = ConversationPolicy.arbitration_priority()
        assert priority[0] == "explicit_user_intent"  # Highest
        assert priority[-1] == "uc1_compliance"  # Lowest
