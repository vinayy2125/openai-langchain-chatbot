# Tests for UC1 Output Sanitizer
#
# These tests verify forbidden topic detection and question filtering.

import pytest
from app.orchestrator.uc1_config import load_uc1_config
from app.orchestrator.state_machine import UC1State
from app.orchestrator.output_sanitizer import (
    LLMOutputSanitizer,
    ForbiddenTopicViolation,
    UnauthorizedQuestionViolation,
)


class TestForbiddenTopicDetection:
    """Tests for forbidden topic detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = load_uc1_config()
        self.sanitizer = LLMOutputSanitizer(self.config)
    
    def test_clean_output_passes(self):
        """Test that clean output passes sanitization."""
        output = "We help companies build great products and scale their teams."
        result = self.sanitizer.sanitize(output, UC1State.ENTRY)
        assert result == output
    
    def test_pricing_topic_detected(self):
        """Test that pricing mention is detected."""
        output = "Our pricing starts at $5000 per month."
        
        with pytest.raises(ForbiddenTopicViolation) as exc_info:
            self.sanitizer.sanitize(output, UC1State.ENTRY)
        
        assert exc_info.value.topic in ["pricing", "$"]
    
    def test_cost_topic_detected(self):
        """Test that cost mention is detected."""
        output = "The total cost depends on your requirements."
        
        with pytest.raises(ForbiddenTopicViolation) as exc_info:
            self.sanitizer.sanitize(output, UC1State.ENTRY)
        
        assert exc_info.value.topic == "cost"
    
    def test_technology_stack_detected(self):
        """Test that technology stack mention is detected."""
        output = "We use React and Python for development."
        
        with pytest.raises(ForbiddenTopicViolation) as exc_info:
            self.sanitizer.sanitize(output, UC1State.ENTRY)
        
        # Either React or Python should be detected
        assert exc_info.value.topic in ["React", "Python"]


class TestQuestionFiltering:
    """Tests for unauthorized question filtering."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = load_uc1_config()
        self.sanitizer = LLMOutputSanitizer(self.config)
    
    def test_no_questions_in_synthesis_state(self):
        """Test that questions are blocked in AI_SYNTHESIS state."""
        output = "Here's what we recommend. What else would you like to know?"
        
        with pytest.raises(UnauthorizedQuestionViolation):
            self.sanitizer.sanitize(output, UC1State.AI_SYNTHESIS)
    
    def test_no_questions_in_alternatives_state(self):
        """Test that questions are blocked in CONSULTATIVE_ALTERNATIVES state."""
        output = "Choose one of these options. Do you have a preference?"
        
        with pytest.raises(UnauthorizedQuestionViolation):
            self.sanitizer.sanitize(output, UC1State.CONSULTATIVE_ALTERNATIVES)
    
    def test_questions_allowed_in_other_states(self):
        """Test that questions are allowed in other states."""
        output = "What should I call you?"
        
        # Should not raise in NAME_CAPTURE state
        result = self.sanitizer.sanitize(output, UC1State.NAME_CAPTURE)
        assert result == output


class TestSafeSanitize:
    """Tests for safe_sanitize method with fallback."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = load_uc1_config()
        self.sanitizer = LLMOutputSanitizer(self.config)
    
    def test_safe_sanitize_returns_fallback_on_violation(self):
        """Test that safe_sanitize returns fallback instead of raising."""
        output = "Check our pricing page for details."
        
        result, error = self.sanitizer.safe_sanitize(output, UC1State.ENTRY)
        
        assert error is not None
        assert isinstance(error, ForbiddenTopicViolation)
        assert "pricing" not in result.lower()
    
    def test_safe_sanitize_removes_questions_gracefully(self):
        """Test that questions are removed gracefully."""
        output = "Here are three options. Which one interests you?"
        
        result, error = self.sanitizer.safe_sanitize(output, UC1State.AI_SYNTHESIS)
        
        assert error is not None
        assert "?" not in result  # Question mark should be replaced
