# Tests for UC1 State Input Validators
#
# These tests verify pre-slot-mutation validation:
# - validate_context_answer: min length, word count
# - validate_name: regex pattern, edge cases
# - validate_exploration_input: min length

import pytest
from app.orchestrator.state_input_validators import (
    validate_context_answer,
    validate_name,
    validate_exploration_input,
)


class TestContextAnswerValidation:
    """Tests for context answer validation."""
    
    def test_valid_context_answer(self):
        """Test valid context answers pass validation."""
        is_valid, reason = validate_context_answer("Building a new mobile app")
        assert is_valid is True
        assert reason == ""
    
    def test_empty_context_fails(self):
        """Test empty input fails with reason."""
        is_valid, reason = validate_context_answer("")
        assert is_valid is False
        assert reason == "empty_input"
    
    def test_none_context_fails(self):
        """Test None input fails."""
        is_valid, reason = validate_context_answer(None)
        assert is_valid is False
        assert reason == "empty_input"
    
    def test_too_short_context_fails(self):
        """Test context shorter than 5 chars fails."""
        is_valid, reason = validate_context_answer("hi")
        assert is_valid is False
        assert reason == "too_short"
        
        is_valid, reason = validate_context_answer("abcd")
        assert is_valid is False
        assert reason == "too_short"
    
    def test_single_word_context_fails(self):
        """Test single word context fails (needs 2+ words)."""
        is_valid, reason = validate_context_answer("development")
        assert is_valid is False
        assert reason == "too_few_words"
    
    def test_whitespace_trimmed(self):
        """Test leading/trailing whitespace is trimmed."""
        is_valid, reason = validate_context_answer("   new project   ")
        assert is_valid is True
    
    def test_exactly_two_words_passes(self):
        """Test exactly 2 words at minimum length passes."""
        is_valid, reason = validate_context_answer("new app")
        assert is_valid is True
    
    def test_long_context_passes(self):
        """Test long context answers pass."""
        long_answer = "We are looking to modernize our legacy enterprise application and migrate to cloud infrastructure with AI capabilities"
        is_valid, reason = validate_context_answer(long_answer)
        assert is_valid is True


class TestNameValidation:
    """Tests for name validation."""
    
    @pytest.mark.parametrize("name", [
        "John",
        "Alice",
        "Bob Smith",
        "John Doe",
        "Mary Jane Watson",
    ])
    def test_valid_names_pass(self, name):
        """Test valid names pass validation."""
        is_valid, reason = validate_name(name)
        assert is_valid is True
        assert reason == ""
    
    def test_empty_name_fails(self):
        """Test empty input fails."""
        is_valid, reason = validate_name("")
        assert is_valid is False
        assert reason == "empty_input"
    
    def test_none_name_fails(self):
        """Test None input fails."""
        is_valid, reason = validate_name(None)
        assert is_valid is False
        assert reason == "empty_input"
    
    @pytest.mark.parametrize("invalid_name", [
        "A",           # Single char
        "AB",          # Two chars (word too short)
        "123",         # Numbers only
        "John123",     # Mixed alphanumeric
        "john@doe",    # Special characters
        "John-Doe",    # Hyphen
        "O'Brien",     # Apostrophe
    ])
    def test_invalid_name_formats_fail(self, invalid_name):
        """Test invalid name formats fail."""
        is_valid, reason = validate_name(invalid_name)
        assert is_valid is False
        assert reason == "invalid_format"
    
    def test_too_many_words_fails(self):
        """Test more than 3 words fails."""
        is_valid, reason = validate_name("John James Smith Junior")
        assert is_valid is False
        assert reason == "invalid_format"
    
    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed before validation."""
        is_valid, reason = validate_name("  John  ")
        assert is_valid is True
    
    def test_two_char_name_minimum(self):
        """Test 2-char names pass (minimum length)."""
        is_valid, reason = validate_name("Jo")
        assert is_valid is True
    
    def test_case_insensitive_validation(self):
        """Test names work with various casings."""
        assert validate_name("JOHN")[0] is True
        assert validate_name("john")[0] is True
        assert validate_name("JoHn")[0] is True


class TestExplorationInputValidation:
    """Tests for exploration input validation."""
    
    def test_valid_exploration_input(self):
        """Test valid exploration inputs pass."""
        is_valid, reason = validate_exploration_input("We need scalability")
        assert is_valid is True
        assert reason == ""
    
    def test_empty_exploration_fails(self):
        """Test empty input fails."""
        is_valid, reason = validate_exploration_input("")
        assert is_valid is False
        assert reason == "empty_input"
    
    def test_none_exploration_fails(self):
        """Test None input fails."""
        is_valid, reason = validate_exploration_input(None)
        assert is_valid is False
        assert reason == "empty_input"
    
    def test_too_short_exploration_fails(self):
        """Test exploration shorter than 3 chars fails."""
        is_valid, reason = validate_exploration_input("ab")
        assert is_valid is False
        assert reason == "too_short"
    
    def test_exactly_three_chars_passes(self):
        """Test exactly 3 chars passes."""
        is_valid, reason = validate_exploration_input("yes")
        assert is_valid is True
    
    def test_whitespace_counts(self):
        """Test whitespace-only input fails after trimming."""
        is_valid, reason = validate_exploration_input("   ")
        assert is_valid is False
        assert reason == "too_short"


class TestValidatorEdgeCases:
    """Edge case tests for all validators."""
    
    def test_context_with_special_chars(self):
        """Test context with special characters."""
        is_valid, reason = validate_context_answer("AI/ML solutions!")
        assert is_valid is True
    
    def test_context_with_numbers(self):
        """Test context with numbers works."""
        is_valid, reason = validate_context_answer("Need 5 developers")
        assert is_valid is True
    
    def test_name_boundary_word_length(self):
        """Test name with exactly 2-char words."""
        is_valid, reason = validate_name("Jo Bo Li")
        assert is_valid is True
    
    def test_exploration_unicode(self):
        """Test exploration with unicode."""
        is_valid, reason = validate_exploration_input("Hello 世界")
        assert is_valid is True
    
    def test_validators_return_tuple(self):
        """Test all validators return (bool, str) tuple."""
        for validator in [validate_context_answer, validate_name, validate_exploration_input]:
            result = validator("test input")
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
