# Tests for UC1 State Input Validators
#
# These tests verify pre-slot-mutation validation:
# - validate_context_answer: global rules + alpha signal + min length
# - validate_name: international names, placeholders rejection
# - validate_exploration_input: global rules + higher min length

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
    
    def test_short_meaningful_answer_passes(self):
        """Test short but meaningful words pass (e.g., 'new', 'existing')."""
        is_valid, reason = validate_context_answer("new")
        assert is_valid is True
        
        is_valid, reason = validate_context_answer("existing")
        assert is_valid is True
        
        is_valid, reason = validate_context_answer("mobile")
        assert is_valid is True
    
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
    
    def test_single_char_fails(self):
        """Test single character fails too_short."""
        is_valid, reason = validate_context_answer("a")
        assert is_valid is False
        assert reason == "too_short"
    
    def test_acknowledgment_denylist_fails(self):
        """Test acknowledgment/filler words are rejected."""
        for word in ["ok", "okay", "yes", "no", "sure", "fine", "cool", "yeah"]:
            is_valid, reason = validate_context_answer(word)
            assert is_valid is False, f"'{word}' should be rejected"
            assert reason == "low_signal"
    
    def test_placeholder_denylist_fails(self):
        """Test test/placeholder inputs are rejected."""
        for word in ["test", "testing", "asdf", "qwerty", "hello", "hi"]:
            is_valid, reason = validate_context_answer(word)
            assert is_valid is False, f"'{word}' should be rejected"
            assert reason == "low_signal"
    
    def test_null_signals_fail(self):
        """Test null signal words are rejected."""
        for word in ["na", "n/a", "none", "nothing", "null", "idk"]:
            is_valid, reason = validate_context_answer(word)
            assert is_valid is False, f"'{word}' should be rejected"
            assert reason == "low_signal"
    
    def test_digit_only_fails(self):
        """Test digit-only input fails."""
        is_valid, reason = validate_context_answer("12345")
        assert is_valid is False
        assert reason == "low_signal"
    
    def test_symbol_only_fails(self):
        """Test symbol-only input fails."""
        is_valid, reason = validate_context_answer("???!!!")
        assert is_valid is False
        assert reason == "low_signal"
    
    def test_repeated_char_fails(self):
        """Test repeated single character fails."""
        is_valid, reason = validate_context_answer("aaaaaa")
        assert is_valid is False
        assert reason == "low_signal"
        
        is_valid, reason = validate_context_answer("......")
        assert is_valid is False
        assert reason == "low_signal"
    
    def test_whitespace_normalized(self):
        """Test leading/trailing whitespace is normalized."""
        is_valid, reason = validate_context_answer("   new project   ")
        assert is_valid is True
    
    def test_long_context_passes(self):
        """Test long context answers pass."""
        long_answer = "We are looking to modernize our legacy enterprise application"
        is_valid, reason = validate_context_answer(long_answer)
        assert is_valid is True


class TestNameValidation:
    """Tests for name validation."""
    
    @pytest.mark.parametrize("name", [
        "John",
        "Alice",
        "Bob Smith",
        "Mary Jane Watson",
        "Jean-Pierre",      # Hyphenated
        "O'Brien",          # Apostrophe
        "María García",     # International
        "Björk",            # Nordic
        "李明",              # Chinese
        "김영희",            # Korean
    ])
    def test_valid_names_pass(self, name):
        """Test valid names pass validation (including international)."""
        is_valid, reason = validate_name(name)
        assert is_valid is True, f"'{name}' should be valid"
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
    
    @pytest.mark.parametrize("placeholder", [
        "test",
        "test user",
        "asdf",
        "qwerty",
        "john doe",
        "anonymous",
        "guest",
        "admin",
    ])
    def test_placeholder_names_fail(self, placeholder):
        """Test placeholder names are rejected."""
        is_valid, reason = validate_name(placeholder)
        assert is_valid is False, f"'{placeholder}' should be rejected"
        assert reason == "placeholder_name"
    
    def test_single_char_word_fails(self):
        """Test names with single-char words fail."""
        is_valid, reason = validate_name("A")
        assert is_valid is False
        assert reason == "invalid_format"
    
    def test_too_many_words_fails(self):
        """Test more than 4 words fails."""
        is_valid, reason = validate_name("John James Smith Junior Extra")
        assert is_valid is False
        assert reason == "invalid_format"
    
    def test_digit_only_fails(self):
        """Test digit-only input fails."""
        is_valid, reason = validate_name("12345")
        assert is_valid is False
        assert reason == "no_alpha_signal"
    
    def test_repeated_chars_fail(self):
        """Test repeated characters fail as placeholder."""
        is_valid, reason = validate_name("aaaa")
        assert is_valid is False
        assert reason == "placeholder_name"
    
    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed before validation."""
        is_valid, reason = validate_name("  John Smith  ")
        assert is_valid is True
    
    def test_two_char_name_passes(self):
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
        """Test exploration shorter than 5 chars fails."""
        is_valid, reason = validate_exploration_input("abcd")
        assert is_valid is False
        assert reason == "too_short"
    
    def test_acknowledgment_words_fail(self):
        """Test acknowledgment words fail (even at 5+ chars)."""
        for word in ["okay", "yeah", "sure", "fine", "help", "maybe"]:
            is_valid, reason = validate_exploration_input(word)
            assert is_valid is False, f"'{word}' should be rejected"
            assert reason == "low_signal"
    
    def test_whitespace_only_fails(self):
        """Test whitespace-only input fails."""
        is_valid, reason = validate_exploration_input("   ")
        assert is_valid is False
        assert reason == "empty_input"
    
    def test_digit_only_fails(self):
        """Test digit-only fails."""
        is_valid, reason = validate_exploration_input("123456")
        assert is_valid is False
        assert reason == "low_signal"


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
            result = validator("meaningful test input here")
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
    
    def test_emoji_only_fails_all_validators(self):
        """Test emoji-only input fails all validators."""
        for validator in [validate_context_answer, validate_exploration_input]:
            is_valid, reason = validator("🎉🎊🎈")
            assert is_valid is False
            assert reason == "low_signal"
