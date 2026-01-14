# Tests for UC1 Input Classifier
#
# These tests verify input classification for control routing:
# - ACK patterns (yes, ok, sure, etc.)
# - Negation patterns (no, nope, not interested)
# - Question detection (interrogative words, ?)
# - Gibberish detection (keyboard mashing, random chars)
# - Statement fallback (default)

import pytest
from app.orchestrator.input_classifier import (
    classify_input,
    InputClass,
    entropy,
    alphabetic_ratio,
)


class TestACKPatterns:
    """Tests for acknowledgment pattern recognition."""
    
    @pytest.mark.parametrize("input_text", [
        "yes", "yeah", "yep", "yup",
        "ok", "okay", "k",
        "sure", "right",
        "got it", "gotcha",
        "fine", "alright", "all right",
        "makes sense", "ok got it",
        "understood", "i see", "cool",
    ])
    def test_ack_patterns_recognized(self, input_text):
        """Test all ACK patterns are correctly classified."""
        assert classify_input(input_text) == InputClass.ACK
    
    def test_ack_case_insensitive(self):
        """Test ACK detection is case insensitive."""
        assert classify_input("YES") == InputClass.ACK
        assert classify_input("Ok") == InputClass.ACK
        assert classify_input("SURE") == InputClass.ACK
    
    def test_ack_with_whitespace(self):
        """Test ACK detection with leading/trailing whitespace."""
        assert classify_input("  yes  ") == InputClass.ACK
        assert classify_input("\tok\n") == InputClass.ACK


class TestNegationPatterns:
    """Tests for negation pattern recognition."""
    
    @pytest.mark.parametrize("input_text", [
        "no", "nope", "nah",
        "not really", "no thanks", "no thank you",
        "not interested", "i'm not interested",
        "never mind", "nevermind",
    ])
    def test_negation_patterns_recognized(self, input_text):
        """Test all negation patterns are correctly classified."""
        assert classify_input(input_text) == InputClass.NEGATION
    
    def test_negation_case_insensitive(self):
        """Test negation detection is case insensitive."""
        assert classify_input("NO") == InputClass.NEGATION
        assert classify_input("Nope") == InputClass.NEGATION
        assert classify_input("NOT INTERESTED") == InputClass.NEGATION


class TestQuestionDetection:
    """Tests for question detection."""
    
    def test_question_mark_detected(self):
        """Test questions ending with ? are detected."""
        assert classify_input("what services do you offer?") == InputClass.QUESTION
        assert classify_input("how much does this cost?") == InputClass.QUESTION
        assert classify_input("why?") == InputClass.QUESTION
    
    @pytest.mark.parametrize("input_text", [
        "what services do you offer",
        "how can you help me",
        "why should I choose you",
        "when can we start",
        "where are you located",
        "which option is best",
        "who will work on my project",
        "can you do this",
        "could you help me",
        "would you recommend this",
        "should I proceed",
        "is this available",
        "are you certified",
        "do you have experience",
        "does this work with Python",
    ])
    def test_interrogative_words_detected(self, input_text):
        """Test questions starting with interrogative words are detected."""
        assert classify_input(input_text) == InputClass.QUESTION


class TestGibberishDetection:
    """Tests for gibberish/invalid input detection."""
    
    def test_too_short_is_gibberish(self):
        """Test inputs shorter than 3 chars are gibberish."""
        assert classify_input("ab") == InputClass.GIBBERISH
        assert classify_input("x") == InputClass.GIBBERISH
    
    def test_no_vowels_is_gibberish(self):
        """Test keyboard mashing without vowels is gibberish."""
        assert classify_input("dfgh") == InputClass.GIBBERISH
        assert classify_input("lkjh") == InputClass.GIBBERISH
        assert classify_input("qwrt") == InputClass.GIBBERISH
    
    def test_no_valid_words_is_gibberish(self):
        """Test inputs without valid 3+ letter words are gibberish."""
        assert classify_input("x y z") == InputClass.GIBBERISH
        assert classify_input("ab cd") == InputClass.GIBBERISH
    
    def test_low_alpha_high_entropy_is_gibberish(self):
        """Test mixed junk with low alphabetic ratio is gibberish."""
        assert classify_input("a1b2c3d4e5") == InputClass.GIBBERISH
        assert classify_input("!@#$%abc") == InputClass.GIBBERISH
    
    def test_valid_short_words_not_gibberish(self):
        """Test valid short inputs are not flagged as gibberish."""
        # Has vowels and valid words
        assert classify_input("hello") != InputClass.GIBBERISH
        assert classify_input("help me") != InputClass.GIBBERISH


class TestStatementFallback:
    """Tests for statement (default) classification."""
    
    def test_normal_statements_classified(self):
        """Test normal statements are classified correctly."""
        assert classify_input("I want to build a new product") == InputClass.STATEMENT
        assert classify_input("We need mobile app development") == InputClass.STATEMENT
        assert classify_input("Looking for AI solutions") == InputClass.STATEMENT
    
    def test_empty_string_is_statement(self):
        """Test empty string returns statement (safe default)."""
        assert classify_input("") == InputClass.STATEMENT
    
    def test_complex_valid_input(self):
        """Test complex valid inputs are statements."""
        assert classify_input("Our company is looking to modernize our legacy systems and migrate to the cloud") == InputClass.STATEMENT


class TestEntropyFunction:
    """Tests for entropy calculation helper."""
    
    def test_empty_string_zero_entropy(self):
        """Test empty string has zero entropy."""
        assert entropy("") == 0.0
    
    def test_single_char_zero_entropy(self):
        """Test repeated single char has zero entropy."""
        assert entropy("aaaa") == 0.0
    
    def test_normal_text_moderate_entropy(self):
        """Test normal English text has moderate entropy."""
        e = entropy("hello world")
        assert 2.0 < e < 4.0
    
    def test_random_chars_high_entropy(self):
        """Test random characters have higher entropy."""
        e = entropy("qwertyuiopasdfghjkl")
        assert e > 3.5


class TestAlphabeticRatio:
    """Tests for alphabetic ratio calculation helper."""
    
    def test_empty_string_zero_ratio(self):
        """Test empty string has zero ratio."""
        assert alphabetic_ratio("") == 0.0
    
    def test_all_alpha_full_ratio(self):
        """Test all-alphabetic string has ratio 1.0."""
        assert alphabetic_ratio("hello") == 1.0
    
    def test_all_numbers_zero_ratio(self):
        """Test all-numeric string has ratio 0.0."""
        assert alphabetic_ratio("12345") == 0.0
    
    def test_mixed_partial_ratio(self):
        """Test mixed content has partial ratio."""
        ratio = alphabetic_ratio("abc123")
        assert 0.4 < ratio < 0.6


class TestEdgeCases:
    """Edge case tests for input classifier."""
    
    def test_none_like_handling(self):
        """Test handling of None-like inputs."""
        # Empty string should work
        assert classify_input("") == InputClass.STATEMENT
    
    def test_unicode_input(self):
        """Test unicode input handling."""
        # Unicode with valid words should be statement
        result = classify_input("hello 你好")
        assert result in (InputClass.STATEMENT, InputClass.GIBBERISH)  # Depends on implementation
    
    def test_mixed_case_ack(self):
        """Test mixed case ACK patterns."""
        assert classify_input("YeS") == InputClass.ACK
        assert classify_input("gOt It") == InputClass.ACK
    
    def test_ack_vs_statement_boundary(self):
        """Test boundary between ACK and statement."""
        # "yes" is ACK
        assert classify_input("yes") == InputClass.ACK
        # "yes we need help" is statement (longer context)
        assert classify_input("yes we need help") == InputClass.STATEMENT
