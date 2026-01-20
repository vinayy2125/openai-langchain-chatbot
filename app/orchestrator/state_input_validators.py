# UC1 State Input Validators - Pre-Slot-Mutation Gate
#
# ARCHITECTURE:
#   Validation MUST occur BEFORE any slot mutation.
#   Order: validate → retry OR set_slot → transition
#   NOT: set_slot → validate → rollback
#
# These validators are deterministic, cheap, and run in the orchestrator
# BEFORE LLM is consulted.

import re
import unicodedata
from typing import Tuple, Set

# =============================================================================
# DENYLISTS
# =============================================================================

ACKNOWLEDGMENT_DENYLIST: Set[str] = frozenset({
    # Filler / acknowledgments
    "ok", "okay", "yes", "no", "ya", "yeah", "yea", "yep", "nope", "nah",
    "hm", "hmm", "hmmm", "uh", "um", "mhm", "uh-huh",
    "fine", "cool", "sure", "alright", "right", "k", "kk",
    # Test / placeholder
    "test", "testing", "asdf", "qwerty", "zxcv", "abc", "xyz", "asd",
    "hello", "hi", "hey",
    # Null signals
    "na", "n/a", "none", "nothing", "null", "nil", "idk", "dunno",
    # Ambiguous
    "maybe", "perhaps", "anything", "whatever", "something", "help",
})

NAME_PLACEHOLDER_DENYLIST: Set[str] = frozenset({
    "test", "test user", "testuser", "user", "name", "your name",
    "asdf", "qwerty", "abc", "xyz", "asd", "anonymous", "anon",
    "nobody", "noname", "no name", "na", "n/a", "none", "null",
    "admin", "administrator", "guest", "sample", "example",
    "john doe", "jane doe", "foo", "bar", "baz", "xxx", "yyy", "zzz",
})


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    if not text:
        return ""
    return " ".join(text.split())


def _has_alpha(text: str) -> bool:
    """Check if text contains at least one alphabetic character (unicode-aware)."""
    return any(unicodedata.category(c).startswith('L') for c in text)


def _is_repeated_char(text: str) -> bool:
    """Check if text is just repeated single character (e.g., 'aaa', '...')."""
    stripped = text.replace(" ", "")
    if len(stripped) < 2:
        return False
    return len(set(stripped.lower())) == 1


def _is_digit_only(text: str) -> bool:
    """Check if text contains only digits and whitespace."""
    return text.replace(" ", "").isdigit()


def _is_symbol_or_emoji_only(text: str) -> bool:
    """Check if text contains only symbols, punctuation, or emoji (no letters/digits)."""
    for c in text:
        if c.isspace():
            continue
        cat = unicodedata.category(c)
        # L = Letter, N = Number
        if cat.startswith('L') or cat.startswith('N'):
            return False
    return True


def _check_global_rules(text: str) -> Tuple[bool, str]:
    """
    Apply global validation rules.
    
    Returns:
        (True, "") if passes all global rules
        (False, failure_reason) if fails
    """
    if not text:
        return False, "empty_input"
    
    normalized = _normalize(text)
    
    if not normalized:
        return False, "empty_input"
    
    # Symbol/emoji only
    if _is_symbol_or_emoji_only(normalized):
        return False, "low_signal"
    
    # Digit only
    if _is_digit_only(normalized):
        return False, "low_signal"
    
    # Repeated single character
    if _is_repeated_char(normalized):
        return False, "low_signal"
    
    # Acknowledgment denylist
    if normalized.lower() in ACKNOWLEDGMENT_DENYLIST:
        return False, "low_signal"
    
    return True, ""


# =============================================================================
# VALIDATORS
# =============================================================================

def validate_context_answer(text: str) -> Tuple[bool, str]:
    """
    Validate context/discovery answer is substantive.
    
    Purpose: Gate discovery/qualification inputs.
    Accept only inputs that contribute information.
    
    Rules:
        - Must pass global rules
        - Must contain alphabetic signal
        - Minimum semantic length > trivial acknowledgments
        - Reject anything that would cause the LLM to hallucinate context
    
    Args:
        text: Raw user input
        
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
    """
    # Global rules first
    passed, reason = _check_global_rules(text)
    if not passed:
        return False, reason
    
    normalized = _normalize(text)
    
    # Must contain alphabetic signal
    if not _has_alpha(normalized):
        return False, "no_alpha_signal"
    
    # Minimum length for semantic content (filters single chars)
    # Allow short meaningful words like "new", "existing", "mobile", "web"
    if len(normalized) < 2:
        return False, "too_short"
    
    return True, ""


def validate_name(text: str) -> Tuple[bool, str]:
    """
    Validate name format for lead generation.
    
    Purpose: Capture real human names.
    
    Rules:
        - Allow international variance (unicode letters)
        - Allow spaces, hyphens, apostrophes
        - 1-4 words
        - Each word >= 2 characters
        - Reject placeholders
        - Prefer false-negative reduction over false-positive rejection
    
    Args:
        text: Raw user input
        
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
    """
    if not text:
        return False, "empty_input"
    
    normalized = _normalize(text)
    
    if not normalized:
        return False, "empty_input"
    
    # Check placeholder denylist
    if normalized.lower() in NAME_PLACEHOLDER_DENYLIST:
        return False, "placeholder_name"
    
    # Must contain alphabetic characters
    if not _has_alpha(normalized):
        return False, "no_alpha_signal"
    
    # Reject if mostly digits
    alpha_count = sum(1 for c in normalized if unicodedata.category(c).startswith('L'))
    if alpha_count < 2:
        return False, "invalid_format"
    
    # Reject repeated characters (e.g., "aaaa", "xxxx")
    if _is_repeated_char(normalized):
        return False, "placeholder_name"
    
    # Split into words (allowing hyphens and apostrophes within words)
    # Pattern: split on whitespace only, keep hyphenated/apostrophe names together
    words = normalized.split()
    
    # 1-4 words
    if len(words) < 1 or len(words) > 4:
        return False, "invalid_format"
    
    # Each word must have at least 2 characters
    for word in words:
        # Remove allowed punctuation for length check
        clean_word = word.replace("-", "").replace("'", "").replace("'", "")
        if len(clean_word) < 2:
            return False, "invalid_format"
        
        # Each word must contain at least one letter
        if not _has_alpha(word):
            return False, "invalid_format"
    
    # Validate allowed characters: letters, spaces, hyphens, apostrophes
    allowed_pattern = re.compile(r"^[\w\s\-'']+$", re.UNICODE)
    if not allowed_pattern.match(normalized):
        # Check if invalid due to symbols
        for c in normalized:
            if c.isspace() or c in "-''":
                continue
            cat = unicodedata.category(c)
            if not cat.startswith('L'):
                return False, "invalid_format"
    
    return True, ""


def validate_exploration_input(text: str) -> Tuple[bool, str]:
    """
    Validate exploration input requires expressed intent.
    
    Purpose: Exploration requires substantive expressed intent, not acknowledgment.
    
    Rules:
        - Must pass global rules
        - Must contain alphabetic signal
        - Minimum length higher than context answers
        - "help", "ok", "anything" are INVALID
        - Accept only inputs that justify state progression
    
    Args:
        text: Raw user input
        
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
    """
    # Global rules first (includes acknowledgment denylist)
    passed, reason = _check_global_rules(text)
    if not passed:
        return False, reason
    
    normalized = _normalize(text)
    
    # Must contain alphabetic signal
    if not _has_alpha(normalized):
        return False, "no_alpha_signal"
    
    # Higher minimum length for exploration (requires real question/intent)
    if len(normalized) < 5:
        return False, "too_short"
    
    # Word count check - exploration should have substance
    words = [w for w in normalized.split() if _has_alpha(w)]
    if len(words) < 1:
        return False, "low_signal"
    
    return True, ""


# =============================================================================
# STATE-SCOPED VALIDATION (Master Entry Point)
# =============================================================================

# States where acknowledgment tokens ARE allowed (confirmation/exit/consent)
ACK_ALLOWED_STATES = frozenset({
    "exit",
    "confirmation", 
    "consent",
    "ai_synthesis",  # Transitional state, ACK is valid
    "recommendation",  # User confirming recommendation
})

# States where acknowledgment tokens are REJECTED (discovery/capture)
ACK_REJECTED_STATES = frozenset({
    "context_question",
    "name_capture",
    "capability_selection",
    "exploration_layer",
    "free_exploration",
    "consultative_alternatives",
})


def validate_input_for_state(user_input: str, state_value: str) -> Tuple[bool, str]:
    """
    Master state-scoped validation entry point.
    
    ARCHITECTURE LAW:
        This MUST run BEFORE any slot mutation.
        This MUST run BEFORE routing to prompts.
        Validation does NOT depend on which prompt is chosen.
    
    Order: input → validate_input_for_state → reject OR continue → route → execute
    
    Args:
        user_input: Raw user input
        state_value: Current UC1 state value (string, e.g., "context_question")
    
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
        
    Failure reasons:
        - "empty_input": No content
        - "low_signal": Gibberish, repeated chars, acknowledgment tokens
        - "no_alpha_signal": No alphabetic content
        - "too_short": Below minimum length
        - "placeholder_name": Known placeholder/test name
        - "invalid_format": Name format violation
        - "ack_not_allowed": Acknowledgment token in discovery state
    """
    if not user_input:
        return False, "empty_input"
    
    normalized = _normalize(user_input)
    state_lower = state_value.lower() if state_value else ""
    
    # ==========================================================================
    # STEP 1: Check if this is an ACK token
    # ==========================================================================
    is_ack_token = normalized.lower() in ACKNOWLEDGMENT_DENYLIST
    
    # ==========================================================================
    # STEP 2: State-scoped ACK handling
    # ==========================================================================
    if is_ack_token:
        # ACK tokens are ALLOWED in confirmation/exit/consent states
        if state_lower in ACK_ALLOWED_STATES:
            # Valid ACK in appropriate state - pass through for intent classification
            return True, ""
        
        # ACK tokens are REJECTED in discovery/capture states
        if state_lower in ACK_REJECTED_STATES:
            return False, "ack_not_allowed"
        
        # Default: reject ACK in unknown states (fail-safe)
        return False, "ack_not_allowed"
    
    # ==========================================================================
    # STEP 3: State-specific validation for non-ACK inputs
    # ==========================================================================
    if state_lower == "context_question":
        return validate_context_answer(user_input)
    
    elif state_lower == "name_capture":
        return validate_name(user_input)
    
    elif state_lower in ("exploration_layer", "free_exploration"):
        return validate_exploration_input(user_input)
    
    elif state_lower == "capability_selection":
        # Capability selection validates via button matching in orchestrator
        # Here we just ensure non-empty
        if not normalized:
            return False, "empty_input"
        return True, ""
    
    # ==========================================================================
    # STEP 4: Default validation (global rules only)
    # ==========================================================================
    return _check_global_rules(user_input)


def is_ack_token(user_input: str) -> bool:
    """
    Quick check if input is an acknowledgment token.
    
    Used by orchestrator to decide intent routing.
    """
    if not user_input:
        return False
    normalized = _normalize(user_input)
    return normalized.lower() in ACKNOWLEDGMENT_DENYLIST

