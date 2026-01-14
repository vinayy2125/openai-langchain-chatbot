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
from typing import Tuple


def validate_context_answer(text: str) -> Tuple[bool, str]:
    """
    Validate context answer is substantive.
    
    Rules:
        - Not empty
        - At least 5 characters
        - At least 2 words
    
    Args:
        text: Raw user input
        
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
    """
    if not text:
        return False, "empty_input"
    
    stripped = text.strip()
    
    if len(stripped) < 5:
        return False, "too_short"
    
    words = stripped.split()
    if len(words) < 2:
        return False, "too_few_words"
    
    return True, ""


def validate_name(text: str) -> Tuple[bool, str]:
    """
    Validate name format: 1-3 alphabetic words, 2+ chars each.
    
    Write-once enforced at orchestrator level (skip if already set).
    
    Args:
        text: Raw user input
        
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
    """
    if not text:
        return False, "empty_input"
    
    stripped = text.strip()
    
    # Regex: 1-3 alphabetic words, each 2+ characters
    NAME_RE = re.compile(r"^[A-Za-z]{2,}(?:\s[A-Za-z]{2,}){0,2}$")
    
    if not NAME_RE.match(stripped):
        return False, "invalid_format"
    
    return True, ""


def validate_exploration_input(text: str) -> Tuple[bool, str]:
    """
    Validate exploration input is substantive.
    
    Rules:
        - Not empty
        - At least 3 characters
    
    Args:
        text: Raw user input
        
    Returns:
        Tuple[bool, str]: (is_valid, failure_reason)
    """
    if not text:
        return False, "empty_input"
    
    stripped = text.strip()
    
    if len(stripped) < 3:
        return False, "too_short"
    
    return True, ""
