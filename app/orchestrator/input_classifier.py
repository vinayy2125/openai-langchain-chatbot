# UC1 Input Classifier - Pre-State-Machine Gate
#
# ARCHITECTURE:
#   This is a CONTROL PLANE component.
#   It runs BEFORE the state machine.
#   It is deterministic and cheap.
#   Gibberish, ACKs, and negations are classified here.
#
# The orchestrator uses this to:
#   1. Block garbage before it pollutes slots
#   2. Force intents for control signals (NEGATION, GIBBERISH)
#   3. Allow LLM to handle STATEMENT/QUESTION

from enum import Enum
import re
import math
from typing import Optional


class InputClass(Enum):
    """Classification of user input for control routing."""
    ACK = "ack"              # Acknowledgment (yes, ok, sure)
    NEGATION = "negation"    # Rejection (no, nope, not interested)
    QUESTION = "question"    # Query (contains ?, interrogative words)
    GIBBERISH = "gibberish"  # Invalid input (keyboard mashing, random chars)
    STATEMENT = "statement"  # Valid statement (default)


def entropy(text: str) -> float:
    """
    Calculate Shannon entropy of text.
    
    High entropy indicates randomness (keyboard mashing).
    Normal English text has entropy ~4.0-4.5.
    Random characters have entropy >4.5.
    """
    if not text:
        return 0.0
    freq = {}
    for c in text.lower():
        freq[c] = freq.get(c, 0) + 1
    length = len(text)
    return -sum((f / length) * math.log2(f / length) for f in freq.values())


def alphabetic_ratio(text: str) -> float:
    """
    Calculate ratio of alphabetic characters to total length.
    
    Low ratio (<0.6) combined with high entropy indicates encoded/mixed junk.
    """
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def classify_input(text: str) -> InputClass:
    """
    Classify user input for control routing.
    
    This is a HARD PRE-GATE. Classifications here override LLM intent.
    
    Precedence:
        1. ACK patterns (exact match)
        2. NEGATION patterns (exact match)
        3. QUESTION patterns (contains ? or interrogative words)
        4. GIBBERISH (fails sanity checks)
        5. STATEMENT (default)
    
    Args:
        text: Raw user input
        
    Returns:
        InputClass: Classification for routing
    """
    if not text:
        return InputClass.STATEMENT
    
    t = text.lower().strip()
    
    # ==========================================================
    # ACK patterns (expanded set per review)
    # ==========================================================
    ACK_PATTERNS = {
        "yes", "yeah", "yep", "yup",
        "ok", "okay", "k",
        "sure", "right",
        "got it", "gotcha",
        "fine", "alright", "all right",
        "makes sense", "ok got it",
        "understood", "i see", "cool",
    }
    if t in ACK_PATTERNS:
        return InputClass.ACK
    
    # ==========================================================
    # NEGATION patterns
    # ==========================================================
    NEGATION_PATTERNS = {
        "no", "nope", "nah",
        "not really", "no thanks", "no thank you",
        "not interested", "i'm not interested",
        "never mind", "nevermind",
    }
    if t in NEGATION_PATTERNS:
        return InputClass.NEGATION
    
    # ==========================================================
    # QUESTION patterns
    # ==========================================================
    QUESTION_WORDS = {"who", "what", "how", "why", "when", "where", "which", "can", "could", "would", "should", "is", "are", "do", "does"}
    words = t.split()
    if "?" in t:
        return InputClass.QUESTION
    if words and words[0] in QUESTION_WORDS:
        return InputClass.QUESTION
    
    # ==========================================================
    # GIBBERISH detection (entropy + alphabetic ratio + lexical)
    # ==========================================================
    
    # Rule 1: Too short
    if len(t) < 3:
        return InputClass.GIBBERISH
    
    # Rule 2: No vowels (keyboard mashing like "dfgh", "lkjh")
    if not any(c in "aeiou" for c in t):
        return InputClass.GIBBERISH
    
    # Rule 3: No valid 3+ letter words (lexical sanity)
    valid_words = re.findall(r"[a-z]{3,}", t)
    if len(valid_words) == 0:
        return InputClass.GIBBERISH
    
    # Rule 4: Low alphabetic ratio + high entropy = encoded/mixed junk
    alpha_ratio = alphabetic_ratio(t)
    text_entropy = entropy(t)
    if alpha_ratio < 0.6 and text_entropy > 3.5:
        return InputClass.GIBBERISH
    
    # ==========================================================
    # Default: STATEMENT
    # ==========================================================
    return InputClass.STATEMENT
