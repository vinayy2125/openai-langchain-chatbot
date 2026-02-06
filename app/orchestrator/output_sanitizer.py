# UC1 Output Sanitizer - Belt-and-Suspenders Safety Layer
#
# ARCHITECTURE RULE: This is the FINAL safety check before
# any response goes to the user. Even if LLM somehow generates
# forbidden content, this layer catches it.
#
# Checks:
# 1. Forbidden topic regex matching
# 2. No unauthorized questions in synthesis states
# 3. CTA count validation in recommendation state

import re
import os
from typing import Set, Optional
from app.orchestrator.uc1_config import UC1Config
from app.orchestrator.state_machine import UC1State
from app.logger import get_logger

logger = get_logger("output_sanitizer")

# Environment mode: DEV or PROD
# In DEV: sanitize() raises immediately on violation (use sanitize(), not safe_sanitize())
# In PROD: safe_sanitize() provides fallbacks
ENV_MODE = os.getenv("ENV_MODE", "PROD").upper()


class ForbiddenTopicViolation(Exception):
    """Raised when output contains a forbidden topic."""
    def __init__(self, topic: str, context: str = ""):
        self.topic = topic
        self.context = context
        super().__init__(f"Forbidden topic detected: '{topic}' in output: '{context[:100]}...'")


class UnauthorizedQuestionViolation(Exception):
    """Raised when output contains an unauthorized question."""
    def __init__(self, question_count: int, state: str):
        self.question_count = question_count
        self.state = state
        super().__init__(f"Unauthorized question(s) detected: {question_count} '?' in state {state}")


class LLMOutputSanitizer:
    """
    Final safety layer for LLM output.
    
    This runs AFTER the LLM adapter and BEFORE sending to user.
    It catches any policy violations that slipped through.
    
    CHECKS:
    1. Forbidden topics - regex match against forbidden_topics list
    2. Unauthorized questions - no '?' in AI_SYNTHESIS or CONSULTATIVE_ALTERNATIVES states
    3. CTA validation - only configured CTAs in RECOMMENDATION state
    """
    
    def __init__(self, config: UC1Config):
        """
        Initialize sanitizer with config.
        
        Args:
            config: The validated UC1Config
        """
        self.config = config
        
        # Pre-compile forbidden topic patterns for efficiency
        # Use word boundaries to avoid false positives (e.g., "cost" in "costume")
        self._forbidden_patterns = []
        for topic in config.forbidden_topics:
            # Escape special regex characters
            escaped = re.escape(topic)
            # Create pattern with word boundaries
            pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
            self._forbidden_patterns.append((topic, pattern))
        
        logger.info(f"[Sanitizer] Initialized with {len(self._forbidden_patterns)} forbidden patterns")
    
    def sanitize(
        self,
        output: str,
        state: UC1State,
    ) -> str:
        """
        Sanitize LLM output before sending to user.
        
        Args:
            output: The raw LLM output
            state: Current conversation state
        
        Returns:
            str: The sanitized output (same as input if valid)
        
        Raises:
            ForbiddenTopicViolation: If forbidden topic detected
            UnauthorizedQuestionViolation: If unauthorized question detected
        """
        if not output:
            return output
        
        # Check 1: Forbidden topics
        self._check_forbidden_topics(output)
        
        # Check 2: No questions in AI_SYNTHESIS or CONSULTATIVE_ALTERNATIVES
        if state in (UC1State.AI_SYNTHESIS, UC1State.CONSULTATIVE_ALTERNATIVES):
            self._check_no_questions(output, state)
        
        # Check 3: CTA validation in RECOMMENDATION state
        if state == UC1State.RECOMMENDATION:
            self._check_valid_ctas(output)
        
        logger.debug(f"[Sanitizer] Output passed all checks for state {state.value}")
        return output
    
    def _check_forbidden_topics(self, output: str) -> None:
        """
        Check for forbidden topics in output.
        
        Raises:
            ForbiddenTopicViolation: If match found
        """
        for topic, pattern in self._forbidden_patterns:
            if pattern.search(output):
                if ENV_MODE == "DEV":
                    logger.error(f"[FATAL DEV] Sanitizer violation: forbidden topic '{topic}'")
                else:
                    logger.error(f"[Sanitizer] Forbidden topic detected: '{topic}'")
                raise ForbiddenTopicViolation(topic, output)
    
    def _check_no_questions(self, output: str, state: UC1State) -> None:
        """
        Check that output contains no question marks in restricted states.
        
        In AI_SYNTHESIS and CONSULTATIVE_ALTERNATIVES, the system has already
        asked the ONE allowed question - LLM should not add more.
        
        Raises:
            UnauthorizedQuestionViolation: If question marks found
        """
        question_count = output.count('?')
        if question_count > 0:
            logger.error(f"[Sanitizer] Unauthorized questions detected: {question_count} in {state.value}")
            raise UnauthorizedQuestionViolation(question_count, state.value)
    
    def _check_valid_ctas(self, output: str) -> None:
        """
        Check that only configured CTAs appear in output.
        
        This is a soft check - we just log warnings for now since
        CTA formatting might vary slightly.
        """
        valid_ctas = set(cta.choice.lower() for cta in self.config.exit_ctas)
        
        # Look for CTA-like patterns (lines starting with - or numbered)
        cta_patterns = re.findall(r'(?:^[-•]\s*|^\d+\.\s*)(.+)$', output, re.MULTILINE)
        
        for potential_cta in cta_patterns:
            potential_lower = potential_cta.strip().lower()
            # Check if any valid CTA is contained in this line
            if not any(valid_cta in potential_lower for valid_cta in valid_ctas):
                logger.warning(f"[Sanitizer] Potential unrecognized CTA: '{potential_cta.strip()}'")
    
    def safe_sanitize(
        self,
        output: str,
        state: UC1State,
    ) -> tuple:
        """
        Sanitize with safety fallback - returns tuple of (output, error).
        
        This is the recommended method for production use.
        Instead of raising exceptions, it returns a tuple so the
        orchestrator can handle violations gracefully.
        
        Args:
            output: The raw LLM output
            state: Current conversation state
        
        Returns:
            tuple: (sanitized_output, error_or_none)
                   If error is not None, sanitized_output will be a fallback message
        """
        try:
            sanitized = self.sanitize(output, state)
            return (sanitized, None)
        except ForbiddenTopicViolation as e:
            logger.error(f"[Sanitizer] Forbidden topic violation: {e}")
            fallback = "I'd be happy to tell you more about our approach. What specific aspect interests you most?"
            return (fallback, e)
        except UnauthorizedQuestionViolation as e:
            logger.error(f"[Sanitizer] Question violation: {e}")
            # Remove question marks and return
            cleaned = output.replace('?', '.')
            return (cleaned, e)
        except Exception as e:
            logger.exception(f"[Sanitizer] Unexpected error: {e}")
            fallback = "Let me help you with that."
            return (fallback, e)
