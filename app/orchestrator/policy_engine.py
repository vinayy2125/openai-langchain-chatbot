# UC1 Policy Engine - Intent Arbitration & Adaptive Control
#
# ARCHITECTURE RULE: Policy Engine governs WHEN to apply business rules.
# - Orchestrator enforces constraints
# - Policy Engine decides business intent
# - LLM generates language
#
# ARBITRATION PRIORITY (strict order):
# 1. Explicit user intent (always overrides UC1 compliance)
# 2. Resistance signals
# 3. Topic drift
# 4. UC1 compliance (lowest priority)
#
# LEAD CAPTURE GOVERNANCE:
# - Never time-based or turn-based
# - Only signal-based (explicit intent, CTA selection, commercial intent)

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set
from app.logger import get_logger

logger = get_logger("policy_engine")


class PolicyDecision(str, Enum):
    """
    Possible outcomes from policy evaluation.
    
    The orchestrator receives this and routes accordingly.
    """
    PROCEED = "proceed"           # Continue normal UC1 flow
    SKIP = "skip"                 # Skip current state, user wants to move on
    PIVOT = "pivot"               # User diverged, switch to FREE_EXPLORATION
    CAPTURE_LEAD = "capture_lead" # Good moment for lead capture (signal-based)
    SUPPRESS_ASK = "suppress_ask" # Don't ask for lead now (suppression active)
    RESUME_UC1 = "resume_uc1"     # User explicitly wants to resume UC1 flow


class UserIntent(str, Enum):
    """
    Detected user intent categories.
    
    Intent detection drives policy decisions.
    """
    CONTINUE_FLOW = "continue_flow"       # User is answering our question
    ASK_QUESTION = "ask_question"         # User is asking their own question
    REQUEST_CONTACT = "request_contact"   # Explicit contact request
    EXPLORE_SERVICES = "explore_services" # Browsing without commitment
    RESISTANCE = "resistance"             # User shows pushback or divergence
    JUST_BROWSING = "just_browsing"       # User explicitly says "just browsing"
    RESUME_GUIDED = "resume_guided"       # User wants to resume UC1 journey


# Explicit contact intent patterns
CONTACT_INTENT_PATTERNS = [
    r"\b(contact|call|email|reach|discuss|talk|speak|chat)\s*(me|us|with)?\b",
    r"\b(get\s+in\s+touch|schedule|book|demo|meeting|consultation)\b",
    r"\b(i('d| would)?\s+like\s+to\s+discuss)\b",
    r"\b(my\s+(email|phone|number|contact))\b",
    r"\b(send\s+me|forward\s+me)\b",
]

# Resistance/divergence patterns
RESISTANCE_PATTERNS = [
    r"\b(not\s+now|later|maybe|no\s+thanks|skip|don't\s+want)\b",
    r"\b(stop\s+asking|already\s+told|asked\s+before|i\s+said)\b",
    r"\b(that's\s+not|you're\s+wrong|incorrect|nope)\b",
    r"\b(can\s+we\s+(just|move)|let's\s+(skip|move))\b",
]

# Just browsing patterns (permanent suppression trigger)
BROWSING_PATTERNS = [
    r"\b(just\s+browsing|just\s+looking|exploring|checking\s+out)\b",
    r"\b(no\s+commitment|not\s+ready|not\s+sure\s+yet)\b",
    r"\b(don't\s+(need|want)\s+(to\s+)?(talk|call|email))\b",
]

# Question patterns (user asking their own question)
QUESTION_PATTERNS = [
    r"\?$",  # Ends with question mark
    r"^(what|how|why|when|where|which|who|can|do|does|is|are|will|would)\b",
    r"\b(tell\s+me|explain|describe|show\s+me|what\s+about)\b",
]


@dataclass
class PolicyContext:
    """
    Context for policy evaluation.
    
    Contains all signals needed to make policy decisions.
    """
    current_state: str
    user_input: str
    engagement_score: float = 0.0
    resistance_score: float = 0.0
    exchange_count: int = 0
    is_in_uc1: bool = False
    is_in_free_exploration: bool = False
    
    # Lead capture tracking
    lead_capture_asked_count: int = 0
    lead_capture_suppressed_until: int = 0
    lead_capture_permanently_suppressed: bool = False
    
    # UC1 state tracking
    uc1_paused: bool = False
    resume_uc1: bool = False
    
    # History for pattern detection
    state_history: List[str] = field(default_factory=list)
    recent_inputs: List[str] = field(default_factory=list)


class ConversationPolicy:
    """
    Policy engine for intent-governed conversation flow.
    
    CORE PRINCIPLE: User intent always overrides UC1 compliance.
    
    Responsibilities:
    1. Intent arbitration - What does the user want?
    2. Resistance detection - Is the user pushing back?
    3. Lead capture governance - Is now the right time?
    4. Flow control - Proceed, skip, or pivot?
    """
    
    # Configuration
    ENGAGEMENT_THRESHOLD = 1.2  # Minimum score for implicit lead capture
    RESISTANCE_THRESHOLD = 0.5  # Above this, user is resisting
    SUPPRESSION_WINDOW = 10     # Exchanges before re-asking
    MAX_LEAD_ASKS = 2           # After this, permanent suppression
    
    def __init__(self, session_id: str):
        """
        Initialize policy engine for a session.
        
        Args:
            session_id: The session identifier
        """
        self.session_id = session_id
        logger.info(f"[PolicyEngine] Initialized for session: {session_id}")
    
    @staticmethod
    def arbitration_priority() -> List[str]:
        """
        Return the strict priority order for policy arbitration.
        
        CRITICAL: Explicit user intent ALWAYS overrides UC1 compliance.
        This prevents funnel bias from leaking into policy decisions.
        """
        return [
            "explicit_user_intent",  # Highest priority
            "resistance",
            "topic_drift",
            "uc1_compliance",        # Lowest priority
        ]
    
    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        """
        Main policy evaluation entry point.
        
        Evaluates intent in strict priority order:
        1. Explicit user intent (contact request, resume UC1)
        2. Resistance signals
        3. Topic drift (user asking questions)
        4. UC1 compliance (normal flow)
        
        Args:
            context: PolicyContext with all evaluation signals
            
        Returns:
            PolicyDecision: What the orchestrator should do
        """
        user_input = context.user_input.lower().strip()
        
        # 1. HIGHEST PRIORITY: Explicit user intent
        intent = self.detect_intent(context.user_input)
        
        if intent == UserIntent.REQUEST_CONTACT:
            if self._can_capture_lead(context):
                logger.info(f"[PolicyEngine] Explicit contact request detected")
                return PolicyDecision.CAPTURE_LEAD
        
        if intent == UserIntent.RESUME_GUIDED:
            logger.info(f"[PolicyEngine] User wants to resume UC1")
            return PolicyDecision.RESUME_UC1
        
        if intent == UserIntent.JUST_BROWSING:
            logger.info(f"[PolicyEngine] User just browsing - permanent suppression")
            return PolicyDecision.SUPPRESS_ASK
        
        # 2. Resistance signals
        resistance = self.detect_resistance(context.user_input, context.state_history)
        if resistance > self.RESISTANCE_THRESHOLD:
            logger.info(f"[PolicyEngine] High resistance detected: {resistance:.2f}")
            return PolicyDecision.SKIP
        
        # 3. Topic drift (user asking their own question)
        if intent == UserIntent.ASK_QUESTION and context.is_in_uc1:
            logger.info(f"[PolicyEngine] User asking question during UC1 - pivot to exploration")
            return PolicyDecision.PIVOT
        
        # 4. UC1 compliance (normal flow)
        return PolicyDecision.PROCEED
    
    def detect_intent(self, user_input: str) -> UserIntent:
        """
        Detect user intent from their message.
        
        Uses pattern matching for explicit signals,
        with graceful fallback to CONTINUE_FLOW.
        
        Args:
            user_input: The user's message
            
        Returns:
            UserIntent: Detected intent category
        """
        text = user_input.lower().strip()
        
        # Check for explicit contact request
        for pattern in CONTACT_INTENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return UserIntent.REQUEST_CONTACT
        
        # Check for just browsing (permanent suppression)
        for pattern in BROWSING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return UserIntent.JUST_BROWSING
        
        # Check for resistance/pushback
        for pattern in RESISTANCE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return UserIntent.RESISTANCE
        
        # Check for user question
        for pattern in QUESTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return UserIntent.ASK_QUESTION
        
        # Check for resume guided flow
        resume_patterns = [
            r"\b(continue|resume|back\s+to|go\s+back)\b",
            r"\b(show\s+me\s+options|what\s+were\s+we)\b",
        ]
        for pattern in resume_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return UserIntent.RESUME_GUIDED
        
        # Default: user is continuing the flow
        return UserIntent.CONTINUE_FLOW
    
    def detect_resistance(
        self, 
        user_input: str, 
        state_history: List[str]
    ) -> float:
        """
        Calculate resistance score from user behavior.
        
        Resistance indicates user is pushing back against the flow.
        Higher scores mean more resistance.
        
        Args:
            user_input: Current user message
            state_history: List of previous states
            
        Returns:
            float: Resistance score (0.0 to 1.0)
        """
        score = 0.0
        text = user_input.lower().strip()
        
        # Pattern-based resistance
        for pattern in RESISTANCE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
        
        # Very short responses can indicate disengagement
        if len(text) < 3 and text not in ['yes', 'no', 'ok', 'hi']:
            score += 0.1
        
        # Repeated same-state visits indicate struggle
        if len(state_history) >= 2:
            if state_history[-1] == state_history[-2]:
                score += 0.2
        
        # Cap at 1.0
        return min(1.0, score)
    
    def should_capture_lead(self, context: PolicyContext) -> Tuple[bool, str]:
        """
        Determine if lead capture should be triggered.
        
        SIGNAL-BASED ONLY. Never time-based or turn-based.
        
        Approved triggers (ranked):
        1. Explicit intent ("contact me", "email me", etc.)
        2. CTA selection ("Discuss my requirement")
        3. Implicit commercial intent + high engagement
        
        Args:
            context: PolicyContext with evaluation signals
            
        Returns:
            Tuple[bool, str]: (should_capture, reason)
        """
        # Check permanent suppression
        if context.lead_capture_permanently_suppressed:
            return False, "permanently_suppressed"
        
        # Check temporary suppression window
        if context.exchange_count < context.lead_capture_suppressed_until:
            return False, "in_suppression_window"
        
        # Check if max asks exceeded
        if context.lead_capture_asked_count >= self.MAX_LEAD_ASKS:
            return False, "max_asks_exceeded"
        
        # Only proceed with signal-based triggers
        intent = self.detect_intent(context.user_input)
        
        # Trigger 1: Explicit contact request
        if intent == UserIntent.REQUEST_CONTACT:
            return True, "explicit_contact_intent"
        
        # Trigger 2: CTA selection (handled externally by orchestrator)
        # This method just validates the context
        
        # Trigger 3: Implicit commercial intent + engagement
        if (
            context.engagement_score >= self.ENGAGEMENT_THRESHOLD and
            self.detect_resistance(context.user_input, context.state_history) < self.RESISTANCE_THRESHOLD
        ):
            # Additional check: is there a natural conversation pause?
            # (detected by orchestrator based on response type)
            return True, "implicit_commercial_intent"
        
        return False, "no_trigger"
    
    def _can_capture_lead(self, context: PolicyContext) -> bool:
        """Check if lead capture is allowed based on suppression state."""
        if context.lead_capture_permanently_suppressed:
            return False
        if context.exchange_count < context.lead_capture_suppressed_until:
            return False
        if context.lead_capture_asked_count >= self.MAX_LEAD_ASKS:
            return False
        return True
    
    def mark_lead_capture_suppression(
        self, 
        context: PolicyContext,
        permanent: bool = False
    ) -> PolicyContext:
        """
        Apply lead capture suppression.
        
        Called when:
        - User refuses lead capture once (temporary)
        - User refuses twice (permanent)
        - User says "just browsing" (permanent)
        - User says "don't ask again" (permanent)
        
        Args:
            context: Current policy context
            permanent: True for permanent suppression
            
        Returns:
            Updated PolicyContext
        """
        if permanent:
            context.lead_capture_permanently_suppressed = True
            logger.info(f"[PolicyEngine] Lead capture PERMANENTLY suppressed")
        else:
            context.lead_capture_suppressed_until = (
                context.exchange_count + self.SUPPRESSION_WINDOW
            )
            logger.info(
                f"[PolicyEngine] Lead capture suppressed until exchange "
                f"{context.lead_capture_suppressed_until}"
            )
        
        return context


# Convenience function for creating policy context from slots
def create_policy_context(
    current_state: str,
    user_input: str,
    slots: dict,
    session_context: dict = None
) -> PolicyContext:
    """
    Create PolicyContext from orchestrator data.
    
    Args:
        current_state: Current UC1 state name
        user_input: User's message
        slots: UC1Slots as dictionary
        session_context: Optional session-level data
        
    Returns:
        PolicyContext ready for evaluation
    """
    session_context = session_context or {}
    
    return PolicyContext(
        current_state=current_state,
        user_input=user_input,
        engagement_score=slots.get("engagement_score", 0.0),
        resistance_score=0.0,  # Calculated during evaluation
        exchange_count=session_context.get("exchange_count", 0),
        is_in_uc1=session_context.get("is_in_uc1", False),
        is_in_free_exploration=session_context.get("is_in_free_exploration", False),
        lead_capture_asked_count=slots.get("lead_capture_asked_count", 0),
        lead_capture_suppressed_until=slots.get("lead_capture_suppressed_until", 0),
        lead_capture_permanently_suppressed=slots.get("lead_capture_permanently_suppressed", False),
        uc1_paused=session_context.get("uc1_paused", False),
        resume_uc1=False,
        state_history=session_context.get("state_history", []),
        recent_inputs=session_context.get("recent_inputs", []),
    )
