# UC1 Slot Manager - Conversation State Persistence
#
# ARCHITECTURE RULE: The SlotManager owns ALL slot-related logic.
# - Engagement scoring is computed HERE, not by LLM
# - Slot validation is done HERE
# - Persistence is handled HERE
#
# ENGAGEMENT SCORE OWNERSHIP (ONLY orchestrator updates via SlotManager):
# +0.2 for valid button click
# +0.3 for providing free text when asked
# +0.1 for completion without retries
# -0.1 per retry on same state

from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Any, Literal, List
from enum import Enum
from app.logger import get_logger
from app.utils.conversation_memory import (
    get_session_memory_manager,
    set_session_metadata,
    get_session_metadata,
    delete_session_metadata,
)

logger = get_logger("slot_manager")


# =============================================================================
# SLOT WRITE PERMISSION ENFORCEMENT
# =============================================================================
# ARCHITECTURE LAW: Only authorized components may write UC1 slots.
# prompts.py MUST NEVER mutate UC1 slots.
# Any call without explicit caller should FAIL.
# =============================================================================

ALLOWED_SLOT_WRITERS = frozenset({
    "orchestrator",      # Primary authority
    "explorer_agent",    # Validated writes only
})

DEFAULT_CALLER = "unknown"


def _verify_caller(caller: str, operation: str) -> None:
    """
    Verify caller is authorized to write slots.
    
    Raises:
        PermissionError: If caller is not authorized
    """
    if caller not in ALLOWED_SLOT_WRITERS:
        raise PermissionError(
            f"[SlotManager] BLOCKED: Caller '{caller}' not authorized for '{operation}'. "
            f"Allowed: {ALLOWED_SLOT_WRITERS}"
        )


# Type for capability bucket IDs
CapabilityBucketId = Literal["UC1-A", "UC1-B", "UC1-C", "UC1-D", "UC1-E", "UC1-F"]


class EngagementEvent(Enum):
    """
    Events that affect engagement score.
    
    OWNERSHIP: Only the orchestrator can emit these events.
    LLM never influences engagement scoring.
    """
    BUTTON_CLICK = "button_click"  # +0.2
    TEXT_PROVIDED = "text_provided"  # +0.3
    COMPLETION_NO_RETRY = "completion_no_retry"  # +0.1
    RETRY = "retry"  # -0.1


# Score deltas for each event type
ENGAGEMENT_DELTAS: Dict[EngagementEvent, float] = {
    EngagementEvent.BUTTON_CLICK: 0.2,
    EngagementEvent.TEXT_PROVIDED: 0.3,
    EngagementEvent.COMPLETION_NO_RETRY: 0.1,
    EngagementEvent.RETRY: -0.1,
}


@dataclass
class UC1Slots:
    """
    Conversation slots for UC1 flow.
    
    NOTE: capability_bucket IS the sub-use case.
    There is no separate sub_use_case slot (removed per architectural review).
    This prevents dual sources of truth.
    """
    # Core slots
    capability_bucket: Optional[CapabilityBucketId] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None  # NEW (2026-01-15): Email for lead capture
    context_signal: Optional[str] = None  # User's answer to context question
    
    # Exploration layer tracking (for S5 state)
    exploration_turn: int = 0  # Current turn in exploration (1, 2, or 3)
    exploration_responses: Optional[List[str]] = None  # User's exploration answers
    last_user_message: Optional[str] = None  # For LLM context
    
    # Selected alternative (informational, not blocking)
    selected_alternative: Optional[str] = None
    
    # Selected CTA outcome
    selected_cta_outcome: Optional[str] = None
    
    # Lead capture tracking (policy engine-owned)
    lead_capture_asked_count: int = 0
    lead_capture_suppressed_until: int = 0  # Exchange count when suppression expires
    lead_capture_permanently_suppressed: bool = False
    email_capture_asked_at: int = 0  # NEW: Turn when email was last asked
    
    # UC1 pause/resume tracking (for FREE_EXPLORATION)
    uc1_paused: bool = False
    paused_state: Optional[str] = None  # State before entering FREE_EXPLORATION
    exchange_count: int = 0  # Total exchanges in session
    
    # Engagement scoring (orchestrator-owned)
    engagement_score: float = 0.0
    retry_count: int = 0
    
    # ================================================================
    # AUTHORITATIVE FLAGS (Per UC1 Robustness Fixes - 2026-01-12)
    # ================================================================
    exploration_complete: bool = False      # IRREVERSIBLE - exploration layer cannot reopen
    alternatives_consumed: bool = False     # Prevents re-showing alternative buttons
    frozen: bool = False                    # Hard freeze after bailout - no mutations
    free_exploration_unclear_count: int = 0 # Stabilizer trigger in FREE_EXPLORATION
    name_declined: bool = False             # User declined to give name (offer again at CTA)
    
    # Shared content tracking (prevents repetitive responses)
    shared_urls: Optional[List[str]] = None    # URLs already shared in this session
    
    # ACC Phase 3: Question budget per state
    question_counts: Optional[Dict[str, int]] = None  # state -> question count
    
    def get_filled_slots(self) -> Set[str]:
        """Return set of slot names that are filled (non-None, non-empty)."""
        filled = set()
        if self.capability_bucket:
            filled.add("capability_bucket")
        if self.user_name and self.user_name.strip():
            filled.add("user_name")
        if self.user_email and self.user_email.strip():
            filled.add("user_email")
        if self.context_signal and self.context_signal.strip():
            filled.add("context_signal")
        return filled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "capability_bucket": self.capability_bucket,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "context_signal": self.context_signal,
            "exploration_turn": self.exploration_turn,
            "exploration_responses": self.exploration_responses,
            "last_user_message": self.last_user_message,
            "selected_alternative": self.selected_alternative,
            "selected_cta_outcome": self.selected_cta_outcome,
            # Lead capture tracking
            "lead_capture_asked_count": self.lead_capture_asked_count,
            "lead_capture_suppressed_until": self.lead_capture_suppressed_until,
            "lead_capture_permanently_suppressed": self.lead_capture_permanently_suppressed,
            "email_capture_asked_at": self.email_capture_asked_at,
            # UC1 pause/resume
            "uc1_paused": self.uc1_paused,
            "paused_state": self.paused_state,
            "exchange_count": self.exchange_count,
            # Engagement
            "engagement_score": self.engagement_score,
            "retry_count": self.retry_count,
            # Authoritative flags
            "exploration_complete": self.exploration_complete,
            "alternatives_consumed": self.alternatives_consumed,
            "frozen": self.frozen,
            "free_exploration_unclear_count": self.free_exploration_unclear_count,
            "name_declined": self.name_declined,
            # ACC Phase 3
            "question_counts": self.question_counts or {},
            # Shared content tracking
            "shared_urls": self.shared_urls or [],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UC1Slots":
        """Reconstruct from dictionary (for persistence recovery)."""
        return cls(
            capability_bucket=data.get("capability_bucket"),
            user_name=data.get("user_name"),
            user_email=data.get("user_email"),
            context_signal=data.get("context_signal"),
            exploration_turn=data.get("exploration_turn", 0),
            exploration_responses=data.get("exploration_responses"),
            last_user_message=data.get("last_user_message"),
            selected_alternative=data.get("selected_alternative"),
            selected_cta_outcome=data.get("selected_cta_outcome"),
            # Lead capture tracking
            lead_capture_asked_count=data.get("lead_capture_asked_count", 0),
            lead_capture_suppressed_until=data.get("lead_capture_suppressed_until", 0),
            lead_capture_permanently_suppressed=data.get("lead_capture_permanently_suppressed", False),
            email_capture_asked_at=data.get("email_capture_asked_at", 0),
            # UC1 pause/resume
            uc1_paused=data.get("uc1_paused", False),
            paused_state=data.get("paused_state"),
            exchange_count=data.get("exchange_count", 0),
            # Engagement
            engagement_score=data.get("engagement_score", 0.0),
            retry_count=data.get("retry_count", 0),
            # Authoritative flags
            exploration_complete=data.get("exploration_complete", False),
            alternatives_consumed=data.get("alternatives_consumed", False),
            frozen=data.get("frozen", False),
            free_exploration_unclear_count=data.get("free_exploration_unclear_count", 0),
            name_declined=data.get("name_declined", False),
            # ACC Phase 3
            question_counts=data.get("question_counts"),
            # Shared content tracking
            shared_urls=data.get("shared_urls"),
        )


class SlotManager:
    """
    Manages UC1 conversation slots with persistence.
    
    OWNERSHIP:
    - Engagement scoring is computed and updated HERE
    - Slot validation is done HERE
    - Persistence is handled HERE (in-memory for now, can be extended to Redis/DB)
    
    LLM NEVER influences slot values or engagement scores.
    """
    
    # In-memory storage for slots (keyed by session_id)
    # In production, this should be backed by Redis or database
    _session_slots: Dict[str, UC1Slots] = {}
    
    def __init__(self, session_id: str):
        """
        Initialize SlotManager for a session.
        
        Args:
            session_id: The session identifier
        """
        self.session_id = session_id
        
        # Get or create slots for this session
        if session_id not in self._session_slots:
            # Try to load persisted slots from conversation memory first
            loaded = self._load_from_memory(session_id)
            if not loaded:
                self._session_slots[session_id] = UC1Slots()
                logger.info(f"[SlotManager] Created new slots for session: {session_id}")
    
    @property
    def slots(self) -> UC1Slots:
        """Get the slots for this session."""
        return self._session_slots[self.session_id]
    
    def set_capability_bucket(self, bucket_id: CapabilityBucketId, caller: str = DEFAULT_CALLER) -> None:
        """Set the selected capability bucket. Caller must be authorized."""
        _verify_caller(caller, "set_capability_bucket")
        self.slots.capability_bucket = bucket_id
        logger.info(f"[SlotManager] Set capability_bucket: {bucket_id} (caller: {caller})")
        self._safe_persist()
    
    def set_user_name(self, name: str, caller: str = DEFAULT_CALLER) -> None:
        """Set the user's name. Caller must be authorized."""
        _verify_caller(caller, "set_user_name")
        self.slots.user_name = name.strip() if name else None
        logger.info(f"[SlotManager] Set user_name: {self.slots.user_name} (caller: {caller})")
        self._safe_persist()
    
    def set_user_email(self, email: str, caller: str = DEFAULT_CALLER) -> bool:
        """
        Set the user's email with basic validation. Caller must be authorized.
        
        Args:
            email: Email address to set
            caller: Authorized caller identifier
            
        Returns:
            bool: True if email was valid and set, False otherwise
        """
        _verify_caller(caller, "set_user_email")
        import re
        if not email:
            return False
        
        email = email.strip().lower()
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            logger.warning(f"[SlotManager] Invalid email format: {email}")
            return False
        
        self.slots.user_email = email
        logger.info(f"[SlotManager] Set user_email: {email} (caller: {caller})")
        self._safe_persist()
        return True
    
    def mark_email_asked(self) -> None:
        """Mark that email was asked at this turn."""
        self.slots.email_capture_asked_at = self.slots.exchange_count
        self._safe_persist()
    
    def should_ask_for_email(self, min_turns: int = 4, max_turns: int = 6) -> bool:
        """
        Determine if email should be requested based on progressive capture rules.
        
        Email is asked:
        1. After min_turns exchanges AND we have name
        2. OR when user shows high intent (ready_for_cta)
        3. NOT if we already have email
        4. NOT if we already asked within last 3 turns
        
        Returns:
            bool: True if should request email
        """
        # Already have email
        if self.slots.user_email:
            return False
        
        # Need name first
        if not self.slots.user_name:
            return False
        
        # Don't ask too frequently
        if self.slots.email_capture_asked_at > 0:
            turns_since_ask = self.slots.exchange_count - self.slots.email_capture_asked_at
            if turns_since_ask < 3:
                return False
        
        # Check turn threshold
        return self.slots.exchange_count >= min_turns
    
    def set_context_signal(self, signal: str, caller: str = DEFAULT_CALLER) -> None:
        """Set the user's context answer. Caller must be authorized."""
        _verify_caller(caller, "set_context_signal")
        self.slots.context_signal = signal.strip() if signal else None
        logger.info(f"[SlotManager] Set context_signal: {self.slots.context_signal[:50] if self.slots.context_signal else None}... (caller: {caller})")
        self._safe_persist()
    
    def set_selected_alternative(self, alternative: str, caller: str = DEFAULT_CALLER) -> None:
        """Set the user's selected alternative. Caller must be authorized."""
        _verify_caller(caller, "set_selected_alternative")
        self.slots.selected_alternative = alternative
        logger.info(f"[SlotManager] Set selected_alternative: {alternative} (caller: {caller})")
        self._safe_persist()
    
    def set_selected_cta_outcome(self, outcome: str, caller: str = DEFAULT_CALLER) -> None:
        """Set the user's selected CTA outcome. Caller must be authorized."""
        _verify_caller(caller, "set_selected_cta_outcome")
        self.slots.selected_cta_outcome = outcome
        logger.info(f"[SlotManager] Set selected_cta_outcome: {outcome} (caller: {caller})")
        self._safe_persist()
    
    def set_exploration_turn(self, turn: int, caller: str = DEFAULT_CALLER) -> None:
        """Set the current exploration turn (1, 2, or 3). Caller must be authorized."""
        _verify_caller(caller, "set_exploration_turn")
        self.slots.exploration_turn = turn
        if self.slots.exploration_responses is None:
            self.slots.exploration_responses = []
        logger.info(f"[SlotManager] Set exploration_turn: {turn} (caller: {caller})")
        self._safe_persist()
    
    def add_exploration_response(self, response: str) -> None:
        """Add a user's exploration response and increment turn."""
        if self.slots.exploration_responses is None:
            self.slots.exploration_responses = []
        self.slots.exploration_responses.append(response)
        self.slots.last_user_message = response
        logger.info(f"[SlotManager] Added exploration response #{len(self.slots.exploration_responses)}")
        self._safe_persist()
    
    def set_last_user_message(self, message: str, caller: str = DEFAULT_CALLER) -> None:
        """Set the last user message for LLM context. Caller must be authorized."""
        _verify_caller(caller, "set_last_user_message")
        self.slots.last_user_message = message
        self._safe_persist()
    
    def _safe_persist(self) -> None:
        """Persist with proper error logging (not silent)."""
        try:
            self.persist()
        except Exception as e:
            logger.warning(f"[SlotManager] Persist failed (continuing): {e}")
    
    def increment_engagement(self, event: EngagementEvent) -> None:
        """
        Update engagement score based on event.
        
        OWNERSHIP: This is the ONLY place engagement scoring happens.
        LLM never influences this.
        """
        delta = ENGAGEMENT_DELTAS.get(event, 0.0)
        old_score = self.slots.engagement_score
        self.slots.engagement_score = max(0.0, self.slots.engagement_score + delta)
        
        if event == EngagementEvent.RETRY:
            self.slots.retry_count += 1
        
        logger.info(
            f"[SlotManager] Engagement event: {event.value}, "
            f"score: {old_score:.2f} → {self.slots.engagement_score:.2f}, "
            f"retries: {self.slots.retry_count}"
        )
        self._safe_persist()
    
    def get_filled_slots(self) -> Set[str]:
        """Get set of filled slot names."""
        return self.slots.get_filled_slots()
    
    def validate_required_slots(self, required_slots: tuple) -> list:
        """
        Validate that required slots are filled.
        
        Args:
            required_slots: Tuple of required slot names
        
        Returns:
            List of missing slot names (empty if all filled)
        """
        filled = self.get_filled_slots()
        missing = [slot for slot in required_slots if slot not in filled]
        if missing:
            logger.warning(f"[SlotManager] Missing required slots: {missing}")
        return missing
    
    def persist(self) -> None:
        """
        Persist slots to storage.
        
        Currently in-memory. Override for Redis/DB persistence.
        """
        try:
            # Store into conversation memory metadata for this session
            set_session_metadata(self.session_id, "uc1_slots", self.slots.to_dict())
            logger.debug(f"[SlotManager] Persisted slots to conversation memory for session: {self.session_id}")
        except Exception as e:
            logger.error(f"[SlotManager] Failed to persist slots: {e}")
    
    def load(self) -> bool:
        """
        Load slots from storage.
        
        Returns:
            bool: True if slots were loaded, False if new session
        """
        # Check in-memory first
        if self.session_id in self._session_slots:
            logger.debug(f"[SlotManager] Load slots for session: {self.session_id}, exists=in-memory")
            return True

        # Attempt to load from conversation memory metadata
        try:
            data = get_session_metadata(self.session_id, "uc1_slots", default=None)
            if data and isinstance(data, dict):
                self._session_slots[self.session_id] = UC1Slots.from_dict(data)
                logger.info(f"[SlotManager] Loaded slots from conversation memory for session: {self.session_id}")
                return True
        except Exception as e:
            logger.error(f"[SlotManager] Failed to load slots from memory: {e}")

        logger.debug(f"[SlotManager] No persisted slots for session: {self.session_id}")
        return False

    @classmethod
    def _load_from_memory(cls, session_id: str) -> bool:
        """Attempt to populate the in-memory cache from conversation memory metadata."""
        try:
            data = get_session_metadata(session_id, "uc1_slots", default=None)
            if data and isinstance(data, dict):
                cls._session_slots[session_id] = UC1Slots.from_dict(data)
                logger.info(f"[SlotManager] Class-loaded slots from conversation memory for session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"[SlotManager] _load_from_memory failed: {e}")
        return False
    
    def clear(self) -> None:
        """Clear slots for this session."""
        if self.session_id in self._session_slots:
            del self._session_slots[self.session_id]
            logger.info(f"[SlotManager] Cleared slots for session: {self.session_id}")
        try:
            delete_session_metadata(self.session_id, "uc1_slots")
        except Exception as e:
            logger.error(f"[SlotManager] Failed to delete session metadata in clear(): {e}")
    
    @classmethod
    def clear_session(cls, session_id: str) -> None:
        """Class method to clear a specific session's slots."""
        if session_id in cls._session_slots:
            del cls._session_slots[session_id]
            logger.info(f"[SlotManager] Cleared slots for session: {session_id}")
        try:
            delete_session_metadata(session_id, "uc1_slots")
        except Exception as e:
            logger.error(f"[SlotManager] Failed to delete session metadata in clear_session(): {e}")
    
    # ============================================================
    # Lead Capture Tracking (Policy Engine Integration)
    # ============================================================
    
    def increment_lead_capture_asked(self) -> int:
        """
        Increment lead capture ask count.
        
        Called when lead capture is attempted.
        After MAX_ASKS (2), permanent suppression should be applied.
        
        Returns:
            int: New ask count
        """
        self.slots.lead_capture_asked_count += 1
        logger.info(f"[SlotManager] Lead capture asked: {self.slots.lead_capture_asked_count}")
        self._safe_persist()
        return self.slots.lead_capture_asked_count
    
    def set_lead_capture_suppression(
        self, 
        until_exchange: int = 0,
        permanent: bool = False
    ) -> None:
        """
        Set lead capture suppression.
        
        Args:
            until_exchange: Exchange count when suppression expires (0 = no temp suppression)
            permanent: If True, never ask again in this session
        """
        if permanent:
            self.slots.lead_capture_permanently_suppressed = True
            logger.info("[SlotManager] Lead capture PERMANENTLY suppressed")
        else:
            self.slots.lead_capture_suppressed_until = until_exchange
            logger.info(f"[SlotManager] Lead capture suppressed until exchange {until_exchange}")
        self._safe_persist()
    
    def is_lead_capture_suppressed(self) -> bool:
        """
        Check if lead capture is currently suppressed.
        
        Returns:
            bool: True if suppressed (temporary or permanent)
        """
        if self.slots.lead_capture_permanently_suppressed:
            return True
        if self.slots.exchange_count < self.slots.lead_capture_suppressed_until:
            return True
        return False
    
    # ============================================================
    # UC1 Pause/Resume (FREE_EXPLORATION Integration)
    # ============================================================
    
    def pause_uc1(self, current_state: str) -> None:
        """
        Pause UC1 flow for FREE_EXPLORATION.
        
        Args:
            current_state: The UC1 state before entering FREE_EXPLORATION
        """
        self.slots.uc1_paused = True
        self.slots.paused_state = current_state
        logger.info(f"[SlotManager] UC1 PAUSED at state: {current_state}")
        self._safe_persist()
    
    def resume_uc1(self) -> str:
        """
        Resume UC1 flow from FREE_EXPLORATION.
        
        Returns:
            str: The state to resume from
        """
        paused_state = self.slots.paused_state
        self.slots.uc1_paused = False
        self.slots.paused_state = None
        logger.info(f"[SlotManager] UC1 RESUMED from state: {paused_state}")
        self._safe_persist()
        return paused_state or "capability_selection"
    
    def increment_exchange(self) -> int:
        """
        Increment exchange count (each user message).
        
        Returns:
            int: New exchange count
        """
        self.slots.exchange_count += 1
        self._safe_persist()
        return self.slots.exchange_count
    
    # ============================================================
    # ROBUSTNESS METHODS (Per UC1 Fixes - 2026-01-12)
    # ============================================================
    
    def build_anchor_summary(self) -> str:
        """
        Build single-line grounding context for LLM injection.
        
        ⚠️ WARNING: FOR LLM CONTEXT ONLY - NEVER SHOW TO USERS!
        This contains internal IDs like "UC1-A" which are not user-friendly.
        For user-facing messages, use bucket.trigger from the config.
        
        This is injected via user-message prefix, NOT system prompt.
        Preserves the single-prompt invariant.
        
        Returns:
            str: Anchor context string for LLM (empty if no context available)
        """
        parts = []
        if self.slots.capability_bucket:
            parts.append(f"Focus: {self.slots.capability_bucket}")
        if self.slots.context_signal:
            # Truncate to 50 chars
            signal = self.slots.context_signal[:50]
            if len(self.slots.context_signal) > 50:
                signal += "..."
            parts.append(f"Goal: {signal}")
        if self.slots.user_name:
            parts.append(f"Name: {self.slots.user_name}")
        return " | ".join(parts) if parts else ""
    
    def mark_exploration_complete(self) -> None:
        """
        Mark exploration as complete.
        
        AUTHORITATIVE: Once set, exploration layer CANNOT be reopened.
        This prevents the "exploration restart after completion" bug.
        """
        if self.slots.frozen:
            logger.warning("[SlotManager] Cannot mark exploration complete - slots frozen")
            return
        self.slots.exploration_complete = True
        logger.info("[SlotManager] Exploration marked COMPLETE (irreversible)")
        self._safe_persist()
    
    def mark_alternatives_consumed(self) -> None:
        """
        Mark alternatives as consumed.
        
        AUTHORITATIVE: Prevents re-showing alternative buttons after selection.
        """
        if self.slots.frozen:
            logger.warning("[SlotManager] Cannot mark alternatives consumed - slots frozen")
            return
        self.slots.alternatives_consumed = True
        logger.info("[SlotManager] Alternatives marked CONSUMED")
        self._safe_persist()
    
    def is_exit_ready(self) -> bool:
        """
        Check if conversation is ready for exit/CTA phase.
        
        When True, ButtonManager should surface CTAs directly instead of
        exploration buttons. This collapses the state machine:
          Exploration → CTA (not Exploration → READY_FOR_CTA → CTA)
        
        Returns:
            bool: True if should show CTAs directly
        """
        # Already selected an alternative - show CTAs
        if self.slots.selected_alternative:
            return True
        # Alternatives already consumed - show CTAs
        if self.slots.alternatives_consumed:
            return True
        # After 2+ exploration turns - conversation is mature enough for CTAs
        if self.slots.exploration_turn >= 2:
            return True
        # Exploration explicitly completed
        if self.slots.exploration_complete:
            return True
        return False
    
    def freeze_slots(self) -> None:
        """
        Hard freeze slots after bailout.
        
        No further mutations allowed. User must restart to clear.
        """
        self.slots.frozen = True
        logger.info("[SlotManager] Slots FROZEN - no further mutations allowed")
        self._safe_persist()
    
    def reset_retry_count(self) -> None:
        """Reset retry count (e.g., after successful progression)."""
        if self.slots.frozen:
            return
        self.slots.retry_count = 0
        self._safe_persist()
    
    def mark_name_declined(self) -> None:
        """Mark that user declined to give name (will re-offer at CTA)."""
        if self.slots.frozen:
            return
        self.slots.name_declined = True
        logger.info("[SlotManager] Name declined by user (will re-offer at CTA)")
        self._safe_persist()
    
    def increment_retry(self) -> int:
        """
        Increment retry count for validation failures.
        
        Used when user provides invalid input (gibberish, numbers-only, etc.)
        and we need to ask again with a fixed prompt.
        
        Returns:
            int: New retry count
        """
        if self.slots.frozen:
            return self.slots.retry_count
        self.slots.retry_count += 1
        logger.info(f"[SlotManager] Retry count: {self.slots.retry_count}")
        self._safe_persist()
        return self.slots.retry_count
    
    def increment_free_exploration_unclear(self) -> int:
        """
        Increment unclear count in FREE_EXPLORATION mode.
        
        Returns:
            int: New unclear count (stabilizer triggers at 2+)
        """
        if self.slots.frozen:
            return self.slots.free_exploration_unclear_count
        self.slots.free_exploration_unclear_count += 1
        logger.info(f"[SlotManager] FREE_EXPLORATION unclear count: {self.slots.free_exploration_unclear_count}")
        self._safe_persist()
        return self.slots.free_exploration_unclear_count
    
    def reset_free_exploration_unclear(self) -> None:
        """Reset unclear count on valid input in FREE_EXPLORATION."""
        if self.slots.frozen:
            return
        if self.slots.free_exploration_unclear_count > 0:
            self.slots.free_exploration_unclear_count = 0
            self._safe_persist()
    
    # ============================================================
    # ACC PHASE 3: QUESTION BUDGET (2026-01-12)
    # ============================================================
    
    def increment_question_count(self, state: str) -> int:
        """
        Increment question count for a state.
        
        ACC INVARIANT: One qualifying question per state. Period.
        
        Returns:
            int: New question count for this state
        """
        if self.slots.frozen:
            return self.get_question_count(state)
        if self.slots.question_counts is None:
            self.slots.question_counts = {}
        self.slots.question_counts[state] = self.slots.question_counts.get(state, 0) + 1
        logger.info(f"[ACC] Question count for {state}: {self.slots.question_counts[state]}")
        self._safe_persist()
        return self.slots.question_counts[state]
    
    def get_question_count(self, state: str) -> int:
        """Get current question count for a state."""
        if self.slots.question_counts is None:
            return 0
        return self.slots.question_counts.get(state, 0)
    
    def question_budget_exceeded(self, state: str, limit: int = 1) -> bool:
        """
        Check if question budget exceeded for a state.
        
        ACC INVARIANT: Once budget exceeded, force synthesis.
        
        Args:
            state: The state to check
            limit: Maximum questions allowed (default 1)
            
        Returns:
            bool: True if budget exceeded
        """
        return self.get_question_count(state) >= limit
    
    # ============================================================
    # SHARED CONTENT TRACKING (Prevents repetitive responses)
    # ============================================================
    
    def add_shared_url(self, url: str) -> None:
        """
        Record a URL that was shared with the user.
        
        Args:
            url: The URL that was shared
        """
        if self.slots.frozen:
            return
        if self.slots.shared_urls is None:
            self.slots.shared_urls = []
        
        url_normalized = url.lower().strip()
        if url_normalized and url_normalized not in self.slots.shared_urls:
            self.slots.shared_urls.append(url_normalized)
            logger.debug(f"[SlotManager] Added shared URL: {url_normalized}")
            self._safe_persist()
    
    def get_shared_urls(self) -> List[str]:
        """Get list of already-shared URLs."""
        return self.slots.shared_urls or []
