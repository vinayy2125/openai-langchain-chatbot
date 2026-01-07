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
from typing import Optional, Set, Dict, Any, Literal
from enum import Enum
from app.logger import get_logger
from app.utils.conversation_memory import (
    get_session_memory_manager,
    set_session_metadata,
    get_session_metadata,
    delete_session_metadata,
)

logger = get_logger("slot_manager")


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
    context_signal: Optional[str] = None  # User's answer to context question
    
    # Selected alternative (informational, not blocking)
    selected_alternative: Optional[str] = None
    
    # Selected CTA outcome
    selected_cta_outcome: Optional[str] = None
    
    # Engagement scoring (orchestrator-owned)
    engagement_score: float = 0.0
    retry_count: int = 0
    
    def get_filled_slots(self) -> Set[str]:
        """Return set of slot names that are filled (non-None, non-empty)."""
        filled = set()
        if self.capability_bucket:
            filled.add("capability_bucket")
        if self.user_name and self.user_name.strip():
            filled.add("user_name")
        if self.context_signal and self.context_signal.strip():
            filled.add("context_signal")
        return filled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "capability_bucket": self.capability_bucket,
            "user_name": self.user_name,
            "context_signal": self.context_signal,
            "selected_alternative": self.selected_alternative,
            "selected_cta_outcome": self.selected_cta_outcome,
            "engagement_score": self.engagement_score,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UC1Slots":
        """Reconstruct from dictionary (for persistence recovery)."""
        return cls(
            capability_bucket=data.get("capability_bucket"),
            user_name=data.get("user_name"),
            context_signal=data.get("context_signal"),
            selected_alternative=data.get("selected_alternative"),
            selected_cta_outcome=data.get("selected_cta_outcome"),
            engagement_score=data.get("engagement_score", 0.0),
            retry_count=data.get("retry_count", 0),
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
    
    def set_capability_bucket(self, bucket_id: CapabilityBucketId) -> None:
        """Set the selected capability bucket."""
        self.slots.capability_bucket = bucket_id
        logger.info(f"[SlotManager] Set capability_bucket: {bucket_id}")
        try:
            self.persist()
        except Exception:
            pass
    
    def set_user_name(self, name: str) -> None:
        """Set the user's name."""
        self.slots.user_name = name.strip() if name else None
        logger.info(f"[SlotManager] Set user_name: {self.slots.user_name}")
        try:
            self.persist()
        except Exception:
            pass
    
    def set_context_signal(self, signal: str) -> None:
        """Set the user's context answer."""
        self.slots.context_signal = signal.strip() if signal else None
        logger.info(f"[SlotManager] Set context_signal: {self.slots.context_signal[:50] if self.slots.context_signal else None}...")
        try:
            self.persist()
        except Exception:
            pass
    
    def set_selected_alternative(self, alternative: str) -> None:
        """Set the user's selected alternative (informational)."""
        self.slots.selected_alternative = alternative
        logger.info(f"[SlotManager] Set selected_alternative: {alternative}")
        try:
            self.persist()
        except Exception:
            pass
    
    def set_selected_cta_outcome(self, outcome: str) -> None:
        """Set the user's selected CTA outcome."""
        self.slots.selected_cta_outcome = outcome
        logger.info(f"[SlotManager] Set selected_cta_outcome: {outcome}")
        try:
            self.persist()
        except Exception:
            pass
    
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
        try:
            self.persist()
        except Exception:
            pass
    
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
        except Exception:
            pass
    
    @classmethod
    def clear_session(cls, session_id: str) -> None:
        """Class method to clear a specific session's slots."""
        if session_id in cls._session_slots:
            del cls._session_slots[session_id]
            logger.info(f"[SlotManager] Cleared slots for session: {session_id}")
        try:
            delete_session_metadata(session_id, "uc1_slots")
        except Exception:
            pass
