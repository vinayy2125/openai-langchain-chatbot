# Prompt Authority Router - HARD ROUTING ENFORCEMENT
#
# =============================================================================
# NON-NEGOTIABLE LAW:
#   Exactly ONE prompt authority may execute per user message.
#   No fall-through. No blending. No shared execution.
# =============================================================================
#
# Priority Order (ABSOLUTE):
#   1. UC1_CANONICAL - If UC1 is active
#   2. EXPLORER_AGENT - If FREE_EXPLORATION state
#   3. STANDARD_DYNAMIC - Default fallback
#
# This router is PURE STATE. No intent inspection. No keyword checks.

from enum import Enum
from typing import Optional
from app.logger import get_logger

logger = get_logger("prompt_router")


class PromptAuthority(Enum):
    """
    The three mutually exclusive prompt authorities.
    Only ONE may execute per user message.
    """
    UC1_CANONICAL = "uc1_canonical"
    EXPLORER_AGENT = "explorer_agent"
    STANDARD_DYNAMIC = "standard_dynamic"


def is_uc1_active(session_id: str) -> bool:
    """
    Check if UC1 mode is active for this session.
    
    Sources (in priority order):
    1. In-memory slot cache
    2. Persisted Redis state
    
    Returns True if session has active UC1 state.
    """
    from app.orchestrator.slot_manager import SlotManager
    
    # Check in-memory cache first (fastest)
    if session_id in SlotManager._session_slots:
        slots = SlotManager._session_slots[session_id]
        # UC1 is active if we have capability bucket selected OR if not frozen
        if not slots.frozen:
            return True
    
    # Check Redis for persisted state
    try:
        from app.utils.conversation_memory import get_session_metadata
        persisted_slots = get_session_metadata(session_id, "uc1_slots", default=None)
        if persisted_slots and isinstance(persisted_slots, dict):
            # Has persisted UC1 state - reload into memory
            from app.orchestrator.slot_manager import UC1Slots
            SlotManager._session_slots[session_id] = UC1Slots.from_dict(persisted_slots)
            return not persisted_slots.get("frozen", False)
    except Exception as e:
        logger.debug(f"[Router] Failed to check Redis for UC1 state: {e}")
    
    return False


def is_free_exploration(session_id: str) -> bool:
    """
    Check if session is in FREE_EXPLORATION state.
    
    This is a UC1 sub-state where the user has broken out of the funnel.
    """
    from app.orchestrator.slot_manager import SlotManager
    from app.orchestrator.state_machine import UC1State
    
    # Must have UC1 slots to be in free exploration
    if session_id not in SlotManager._session_slots:
        return False
    
    slots = SlotManager._session_slots[session_id]
    
    # Check if UC1 is paused (user broke out of funnel)
    if slots.uc1_paused and slots.paused_state:
        return True
    
    return False


def route_to_authority(session_id: str, query: str) -> PromptAuthority:
    """
    HARD ROUTER - Determines exactly ONE prompt authority.
    
    PRIORITY ORDER (ABSOLUTE - DO NOT CHANGE):
        1. UC1_CANONICAL - If UC1 is active, NOTHING else speaks
        2. EXPLORER_AGENT - Only if UC1 is NOT active AND free exploration
        3. STANDARD_DYNAMIC - Default fallback
    
    CRITICAL INVARIANT:
        UC1 decides when exploration is allowed.
        Explorer NEVER preempts UC1.
        Explorer is a recovery hatch, not a co-equal authority.
    
    RULES:
    - No intent inspection here
    - No keyword checks
    - Pure state-based routing
    
    Args:
        session_id: The session identifier
        query: User's input (only used for logging, not routing decisions)
    
    Returns:
        PromptAuthority: The single authority that will handle this message
    """
    # ==========================================================================
    # PRIORITY 1: UC1 CANONICAL (HIGHEST - If active, NOTHING else speaks)
    # ==========================================================================
    if is_uc1_active(session_id):
        logger.info(f"[Router] Authority: UC1_CANONICAL (UC1 active - supreme)")
        return PromptAuthority.UC1_CANONICAL
    
    # ==========================================================================
    # PRIORITY 2: EXPLORER AGENT (Only when UC1 is NOT active)
    # ==========================================================================
    if is_free_exploration(session_id):
        logger.info(f"[Router] Authority: EXPLORER_AGENT (UC1 inactive, free exploration)")
        return PromptAuthority.EXPLORER_AGENT
    
    # ==========================================================================
    # PRIORITY 3: STANDARD DYNAMIC (Default fallback)
    # ==========================================================================
    logger.info(f"[Router] Authority: STANDARD_DYNAMIC (no UC1 state)")
    return PromptAuthority.STANDARD_DYNAMIC


def get_authority_name(authority: PromptAuthority) -> str:
    """Human-readable name for logging."""
    names = {
        PromptAuthority.UC1_CANONICAL: "UC1 Canonical System Prompt",
        PromptAuthority.EXPLORER_AGENT: "Explorer Agent Prompt",
        PromptAuthority.STANDARD_DYNAMIC: "Standard Dynamic Prompt",
    }
    return names.get(authority, authority.value)
