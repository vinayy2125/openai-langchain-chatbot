# Prompt Authority Router - HARD ROUTING ENFORCEMENT
#
# =============================================================================
# NON-NEGOTIABLE LAW:
#   Exactly ONE prompt authority may execute per user message.
#   No fall-through. No blending. No shared execution.
#
#   NEW ARCHITECTURE:
#   Returns Tuple[LLMAuthority, ContentMode]
#   Authority = Who controls flow (UC1 vs NON_UC1)
#   ContentMode = How model speaks (Strict vs Exploration vs Generic)
# =============================================================================

import os
import re
from typing import Tuple, Optional
from app.logger import get_logger
from app.orchestrator.llm_adapter import LLMAuthority, ContentMode

logger = get_logger("prompt_router")

# Kill Switch for Strict Fact Mode
# If False, downgrades STRICT -> EXPLORATION
WEBSITE_FACT_MODE_ENABLED = os.getenv("WEBSITE_FACT_MODE_ENABLED", "true").lower() == "true"

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
            UC1Slots.from_dict(persisted_slots) # Just validating, logic is in slot manager usually
            # We just return true if not frozen
            return not persisted_slots.get("frozen", False)
    except Exception as e:
        logger.debug(f"[Router] Failed to check Redis for UC1 state: {e}")
    
    return False


def is_free_exploration(session_id: str) -> bool:
    """
    Check if session is in FREE_EXPLORATION state.
    """
    from app.orchestrator.slot_manager import SlotManager
    
    # Must have UC1 slots to be in free exploration
    if session_id not in SlotManager._session_slots:
        return False
    
    slots = SlotManager._session_slots[session_id]
    
    # Check if UC1 is paused (user broke out of funnel)
    if slots.uc1_paused and slots.paused_state:
        return True
    
    return False

def is_strictly_factual_query(query: str) -> bool:
    """
    Determines if a query is a STRUCTURAL Factual/Capability lookup.
    
    Structure MUST be in: {LIST, WHO, WHAT, DO_YOU, HAVE_YOU}
    Intent MUST be FACTUAL.
    """
    if not query:
        return False
        
    q = query.lower().strip()
    
    # Structural Keywords
    STARTS_WITH_TRIGGERS = (
        "list",
        "who",
        "what",
        "do you",
        "have you",
        "can you",
        "does ditstek",
        "how many"
    )
    
    # Simple check: Does it start with one of these?
    has_structure = q.startswith(STARTS_WITH_TRIGGERS)
    
    if has_structure:
        # Additional heuristic: If it's too long/narrative, it might be exploration.
        # Strict mode is usually short questions.
        word_count = len(q.split())
        if word_count > 25:
             # Long questions with "what..." are often consultative/exploration
            return False
        return True
        
    return False


def route_to_authority(session_id: str, query: str) -> Tuple[LLMAuthority, ContentMode]:
    """
    HARD ROUTER - Determines exactly ONE (Authority, ContentMode) pair.
    
    PRIORITY ORDER (ABSOLUTE):
        1. UC1_CANONICAL - If UC1 is active (Supreme)
        2. WEBSITE_REPRESENTATIVE_STRICT - If Structural Factual Query
        3. WEBSITE_REPRESENTATIVE_EXPLORATION - If Free Exploration or Fallback
    """
    # ==========================================================================
    # PRIORITY 1: UC1 CANONICAL (HIGHEST - If active, NOTHING else speaks)
    # ==========================================================================
    if is_uc1_active(session_id):
        logger.info(f"[Router] {session_id} -> Authority: UC1_CANONICAL (Active Flow)")
        return LLMAuthority.UC1_CANONICAL, ContentMode.GENERIC
    
    # ==========================================================================
    # PRIORITY 2: STRICT WEBSITE (Structural Factual Queries)
    # ==========================================================================
    # Only applies if NOT in active UC1 flow (checked above)
    if is_strictly_factual_query(query):
        # KILL SWITCH CHECK
        if not WEBSITE_FACT_MODE_ENABLED:
            logger.warning(f"[Router] KILL SWITCH ACTIVE: Downgrading STRICT to EXPLORATION for query: '{query[:20]}...'")
            return LLMAuthority.NON_UC1, ContentMode.WEBSITE_REPRESENTATIVE_EXPLORATION
            
        logger.info(f"[Router] {session_id} -> Authority: NON_UC1 (Strict Factual)")
        return LLMAuthority.NON_UC1, ContentMode.WEBSITE_REPRESENTATIVE_STRICT
    
    # ==========================================================================
    # PRIORITY 3: EXPLORATION (Default for non-UC1)
    # ==========================================================================
    # If we are here, UC1 is NOT active, and it's NOT a strict factual query.
    # It routes to Exploration.
    # NOTE: Even without "Free Exploration" state explicitly set, if we are not in UC1,
    # we default to the Website Representative Explorer (instead of Standard Dynamic).
    # This ensures "One Truth" covering all interactions outside UC1.
    
    logger.info(f"[Router] {session_id} -> Authority: NON_UC1 (Exploration/Fallback)")
    return LLMAuthority.NON_UC1, ContentMode.WEBSITE_REPRESENTATIVE_EXPLORATION

def get_authority_name(authority: LLMAuthority) -> str:
    """Human-readable name for logging."""
    return authority.value

