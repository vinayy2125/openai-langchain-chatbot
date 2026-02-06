"""
Slot Writer Tool

Enables the agent to update conversation slots (context, name, etc.)
for lead capture and state tracking.
"""

from typing import Optional, Dict, Any
from langchain_core.tools import tool
from app.logger import get_logger

logger = get_logger("agent_tools")

# In-memory slot storage for agent sessions
# In production, this integrates with SlotManager
_agent_slots: Dict[str, Dict[str, Any]] = {}


@tool
def save_slot(session_id: str, slot_name: str, value: str) -> str:
    """
    Save a value to a conversation slot.
    
    Use this tool to capture user information like:
    - user_name: When user shares their name
    - context_signal: User's main challenge or need
    - email: User's email address
    
    Args:
        session_id: The current session ID
        slot_name: Name of the slot (user_name, context_signal, email)
        value: Value to save
        
    Returns:
        Confirmation message
    """
    # =========================================================================
    # MANDATORY VALIDATION - Explorer must validate before every slot write
    # =========================================================================
    from app.orchestrator.state_input_validators import validate_input_for_state, validate_name, validate_context_answer
    
    # Determine validation based on slot type
    is_valid = True
    failure_reason = ""
    
    if slot_name == "user_name":
        is_valid, failure_reason = validate_name(value)
    elif slot_name == "context_signal":
        is_valid, failure_reason = validate_context_answer(value)
    elif slot_name == "email":
        # Basic email validation
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = bool(re.match(email_pattern, value.strip().lower())) if value else False
        failure_reason = "invalid_email" if not is_valid else ""
    
    if not is_valid:
        logger.warning(f"[Slot Tool] BLOCKED: Invalid input for {slot_name}: {failure_reason}")
        return f"Could not save {slot_name}: {failure_reason}"
    
    # =========================================================================
    # VALIDATED - Now safe to write slots
    # =========================================================================
    if session_id not in _agent_slots:
        _agent_slots[session_id] = {}
    
    _agent_slots[session_id][slot_name] = value
    logger.info(f"[Slot Tool] Saved {slot_name}={value[:30]}... for session {session_id[:8]}")
    
    # Sync with orchestrator's SlotManager if available
    try:
        from app.orchestrator.slot_manager import SlotManager
        # Use the singleton pattern - SlotManager instances are cached by session
        sm = SlotManager(session_id)
        
        # ALL writes must include caller="explorer_agent"
        if slot_name == "user_name":
            sm.set_user_name(value, caller="explorer_agent")
        elif slot_name == "context_signal":
            sm.set_context_signal(value, caller="explorer_agent")
        elif slot_name == "capability_bucket":
            sm.set_capability_bucket(value, caller="explorer_agent")
            
    except Exception as e:
        logger.warning(f"[Slot Tool] Could not sync with SlotManager: {e}")
    
    return f"Saved {slot_name}: {value}"


@tool
def get_slots(session_id: str) -> Dict[str, Any]:
    """
    Get all current slot values for a session.
    
    Use this to check what information has already been collected.
    
    Args:
        session_id: The current session ID
        
    Returns:
        Dictionary of slot names to values
    """
    # Try SlotManager first
    try:
        from app.orchestrator.slot_manager import SlotManager
        sm = SlotManager.get_or_create(session_id)
        slots = sm.slots
        return {
            "user_name": slots.user_name,
            "context_signal": slots.context_signal,
            "capability_bucket": slots.capability_bucket,
            "email": getattr(slots, "email", None),
        }
    except Exception:
        pass
    
    # Fallback to agent memory
    return _agent_slots.get(session_id, {})


def calculate_lead_score(session_id: str, turn_count: int) -> int:
    """
    Calculate lead engagement score based on collected data.
    
    Scoring:
    - Name provided: +20
    - Email provided: +30
    - Context provided: +15
    - 3+ turns engaged: +15
    - Budget mentioned: +20
    
    Args:
        session_id: Session to score
        turn_count: Number of conversation turns
        
    Returns:
        Score from 0-100
    """
    slots = get_slots.invoke({"session_id": session_id})
    score = 0
    
    if slots.get("user_name"):
        score += 20
    if slots.get("email"):
        score += 30
    if slots.get("context_signal"):
        score += 15
        # Check for budget mention
        context = str(slots.get("context_signal", "")).lower()
        if any(word in context for word in ["budget", "cost", "price", "$", "spend"]):
            score += 20
    if turn_count >= 3:
        score += 15
    
    return min(score, 100)
