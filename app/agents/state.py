"""
Agent State Definition

Defines the TypedDict for agent state that flows through the LangGraph.
Uses Annotated types for proper message accumulation.
"""

from typing import TypedDict, Annotated, Optional, Dict, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State object that flows through the ReAct agent graph.
    
    Attributes:
        messages: Accumulated conversation messages (auto-appended)
        slots: Current slot values from SlotManager
        session_id: Session identifier for persistence
        turn_count: Number of agent turns in current session
        lead_score: Calculated engagement score (0-100)
        is_ready: Whether user is ready for options
        context_summary: Compressed history summary for efficiency
        tool_just_used: Whether a tool was just executed (for loop control)
    """
    messages: Annotated[list, add_messages]
    slots: Dict[str, Any]
    session_id: str
    turn_count: int
    lead_score: int
    is_ready: bool
    context_summary: Optional[str]
    tool_just_used: bool
