"""
Multi-Turn Agent Package

This package implements a ReAct-based conversational agent for dynamic
exploration and lead capture, replacing the rigid state machine for S5.
"""

from app.agents.explorer_agent import ExplorerAgent
from app.agents.state import AgentState

__all__ = ["ExplorerAgent", "AgentState"]
