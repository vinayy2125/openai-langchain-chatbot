"""
Agent Supervisor

Implements guardrails and safety checks for the Explorer Agent:
- Maximum turn limits
- Required slot validation before exit
- Low-confidence routing
"""

from typing import Dict, Any
from app.agents.state import AgentState
from app.logger import get_logger

logger = get_logger("agent_supervisor")


class AgentSupervisor:
    """
    Supervises agent execution with safety guardrails.
    
    Ensures the agent:
    1. Doesn't loop indefinitely (max turns)
    2. Collects required information before exiting
    3. Routes low-confidence responses appropriately
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        required_slots: list = None,
        min_lead_score: int = 20
    ):
        """
        Initialize supervisor.
        
        Args:
            max_turns: Maximum agent turns before forced exit
            required_slots: Slots that must be filled before normal exit
            min_lead_score: Minimum score for quality lead
        """
        self.max_turns = max_turns
        self.required_slots = required_slots or ["context_signal"]
        self.min_lead_score = min_lead_score
    
    def check_should_continue(self, state: AgentState) -> str:
        """
        Determine if agent should continue, exit, or show options.
        
        Returns:
            'continue': Keep reasoning
            'ready': User is ready for options
            'exit': Force exit (max turns)
        """
        turn_count = state.get("turn_count", 0)
        is_ready = state.get("is_ready", False)
        
        # Hard limit
        if turn_count >= self.max_turns:
            logger.warning(f"[Supervisor] Max turns ({self.max_turns}) reached, forcing exit")
            return "exit"
        
        # User signaled readiness
        if is_ready:
            logger.info("[Supervisor] User ready for options")
            return "ready"
        
        return "continue"
    
    def check_exit_requirements(self, state: AgentState) -> Dict[str, Any]:
        """
        Check if required slots are filled before allowing exit.
        
        Returns:
            dict with 'can_exit' bool and 'missing' list of missing slots
        """
        slots = state.get("slots", {})
        missing = []
        
        for slot in self.required_slots:
            if not slots.get(slot):
                missing.append(slot)
        
        can_exit = len(missing) == 0
        
        if not can_exit:
            logger.info(f"[Supervisor] Exit blocked, missing slots: {missing}")
        
        return {
            "can_exit": can_exit,
            "missing": missing
        }
    
    def get_prompt_for_missing_slots(self, missing: list) -> str:
        """
        Generate a natural prompt to gather missing information.
        
        Args:
            missing: List of missing slot names
            
        Returns:
            Natural language prompt for the agent
        """
        prompts = {
            "user_name": "By the way, who am I speaking with today?",
            "context_signal": "Could you tell me more about your specific needs?",
            "email": "If you'd like, I can send you more details. What's a good email?"
        }
        
        for slot in missing:
            if slot in prompts:
                return prompts[slot]
        
        return "Is there anything specific you'd like me to address?"


# Default supervisor instance
default_supervisor = AgentSupervisor()
