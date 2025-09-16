"""Shared models for chatbot components."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum

class FollowUpType(str, Enum):
    YES_NO = "yes_no"
    NESTED = "nested"
    EXPANSION = "expansion"
    CLARIFICATION = "clarification"

class FollowUp(BaseModel):
    type: FollowUpType
    question: str
    context: str
    validation_rules: Optional[Dict[str, Any]] = None
    context_key: Optional[str] = None
    options: Optional[List[str]] = None
    required: bool = True

    class Config:
        arbitrary_types_allowed = True

class SessionState(BaseModel):
    prompt_id: Optional[str] = None
    prompt_text: Optional[str] = None
    current_context: str = ""
    current_follow_up: Optional[FollowUp] = None
    gathered_requirements: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []
    follow_up_count: int = 0
    max_follow_ups: int = 5
    requirements_met: bool = False
    previous_response: Optional[str] = None
    last_follow_up_context: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
