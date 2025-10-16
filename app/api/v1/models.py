from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
)
class PromptType(str, Enum):
    ROOT = "root"
    FOLLOW_UP = "follow_up"

class PromptBase(BaseModel):
    prompt_text: str = Field(..., description="The text of the prompt")
    response_text: Optional[str] = Field(None, description="The default response text")
    display_order: int = Field(
        default=0, description="The order in which to display the prompt"
    )
    type: PromptType = Field(default=PromptType.ROOT, description="The type of prompt")

class Prompt(PromptBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    parent_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class UserCreate(BaseModel):
    """User registration data - only browser and ip are required."""

    username: Optional[str] = None
    email: Optional[str] = None
    mobile: Optional[str] = None
    browser: str
    ip: str

class MessageBase(BaseModel):
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    reply_to: Optional[str] = Field(
        None, description="ID of the message this is replying to"
    )
    follow_up_to: Optional[str] = Field(
        None, description="ID of the message this is following up"
    )
    follow_up_depth: int = Field(default=0, description="Depth of follow-up chain")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageCreate(MessageBase):
    session_id: str


class Message(MessageBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    created_at: datetime
    updated_at: datetime


class UserRegisterResponse(BaseModel):
    status: str
    message: str
    session_id: str

class SentMessage(BaseModel):
    """Message sent by the user to the chatbot."""

    query: Optional[str] = Field(
        None, description="The user's message or query", max_length=4000
    )
    session_id: str = Field(..., description="Unique session identifier")
    prompt_id: Optional[str] = None
    stream: bool = Field(default=True, description="Whether to stream the response")
    detailed: bool = Field(
        default=False, description="Whether to generate detailed responses"
    )

class HistoryResponse(BaseModel):
    session_id: str = Field(..., description="UUID of the chat session")
    messages: List[Dict[str, Any]] = Field(
        ..., description="List of messages in the conversation"
    )
