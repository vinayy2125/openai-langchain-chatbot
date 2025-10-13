from datetime import datetime
from enum import Enum
from app.db.base import get_db_conn
from typing import List, Optional, Dict, Any
from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    model_validator,
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

    @model_validator(mode="after")
    def validate_prompt_or_query_for_new_session(self) -> "SentMessage":
        """Allow starting a session with either a prompt_id or an initial query.
        Only raise if neither is provided for a brand-new DB session."""
        

        conn = get_db_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id FROM sessions WHERE session_id = %s", (self.session_id,)
            )
            session_exists = cursor.fetchone() is not None
            if not session_exists and not (
                self.prompt_id or (self.query and self.query.strip())
            ):
                raise ValueError("prompt_id or initial query required to start session")
            return self
        finally:
            cursor.close()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How can I help you?",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "prompt_id": "some-prompt-id",  # Added this to example
                "stream": True,
                "detailed": False,
            }
        }
    )

class HistoryResponse(BaseModel):
    session_id: str = Field(..., description="UUID of the chat session")
    messages: List[Dict[str, Any]] = Field(
        ..., description="List of messages in the conversation"
    )
