import logging
from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from typing import List
from .models import (
    UserCreate,
    UserRegisterResponse,
    SentMessage,
    HistoryResponse,
    Prompt,
    PromptType,
)
from .helpers import (
    get_messages_for_session,
)
from . import helpers
from app.db.base import get_db_conn
from app.api.deps import get_follow_up_manager

# Get logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["v1"])

# Common headers for SSE
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
}

# User Management Routes
@router.post("/user/register", response_model=UserRegisterResponse)
async def register_user(user: UserCreate):
    """Create a new session with browser and IP information."""
    conn = get_db_conn()
    cursor = conn.cursor()

    try:
        # Create new user entry - let Postgres generate the UUID
        cursor.execute(
            """
            INSERT INTO users (browser, ip)
            VALUES (%s, %s)
            RETURNING id::text
        """,
            (user.browser, user.ip),
        )
        user_row = cursor.fetchone()
        if not user_row or not user_row[0]:
            raise HTTPException(status_code=500, detail="Failed to create user")
        user_id = user_row[0]  # Get the UUID as string

        # Create new session - let Postgres handle both UUIDs
        cursor.execute(
            """
            INSERT INTO sessions (user_id, browser, ip, is_active)
            VALUES (%s, %s, %s, TRUE)
            RETURNING session_id::text
        """,
            (user_id, user.browser, user.ip),
        )

        session_row = cursor.fetchone()
        if not session_row or not session_row[0]:
            raise HTTPException(status_code=500, detail="Failed to create session")
        session_id = session_row[0]  # Get the UUID as string
        conn.commit()

        return UserRegisterResponse(
            status="success",
            message="Session created successfully",
            session_id=session_id,
        )
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error in register_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create session")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# Message History Route

@router.get("/chat/{session_id}/messages", response_model=HistoryResponse)
async def get_chat_messages(session_id: str):
    """Get all messages for a chat session."""
    try:
        session_uuid = UUID(session_id)
        messages = await get_messages_for_session(session_uuid)
        if not messages:
            raise HTTPException(
                status_code=404, detail="No messages found for this session"
            )

        formatted_messages = []
        for message in messages:
            ts = message.created_at
            if hasattr(ts, "isoformat"):
                timestamp = ts.isoformat()
            else:
                timestamp = str(ts) if ts else None
            formatted_messages.append(
                {
                    "role": message.role,
                    "message": message.content,
                    "timestamp": timestamp,
                }
            )
        return HistoryResponse(session_id=session_id, messages=formatted_messages)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chat messages")


# Health Check Route
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "streaming": "enabled", "chatbot": "optimized"}


# Prompt Routes
@router.get("/prompts/root", response_model=List[Prompt])
async def get_root_prompts():
    """Fetch top-level prompts."""
    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        # Fetch prompts with all required fields
        cursor.execute(
            """
            SELECT 
                id::text,
                prompt_text,
                response_text,
                display_order,
                created_at,
                updated_at
            FROM prompts
            WHERE parent_id IS NULL
            ORDER BY display_order ASC
        """
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Convert to Prompt models with all required fields
        prompts = [
            Prompt(
                id=row[0],
                prompt_text=row[1],
                response_text=row[2] if row[2] is not None else "",
                display_order=row[3],
                type=PromptType.ROOT,
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        ]

        return prompts

    except Exception as e:
        logger.error(f"Error fetching root prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching prompts: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@router.post("/chat/send-stream")
async def post_send_message_stream(
    req: SentMessage, follow_up_manager=Depends(get_follow_up_manager)
):
    """Delegate to helper implementation for streaming chat."""
    return await helpers.send_message_stream(req, follow_up_manager)



