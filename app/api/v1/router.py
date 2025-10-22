from app.logger import get_logger
from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID, uuid4
from typing import List
from datetime import datetime
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
from app.core.nested_follow_up_manager import FollowUpManager

# Get centralized logger
logger = get_logger(__name__)

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
            # Return empty list if no messages found (do not raise 404)
            return HistoryResponse(session_id=session_id, messages=[])

        formatted_messages = []
        for message in messages:
            ts = message.created_at
            if hasattr(ts, "isoformat"):
                timestamp = ts.isoformat()
            else:
                timestamp = str(ts) if ts else None
            formatted_messages.append(
                {
                    "id": message.id,
                    "role": message.role,
                    "message": message.content,
                    "timestamp": timestamp,
                    "reply_to": message.reply_to,
                    "follow_up_to": message.follow_up_to,
                    "metadata": message.metadata,
                }
            )
        return HistoryResponse(session_id=session_id, messages=formatted_messages)
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

        # Only return the specific set of prompts we want to show for now.
        desired_order = [
            "Start a Project",
            "Explore DITS Services",
            "See our Work",
            "Talk to our team",
        ]

        # Include the DirectMessageHint as well (we'll always ensure it's present below)
        allowed_texts = desired_order + ["DirectMessageHint"]

        cursor.execute(
            """
            SELECT id::text, prompt_text, response_text, display_order, type, created_at, updated_at
            FROM prompts
            WHERE parent_id IS NULL AND prompt_text = ANY(%s)
            """,
            (allowed_texts,)
        )
        rows = cursor.fetchall()

        # Map DB rows by prompt_text for easy ordering
        rows_by_text = {row[1]: row for row in rows}

        prompts = []
        for idx, text in enumerate(desired_order, start=1):
            row = rows_by_text.get(text)
            if row:
                pid, prompt_text, response_text, display_order, ptype, created_at, updated_at = row
                created_at = created_at or datetime.utcnow()
                updated_at = updated_at or datetime.utcnow()
                try:
                    prompt_type = PromptType(ptype) if ptype else PromptType.ROOT
                except Exception:
                    prompt_type = PromptType.ROOT

                prompts.append(
                    Prompt(
                        id=str(pid),
                        prompt_text=prompt_text,
                        response_text=response_text or "",
                        display_order=display_order or idx,
                        type=prompt_type,
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )
            else:
                # If a specific prompt is missing from DB, append a non-persisted placeholder
                now = datetime.utcnow()
                prompts.append(
                    Prompt(
                        id=str(uuid4()),
                        prompt_text=text,
                        response_text="",
                        display_order=idx,
                        type=PromptType.ROOT,
                        created_at=now,
                        updated_at=now,
                    )
                )

        # We intentionally return only the four action prompts for now.
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
    req: SentMessage, follow_up_manager: FollowUpManager = Depends(get_follow_up_manager)
):
    """Delegate to helper implementation for streaming chat."""
    return await helpers.send_message_stream(req, follow_up_manager)


# Update user details (PATCH) - now keyed by session_id per register flow
@router.patch("/user/{session_id}")
async def update_user(session_id: str, user: UserCreate):
    """Update user's browser and IP information by session_id using helper."""
    try:
        # Pass the full Pydantic model to the helper for partial updates
        updated_session = await helpers.update_user_by_session(session_id, user)
        return {"status": "success", "message": "User updated successfully", "session_id": updated_session}
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in update_user route: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating user")



