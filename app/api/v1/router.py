from app.logger import get_logger
from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from .models import (
    UserCreate,
    UserRegisterResponse,
    SentMessage,
    HistoryResponse
)
from .helpers import (
    get_messages_for_session
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

        # Get root prompts (greeting/options/hint)
        root_prompts = None
        formatted_messages = []
        try:
            root_prompts = await helpers.fetch_root_prompts()
            logger.info(f"Fetched root prompts: {root_prompts}")
        except Exception as e:
            logger.error(f"Error fetching root prompts for chat messages: {str(e)}")
            root_prompts = None

        if messages:
            for message in messages:
                ts = message.created_at
                if hasattr(ts, "isoformat"):
                    timestamp = ts.isoformat()
                else:
                    timestamp = str(ts) if ts else None
                # Return raw markdown/plain text as stored in DB
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

        return {
            "root_prompts": root_prompts,
            "session_id": session_id,
            "messages": formatted_messages
        }
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chat messages")


# Health Check Route
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "streaming": "enabled", "chatbot": "optimized"}


# Prompt Routes
@router.get("/prompts/root")
async def get_root_prompts():
    """Fetch top-level prompts."""
    return await helpers.fetch_root_prompts()


@router.post("/chat/send-stream")
async def post_send_message_stream(
    req: SentMessage, follow_up_manager: FollowUpManager = Depends(get_follow_up_manager)
):
    """Streaming chat endpoint with hybrid form-trigger behavior."""
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



