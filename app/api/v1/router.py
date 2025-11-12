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
        await helpers.update_user_by_session(session_id, user)
        # Delete the last user message from the DB for this session
        deleted_id = helpers.delete_last_user_message(session_id)
        logger.info(f"Deleted last user message id: {deleted_id} for session_id: {session_id}")
        return {"status": "success", "message": "User updated successfully", "session_id": session_id, "deleted_message_id": deleted_id}
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in update_user route: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating user")


# End session API
@router.post("/session/end/{session_id}")
async def end_session(session_id: str):
    """End a session and send closure email if conditions are met."""
    try:
        result = await helpers.end_session_helper(session_id)
        return result
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error in end_session route: {str(e)}")
        raise HTTPException(status_code=500, detail="Error ending session")


# Refresh Api Route - Get all messages for a chat session
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

        # Also fetch session is_active flag
        is_active = False
        try:
            conn2 = get_db_conn()
            cur2 = conn2.cursor()
            cur2.execute(
                """
                SELECT is_active FROM sessions WHERE session_id = %s
                """,
                (str(session_uuid),),
            )
            row = cur2.fetchone()
            if row and row[0] is not None:
                is_active = bool(row[0])
        except Exception as e:
            logger.warning(f"Could not fetch is_active for session {session_id}: {e}")
        finally:
            try:
                cur2.close()
            except Exception:
                pass
            try:
                conn2.close()
            except Exception:
                pass

        return {
            "root_prompts": root_prompts,
            "session_id": session_id,
            "is_active": is_active,
            "messages": formatted_messages,
        }
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chat messages")

