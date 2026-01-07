from app.logger import get_logger
from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from .models import (
    UserCreate,
    UserRegisterResponse,
    SentMessage,
    HistoryResponse,
    PromptRequest
)
from .helpers import (
    get_messages_for_session
)
from . import helpers
from app.db.base import get_db_conn, return_db_conn
# Get centralized logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/api")

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
            return_db_conn(conn)


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
async def post_send_message_stream(req: SentMessage):
    """Streaming chat endpoint with optimized chatbot only."""
    return await helpers.send_message_stream(req)



# Update user details (PATCH) - now keyed by session_id per register flow
@router.patch("/user/{session_id}")
async def update_user(session_id: str, user: UserCreate):
    """Update user's browser and IP information by session_id using helper."""
    try:
        # Pass the full Pydantic model to the helper for partial updates
        await helpers.update_user_by_session(session_id, user)
        # Delete the last user message from the DB for this session
        deleted_id = await helpers.delete_last_user_message(session_id)
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
        is_active = await helpers.get_session_is_active(session_uuid)

        # Try to include UC1 slot state (if present in conversation memory or in-memory)
        uc1_slots = None
        try:
            # Import locally to avoid top-level circular import
            from app.orchestrator.slot_manager import SlotManager

            if session_id in SlotManager._session_slots:
                uc1_slots = SlotManager(session_id).slots.to_dict()
            else:
                # Attempt to load from conversation memory metadata
                try:
                    from app.utils.conversation_memory import get_session_metadata
                    data = get_session_metadata(session_id, "uc1_slots", default=None)
                    if data and isinstance(data, dict):
                        uc1_slots = data
                except Exception:
                    uc1_slots = None
        except Exception:
            uc1_slots = None

        return {
            "root_prompts": root_prompts,
            "session_id": session_id,
            "is_active": is_active,
            "messages": formatted_messages,
            "uc1_slots": uc1_slots,
        }
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chat messages")




