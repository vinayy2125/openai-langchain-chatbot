from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import List
from app.core.services.thread_router import ChatRouter

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
from app.db import redis_operations as redis_crud

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
        user_id = cursor.fetchone()[0]  # Get the UUID as string

        # Create new session - let Postgres handle both UUIDs
        cursor.execute(
            """
            INSERT INTO sessions (user_id, browser, ip, is_active)
            VALUES (%s, %s, %s, TRUE)
            RETURNING session_id::text
        """,
            (user_id, user.browser, user.ip),
        )

        session_id = cursor.fetchone()[0]  # Get the UUID as string
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
        messages = await get_messages_for_session(session_id)
        if not messages:
            raise HTTPException(
                status_code=404, detail="No messages found for this session"
            )

        formatted_messages = []
        for role, msg, ts in messages:
            formatted_messages.append(
                {
                    "role": role,
                    "message": msg,
                    "timestamp": ts.isoformat() if ts else None,
                }
            )
        return HistoryResponse(session_id=session_id, messages=formatted_messages)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving chat messages")


# Health Check Route
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "streaming": "enabled", "chatbot": "optimized"}


# Chat Routes


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


@router.post("/init-embeddings")
async def init_embeddings():
    """Initialize embedding model and Redis index, return connection/model status."""
    try:
        r = redis_crud.get_redis_client()
    except Exception as e:
        logger.error("Redis connection failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Redis connection failed: {e}")

    # Ensure index exists (will create if missing)
    try:
        redis_crud.ensure_index_exists(r)
    except Exception as e:
        logger.error("Failed to ensure index: %s", e)
        raise HTTPException(status_code=500, detail=f"Index creation failed: {e}")

    # Try loading embedding model
    try:
        model = redis_crud.get_embedding_model()
    except Exception as e:
        logger.error("Embedding model load failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding model load failed: {e}")

    return {"status": "ok", "redis_index": redis_crud.INDEX_NAME, "embed_dim": redis_crud.EMBED_DIM}


# ---------------- Redis document endpoints -----------------
@router.get("/redis/docs")
async def list_redis_docs(limit: int = 50):
    """List JSON documents stored under the configured prefix."""
    try:
        r = redis_crud.get_redis_client()
        # Scan keys using the prefix
        keys = r.scan_iter(match=f"{redis_crud.PREFIX}*")
        docs = []
        count = 0
        for k in keys:
            if count >= limit:
                break
            doc = r.json().get(k)
            docs.append({"key": k, "doc": doc})
            count += 1
        return {"count": len(docs), "docs": docs}
    except Exception as e:
        logger.error("Error listing redis docs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/redis/docs/{doc_id}")
async def get_redis_doc(doc_id: str):
    """Fetch a single JSON document by id (without prefix)."""
    try:
        r = redis_crud.get_redis_client()
        key = f"{redis_crud.PREFIX}{doc_id}"
        doc = r.json().get(key)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"key": key, "doc": doc}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching redis doc %s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redis/docs")
async def create_redis_doc(payload: dict):
    """Create a new message document and generate/store its embedding.

    Payload example: {"message": "hello world", "metadata": {"user_id": "123"}}
    """
    try:
        message = payload.get("message")
        metadata = payload.get("metadata", {})
        if not message:
            raise HTTPException(status_code=400, detail="'message' is required")

        r = redis_crud.get_redis_client()
        # Ensure index exists
        redis_crud.ensure_index_exists(r)

        msg_id = redis_crud.generate_and_store_embedding(r, message, metadata)
        return {"status": "created", "id": msg_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating redis doc: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
