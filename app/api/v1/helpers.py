import json
import os
from typing import Dict, Any, List
from fastapi.responses import StreamingResponse
from uuid import UUID
import psycopg2
from app.logger import get_logger
from fastapi import HTTPException, Depends
from app.api.deps import get_follow_up_manager
from app.core.services.thread_router import ChatRouter
from app.core.chat_logic import build_chatbot_response
from app.api.v1.models import (
    MessageCreate,
    Message,
    SentMessage,
)
from app.api.v1.models import UserCreate
from app.core.nested_follow_up_manager import FollowUpManager
from fastapi import HTTPException


logger = get_logger(__name__)
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
}


# Database connection helper
def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        options="-c client_encoding=UTF8",
    )


async def update_user_by_session(session_id: str, user: UserCreate):
    """Update a user's fields using the session_id.

    Accepts a `UserCreate` model and performs a partial update of only the
    provided fields (username, email, mobile, browser, ip). Also keeps the
    sessions table in sync for browser/ip when provided.

    Returns the session_id on success.
    """
    try:
        sid = UUID(str(session_id))
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid session_id format")

    conn = None
    cursor = None
    try:
        logger.debug(f"update_user_by_session called for session={session_id} with data={user.dict()}")
        conn = get_db_conn()
        cursor = conn.cursor()

        # Find the user for this session
        cursor.execute(
            """
            SELECT user_id::text FROM sessions WHERE session_id = %s
            """,
            (str(sid),),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            raise HTTPException(status_code=404, detail="Session not found")

        user_id = row[0]

        # Build a partial update for any provided user fields
        allowed_fields = ["username", "email", "mobile", "browser", "ip"]
        set_clauses = []
        params = []
        for field in allowed_fields:
            val = getattr(user, field, None)
            if val is not None:
                set_clauses.append(f"{field} = %s")
                params.append(val)

        if not set_clauses:
            raise HTTPException(status_code=400, detail="No user fields provided to update")

        # Append updated_at and WHERE param
        set_sql = ",\n                ".join(set_clauses)
        sql = f"""
            UPDATE users
            SET {set_sql},
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING id::text
        """
        params.append(user_id)

        cursor.execute(sql, tuple(params))
        updated = cursor.fetchone()
        if not updated or not updated[0]:
            conn.rollback()
            raise HTTPException(status_code=404, detail="User not found for session")

        # Keep sessions.browser/ip in sync if provided (do not force both)
        session_update_params = []
        session_set = []
        if getattr(user, "browser", None) is not None:
            session_set.append("browser = %s")
            session_update_params.append(user.browser)
        if getattr(user, "ip", None) is not None:
            session_set.append("ip = %s")
            session_update_params.append(user.ip)

        if session_set:
            session_sql = f"UPDATE sessions SET {', '.join(session_set)} WHERE session_id = %s"
            session_update_params.append(str(sid))
            cursor.execute(session_sql, tuple(session_update_params))

        conn.commit()
        return str(sid)
    except HTTPException:
        raise
    except psycopg2.IntegrityError as ie:
        # Constraint failure like unique email, etc.
        if conn:
            conn.rollback()
        logger.warning(f"Integrity error updating user for session {session_id}: {str(ie)}")
        raise HTTPException(status_code=400, detail=str(ie))
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error updating user by session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating user")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


async def initialize_session_with_prompt(
    session_id: UUID, prompt_id: UUID
) -> Dict[str, Any]:
    """Initialize a new session with the selected prompt."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        # Get prompt details
        cursor.execute(
            """
            SELECT prompt_text, response_text, type
            FROM prompts
            WHERE id = %s
        """,
            (prompt_id,),
        )
        prompt = cursor.fetchone()

        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")

        prompt_text, response_text, prompt_type = prompt

        # Update session with prompt
        cursor.execute(
            """
            UPDATE sessions
            SET current_prompt_id = %s,
                last_interaction_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
            RETURNING id
        """,
            (prompt_id, session_id),
        )

        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Session not found")

        # Initialize session state
        return {
            "prompt_id": str(prompt_id),
            "prompt_text": prompt_text,
            "response_text": response_text,
            "prompt_type": prompt_type,
        }

    finally:
        cursor.close()
        conn.close()

async def save_message(message_data: MessageCreate) -> Message:
    """Save a message to the database with proper relationships and metadata."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        # Update session's last interaction time
        cursor.execute(
            """
            UPDATE sessions
            SET last_interaction_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                follow_up_count = CASE 
                    WHEN %s IS NOT NULL THEN follow_up_count + 1 
                    ELSE follow_up_count 
                END
            WHERE session_id = %s
            RETURNING id
        """,
            (message_data.follow_up_to, message_data.session_id),
        )

        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Session not found")

        # Insert the message
        cursor.execute(
            """
            INSERT INTO messages (
                session_id, role, content, reply_to, follow_up_to, 
                follow_up_depth, metadata, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING *
        """,
            (
                message_data.session_id,
                message_data.role,
                message_data.content,
                message_data.reply_to,
                message_data.follow_up_to,
                message_data.follow_up_depth,
                json.dumps(message_data.metadata),
            ),
        )

        message_row = cursor.fetchone()
        conn.commit()

        return Message(
            id=message_row[0],
            session_id=message_row[1],
            content=message_row[2],
            role=message_row[3],
            reply_to=message_row[4],
            follow_up_to=message_row[5],
            follow_up_depth=message_row[6],
            metadata=json.loads(message_row[7]) if message_row[7] else {},
            created_at=message_row[8],
            updated_at=message_row[9],
        )
    finally:
        cursor.close()
        conn.close()

async def get_messages_for_session(session_id: UUID) -> List[Message]:
    """Retrieve all messages for a given session with their relationships and metadata."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT 
                m.id,
                m.session_id,
                m.content,
                m.role,
                m.reply_to,
                m.follow_up_to,
                m.follow_up_depth,
                m.metadata,
                m.created_at,
                m.updated_at
            FROM messages m
            WHERE m.session_id = %s
            ORDER BY m.created_at ASC
        """,
            (str(session_id),),
        )

        messages = []
        for row in cursor.fetchall():
            message = Message(
                id=row[0],
                session_id=row[1],
                content=row[2],
                role=row[3],
                reply_to=row[4],
                follow_up_to=row[5],
                follow_up_depth=row[6],
                metadata=json.loads(row[7]) if row[7] else {},
                created_at=row[8],
                updated_at=row[9],
            )
            messages.append(message)

        return messages
    finally:
        cursor.close()
        conn.close()

async def send_message_stream(
    req: SentMessage, follow_up_manager: FollowUpManager = Depends(get_follow_up_manager)
):

    try:
        session_id = (req.session_id)    
        if not session_id:
            raise HTTPException(status_code=422, detail="Invalid session_id provided")

        # Lazy init ChatRouter
        global chat_router
        if "chat_router" not in globals() or chat_router is None:
            # Initialize ChatRouter without similarity_fn
            chat_router = ChatRouter(follow_up_manager)

        session_data = follow_up_manager.get_session_data(session_id)
        is_new = not session_data or not session_data.get("prompt_context")

        # --- First-time session setup ---
        if is_new:
            prompt_context = None
            prompt_id_str = None
            if req.prompt_id:
                try:
                    # Validate UUID format; ignore invalid prompt IDs gracefully
                    _ = UUID(str(req.prompt_id))
                    prompt_db = await initialize_session_with_prompt(
                        session_id, UUID(str(req.prompt_id))
                    )
                    prompt_context = prompt_db["prompt_text"]
                    prompt_id_str = str(req.prompt_id)
                except (ValueError, HTTPException):
                    # Invalid UUID or prompt lookup failure → fall back to free-text query
                    prompt_context = (req.query or "").strip()
            else:
                prompt_context = (req.query or "").strip() or "General assistance"

            follow_up_manager.initialize_session(
                session_id=session_id,
                prompt_id=prompt_id_str,
                prompt_context=prompt_context,
            )

            if prompt_id_str and hasattr(
                follow_up_manager.chatbot, "reset_follow_up_count"
            ):
                follow_up_manager.chatbot.reset_follow_up_count(session_id)

            session_data = follow_up_manager.get_session_data(session_id)

            if req.query:
                await save_message(
                    MessageCreate(
                        content=req.query.strip(),
                        role="user",
                        session_id=session_id,
                        reply_to=None,
                        follow_up_to=None,
                    )
                )
                follow_up_manager.add_to_conversation_history(
                    session_id, "user", req.query.strip()
                )
        else:
            if req.prompt_id and str(req.prompt_id) != str(
                session_data.get("prompt_id")
            ):
                logger.warning(
                    f"Ignoring new prompt_id {req.prompt_id} for existing session {session_id}; "
                    f"continuing with original {session_data.get('prompt_id')}"
                )
            if req.query:
                await save_message(
                    MessageCreate(
                        content=req.query.strip(),
                        role="user",
                        session_id=session_id,
                        reply_to=None,
                        follow_up_to=None,
                    )
                )
                follow_up_manager.add_to_conversation_history(
                    session_id, "user", req.query.strip()
                )

        conversation_history = follow_up_manager.get_conversation_history(session_id)
        prompt_context = session_data.get("prompt_context")



        # --- Decide if more follow-ups are needed ---
        if not follow_up_manager.check_requirements(session_id):

            async def stream_follow_up():
                

                
                # Add the `data:` prefix to the JSON-encoded string
                yield "data: " + json.dumps(
                    {"status": "processing", "message": "Preparing response..."}
                ) + "\n\n"

                # Reuse streaming follow-up generator with context switch & suggestions
                sent_chunks = set()
                async for evt in build_chatbot_response(
                    session_id=session_id,
                    follow_up_manager=follow_up_manager,
                    conversation_history=conversation_history,
                    prompt_context=prompt_context,
                    mode="follow_up",
                ):
                    # Normalize to a textual chunk when possible
                    if isinstance(evt, dict):
                        status = evt.get("status", "unknown")
                        chunk = evt.get("chunk", "")
                    else:
                        status = "complete_chunk"
                        chunk = str(evt)

                    # Normalize and skip empty chunks; dedupe any repeats in this response
                    chunk_norm = "" if chunk is None else str(chunk).strip()
                    if not chunk_norm:
                        continue
                    if chunk_norm in sent_chunks:
                        continue
                    sent_chunks.add(chunk_norm)

                    # Persist embeddings for completed chunks if applicable
                    if status == "complete_chunk":
                        logger.info(f"========== chunk ========== {chunk_norm}")

                    yield "data: " + json.dumps({"status": status, "chunk": chunk}) + "\n\n"

            return StreamingResponse(
                stream_follow_up(), media_type="text/event-stream", headers=SSE_HEADERS
            )
        
        # --- Requirements captured: full response streaming ---
        async def generate_full_response_stream():
            

            yield "data: " + json.dumps(
                {
                    "status": "processing",
                    "message": "Generating comprehensive response...",
                }
            ) + "\n\n"

            sent_chunks = set()
            async for evt in build_chatbot_response(
                session_id=session_id,
                follow_up_manager=follow_up_manager,
                conversation_history=conversation_history,
                prompt_context=prompt_context,
                mode="complete",
            ):  
                # Normalize to a textual chunk when possible
                if isinstance(evt, dict):
                    status = evt.get("status", "unknown")
                    chunk = evt.get("chunk", "")
                else:
                    status = "complete_chunk"
                    chunk = str(evt)

                chunk_norm = "" if chunk is None else str(chunk).strip()
                if not chunk_norm:
                    continue
                if chunk_norm in sent_chunks:
                    logger.debug(f"[SSE] Skipping duplicate chunk (normalized)='{chunk_norm[:80]}'")
                    continue
                sent_chunks.add(chunk_norm)

                logger.debug(f"[SSE] Sending chunk status={status} preview='{chunk_norm[:80]}'")
                yield "data: " + json.dumps({"status": status, "chunk": chunk}) + "\n\n"
        
        
        return StreamingResponse(
            generate_full_response_stream(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error in send_message_stream: {str(e)}")

        async def generate_error_stream(e=e):
            yield f"data: {json.dumps({'status': 'processing', 'message': 'An error occurred'})}\n\n"
            error_response = {
                "status": "error",
                "message": "An unexpected error occurred",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(
            generate_error_stream(), media_type="text/event-stream", headers=SSE_HEADERS
        )
