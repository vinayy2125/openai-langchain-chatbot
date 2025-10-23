
import json
import os
from typing import Dict, Any, List, Optional
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
from datetime import datetime
from app.api.v1.models import PromptType
from app.db.base import get_db_conn


logger = get_logger(__name__)
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
}

# --- Form Trigger Logic (modular, reused by send_message_stream) ---
def is_prompt_trigger(response: str) -> bool:
    """Detect actionable cues in prompt response for form triggering."""
    action_cues = [
        "Would you like help scheduling a call?",
        "Can I have your best email",
        "Would you like to discuss a proposal",
        "Can we connect for a meeting",
        # Add more cues as needed
    ]
    return any(cue.lower() in response.lower() for cue in action_cues)
import re




def should_trigger_form(session_data: dict, user_message: str, prompt_response: Optional[str] = None) -> bool:
    """Primary: prompt-driven form trigger. Backup: intent detection.
    prompt_response: Optional[str] -- response from final_response_prompt or None
    """
    # Prompt-driven form trigger
    if prompt_response and is_prompt_trigger(prompt_response):
        return True
    # Conversation-length based trigger
    user_msgs = [m for m in session_data.get("conversation_history", []) if m.get("role") == "user"]
    if len(user_msgs) >= 10 and not session_data.get("form_shown", False):
        return True
    return False


def mark_form_shown(session_data: dict):
    session_data["form_shown"] = True
    return session_data


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
        if not message_row:
            raise HTTPException(status_code=500, detail="Failed to save message to database")
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
    """Retrieve all messages for a given session ordered by creation time.

    Returns a list of `Message` objects expected by the router.
    """
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

        messages: List[Message] = []
        rows = cursor.fetchall()
        for row in rows:
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
        session_id = req.session_id
        if not session_id:
            raise HTTPException(status_code=422, detail="Invalid session_id provided")

        # Lazy init ChatRouter
        global chat_router
        if "chat_router" not in globals():
            chat_router = None
        if chat_router is None:
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
                    _ = UUID(str(req.prompt_id))
                    # Ensure session_id is UUID for initialize_session_with_prompt
                    session_uuid = session_id if isinstance(session_id, UUID) else UUID(str(session_id))
                    prompt_db = await initialize_session_with_prompt(
                        UUID(str(session_id)), UUID(str(req.prompt_id))
                    )
                    prompt_context = prompt_db["prompt_text"]
                    prompt_id_str = str(req.prompt_id)
                except (ValueError, HTTPException):
                    prompt_context = (req.query or "").strip()
            else:
                prompt_context = (req.query or "").strip() or "General assistance"

            follow_up_manager.initialize_session(
                session_id=session_id,
                prompt_id=prompt_id_str,
                prompt_context=prompt_context,
            )
            if prompt_id_str and hasattr(follow_up_manager.chatbot, "reset_follow_up_count"):
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
            if req.prompt_id and str(req.prompt_id) != str(session_data.get("prompt_id")):
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
        user_message = req.query.strip() if req.query else ""

        # --- Hybrid Form Trigger ---
        # Get prompt response (simulate or fetch from prompt logic)
        prompt_response: Optional[str] = None
        if 'final_response' in session_data:
            prompt_response = session_data['final_response']
        form_triggered = should_trigger_form(session_data, user_message, prompt_response)
        if form_triggered and not session_data.get("form_shown", False):
            # Mark form as shown in session
            session_data = mark_form_shown(session_data)
            follow_up_manager.sessions[session_id] = session_data
            async def stream_form_trigger():
                yield "data: " + json.dumps({"status": "processing", "message": "Preparing response..."}) + "\n\n"
                # Single form trigger event for frontend to show the form
                yield "data: " + json.dumps({"status": "form_trigger", "chunk": "FORM_TRIGGER"}) + "\n\n"
            return StreamingResponse(stream_form_trigger(), media_type="text/event-stream", headers=SSE_HEADERS)

        # --- Decide if more follow-ups are needed ---
        if not follow_up_manager.check_requirements(session_id):
            async def stream_follow_up():
                # Inform client we're preparing a response
                yield "data: " + json.dumps(
                    {"status": "processing", "message": "Preparing response..."}
                ) + "\n\n"

                # Call the response generator and emit only the final complete_chunk(s)
                final_response = None
                async for evt in build_chatbot_response(
                    session_id=session_id,
                    follow_up_manager=follow_up_manager,
                    conversation_history=conversation_history,
                    prompt_context=prompt_context,
                    mode="follow_up",
                ):
                    if isinstance(evt, dict):
                        status = evt.get("status", "unknown")
                        chunk = evt.get("chunk", "")
                    else:
                        status = "complete_chunk"
                        chunk = str(evt)
                    # Only skip empty chunks if not form_trigger
                    if chunk is None or (str(chunk).strip() == "" and status != "form_trigger"):
                        continue
                    yield "data: " + json.dumps({"status": status, "chunk": chunk}) + "\n\n"
                    if status == "complete_chunk":
                        final_response = chunk
                        yield "data: " + json.dumps({"status": status, "chunk": chunk}) + "\n\n"
                        # Save assistant message after streaming
                        if final_response:
                            await save_message(
                                MessageCreate(
                                    content=final_response.strip(),
                                    role="assistant",
                                    session_id=str(session_id),
                                    reply_to=None,
                                    follow_up_to=None,
                                )
                            )
                        return
            return StreamingResponse(stream_follow_up(), media_type="text/event-stream", headers=SSE_HEADERS)

        # --- Requirements captured: full response streaming ---
        async def generate_full_response_stream():
            # Inform client generation is starting
            yield "data: " + json.dumps(
                {"status": "processing", "message": "Generating comprehensive response..."}
            ) + "\n\n"

            final_response = None
            async for evt in build_chatbot_response(
                session_id=session_id,
                follow_up_manager=follow_up_manager,
                conversation_history=conversation_history,
                prompt_context=prompt_context,
                mode="complete",
            ):
                if isinstance(evt, dict):
                    status = evt.get("status", "unknown")
                    chunk = evt.get("chunk", "")
                else:
                    status = "complete_chunk"
                    chunk = str(evt)
                # Only skip empty chunks if not form_trigger
                if chunk is None or (str(chunk).strip() == "" and status != "form_trigger"):
                    continue

                # Save the assistant message only for the final complete_chunk
                if status == "complete_chunk":
                    final_response = chunk
                # Forward chunks as-is (chat_logic currently yields final formatted response)
                yield "data: " + json.dumps({"status": status, "chunk": chunk}) + "\n\n"

            # After streaming, save the assistant message if present
            if final_response:
                # Save assistant message using same fields as user message
                await save_message(
                    MessageCreate(
                        content=final_response.strip(),
                        role="assistant",
                        session_id=str(session_id),
                        reply_to=None,
                        follow_up_to=None,
                    )
                )

        return StreamingResponse(generate_full_response_stream(), media_type="text/event-stream", headers=SSE_HEADERS)
    except Exception as e:
        logger.error(f"Unexpected error in send_message_stream: {str(e)}")
        raise

# Internal async function to fetch root prompts
async def fetch_root_prompts():
    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        greeting_text = "Hi I'm DITS AI 👋! I'd love to assist you and get to know you better.\n\n**What would you like to talk about?**"
        bottom_hint_text = "Please type if you are looking for Something else.."
        desired_order = [
            "See our Work",
            "Start a Project",
            "Talk to our team",
            "Explore DITS Services",
        ]
        all_prompt_texts = [greeting_text] + desired_order + [bottom_hint_text]

        cursor.execute(
            """
            SELECT prompt_text FROM prompts WHERE prompt_text = ANY(%s)
            """,
            (all_prompt_texts,)
        )
        existing = set(row[0] for row in cursor.fetchall())

        now = datetime.utcnow()
        for idx, text in enumerate(all_prompt_texts, start=1):
            if text not in existing:
                cursor.execute(
                    """
                    INSERT INTO prompts (prompt_text, response_text, display_order, type, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (text, "", idx, "ROOT", now, now)
                )
        conn.commit()

        cursor.execute(
            """
            SELECT id::text, prompt_text, response_text, display_order, type, created_at, updated_at
            FROM prompts
            WHERE prompt_text = ANY(%s)
            ORDER BY display_order ASC
            """,
            (all_prompt_texts,)
        )
        rows = cursor.fetchall()
        greeting = None
        desired_order_prompts = []
        hint = None
        for row in rows:
            pid, prompt_text, response_text, display_order, ptype, created_at, updated_at = row
            try:
                prompt_type_val = PromptType[ptype] if ptype in PromptType.__members__ else PromptType.ROOT
            except Exception:
                prompt_type_val = PromptType.ROOT
            prompt_obj = {
                "id": str(pid),
                "prompt_text": prompt_text,
                "response_text": response_text or "",
                "display_order": display_order,
                "type": str(prompt_type_val),
                "created_at": created_at,
                "updated_at": updated_at
            }
            if prompt_text == greeting_text:
                greeting = prompt_obj
            elif prompt_text == bottom_hint_text:
                hint = prompt_obj
            elif prompt_text in desired_order:
                desired_order_prompts.append(prompt_obj)

        # Sort desired_order_prompts by the order in desired_order
        desired_order_prompts_sorted = []
        for text in desired_order:
            for prompt in desired_order_prompts:
                if prompt["prompt_text"] == text:
                    desired_order_prompts_sorted.append(prompt)
                    break

        return {
            "greeting_text": greeting["prompt_text"] if greeting else None,
            "root_prompts": desired_order_prompts_sorted,
            "bottom_hint_text": hint["prompt_text"] if hint else None
        }
    except Exception as e:
        logger.error(f"Error fetching root prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching prompts: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
