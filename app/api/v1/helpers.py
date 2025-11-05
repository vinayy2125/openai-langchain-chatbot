def delete_message_by_id(message_id: str):
    """Delete a message by its id from the database."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            DELETE FROM messages WHERE id = %s
            """,
            (str(message_id),)
        )
        conn.commit()
        logger.info(f"[delete_message_by_id] Deleted message id: {message_id}")
        return True
    except Exception as e:
        logger.error(f"[delete_message_by_id] Error deleting message id={message_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()
def delete_last_assistant_message(session_id: str):
    """Delete the last assistant message for a session from the database."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            DELETE FROM messages
            WHERE id = (
                SELECT id FROM messages
                WHERE session_id = %s AND role = 'assistant'
                ORDER BY created_at DESC
                LIMIT 1
            )
            RETURNING id
            """,
            (str(session_id),)
        )
        deleted = cursor.fetchone()
        conn.commit()
        logger.info(f"[delete_last_assistant_message] Deleted assistant message id: {deleted[0] if deleted else None} for session_id={session_id}")
        return deleted[0] if deleted else None
    except Exception as e:
        logger.error(f"[delete_last_assistant_message] Error deleting last assistant message for session_id={session_id}: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()
import re
import os
import json
import psycopg2
from typing import Dict, Any, List, Optional
from fastapi.responses import StreamingResponse
from uuid import UUID
from app.logger import get_logger
from fastapi import HTTPException, Depends
from app.api.deps import get_follow_up_manager
from app.core.chat_logic import build_chatbot_response
from app.api.v1.models import (
    MessageCreate,
    Message,
    SentMessage,
)
from app.api.v1.models import UserCreate
from app.core.nested_follow_up_manager import FollowUpManager
from app.core.services.email_sender import send_closure_email
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
    logger.info(f"[is_prompt_trigger] Checking response for action cues.")
    action_cues = [
        "Would you like help scheduling a call?",
        "Can I have your best email",
        "Would you like to discuss a proposal",
        "Can we connect for a meeting",
        # Add more cues as needed
    ]
    result = any(cue.lower() in response.lower() for cue in action_cues)
    logger.debug(f"[is_prompt_trigger] Result: {result}")
    return result




def should_trigger_form(session_data: dict, user_message: str, prompt_response: Optional[str] = None) -> bool:
    """
    Form trigger logic:
    - If user_details_known is True, never trigger the form.
    - If user_details_known is False, trigger form based on prompt cues or intent (conversation length).
    """
    # Always check user_details_known from DB for the session's user_id
    logger.info(f"[should_trigger_form] Called for session_id={session_data.get('session_id')}")

    session_id = session_data.get("session_id")
    user_details_known = get_user_details_known_from_db(session_id) if session_id else False
    logger.debug(f"[should_trigger_form] user_details_known={user_details_known}")
    if user_details_known:
        logger.info(f"[should_trigger_form] User details known, not triggering form.")
        return False
    # Only trigger form if user_details_known is False
    if prompt_response and is_prompt_trigger(prompt_response):
        logger.info(f"[should_trigger_form] Prompt response triggered form.")
        return True
    user_msgs = [m for m in session_data.get("conversation_history", []) if m.get("role") == "user"]
    logger.debug(f"[should_trigger_form] user_msgs count={len(user_msgs)} form_shown={session_data.get('form_shown', False)}")
    if len(user_msgs) >= 10 and not session_data.get("form_shown", False):
        logger.info(f"[should_trigger_form] Conversation length triggered form.")
        return True
    logger.info(f"[should_trigger_form] No form trigger conditions met.")
    return False

def get_user_details_known_from_db(session_id: str) -> bool:
    """Fetch user_details_known from the database for the user associated with the session."""
    logger.info(f"[get_user_details_known_from_db] Fetching user_details_known for session_id={session_id}")
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            options="-c client_encoding=UTF8",
        )
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.user_details_known FROM users u
            JOIN sessions s ON u.id = s.user_id
            WHERE s.session_id = %s
            """,
            (str(session_id),)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        logger.debug(f"[get_user_details_known_from_db] DB row: {row}")
        if row and row[0]:
            logger.info(f"[get_user_details_known_from_db] user_details_known=True for session_id={session_id}")
            return bool(row[0])
    except Exception as e:
        logger.warning(f"[get_user_details_known_from_db] DB error for session {session_id}: {e}")
    logger.info(f"[get_user_details_known_from_db] user_details_known=False for session_id={session_id}")
    return False

def mark_form_shown(session_data: dict):
    logger.info(f"[mark_form_shown] Marking form_shown=True for session_id={session_data.get('session_id')}")
    session_data["form_shown"] = True
    return session_data


def is_valid_ip(ip_str: str) -> bool:
    # Very small regex-based validation for IPv4 and IPv6-like patterns (best-effort)
    logger.info(f"[is_valid_ip] Validating IP: {ip_str}")
    if not ip_str:
        logger.debug("[is_valid_ip] IP string is empty.")
        return False
    ip_str = str(ip_str).strip()
    ipv4_re = r"^((25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(25[0-5]|2[0-4]\d|[01]?\d?\d)$"
    ipv6_re = r"^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$"
    try:
        if re.match(ipv4_re, ip_str):
            logger.debug("[is_valid_ip] Valid IPv4.")
            return True
        if re.match(ipv6_re, ip_str):
            logger.debug("[is_valid_ip] Valid IPv6.")
            return True
    except Exception as e:
        logger.error(f"[is_valid_ip] Exception: {e}")
        return False
    logger.debug("[is_valid_ip] IP is not valid.")
    return False


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
    provided fields (username, email, mobile). Also keeps the
    sessions table in sync for browser/ip when provided.

    Returns the session_id on success.
    """
    try:
        logger.info(f"[update_user_by_session] Called with session_id={session_id}, payload={user.dict()}")
        sid = UUID(str(session_id))
    except Exception as ex:
        logger.error(f"[update_user_by_session] Invalid session_id format: {session_id} | Exception: {ex}")
        raise HTTPException(status_code=422, detail="Invalid session_id format")

    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        logger.info(f"[update_user_by_session] DB connection established.")

        # Find the user for this session
        cursor.execute(
            """
            SELECT user_id::text FROM sessions WHERE session_id = %s
            """,
            (str(sid),),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            logger.error(f"[update_user_by_session] Session not found for session_id={session_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        user_id = row[0]
        logger.info(f"[update_user_by_session] Found user_id={user_id} for session_id={session_id}")

        # Update username, email, mobile, and user_details_known in users table
        allowed_fields = ["username", "email", "mobile", "user_details_known"]
        set_clauses = []
        params = []
        user.user_details_known = True
        for field in allowed_fields:
            val = getattr(user, field, None)
            logger.info(f"[update_user_by_session] Field {field} value: {val}")
            if val is not None:
                set_clauses.append(f"{field} = %s")
                params.append(val)

        if not set_clauses:
            logger.error(f"[update_user_by_session] No user fields provided to update for user_id={user_id}")
            raise HTTPException(status_code=400, detail="No user fields provided to update")

        set_sql = ",\n                ".join(set_clauses)
        sql = f"""
            UPDATE users
            SET {set_sql},
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING id::text
        """
        params.append(user_id)
        logger.info(f"[update_user_by_session] Executing SQL: {sql} | Params: {params}")

        cursor.execute(sql, tuple(params))
        updated = cursor.fetchone()
        if not updated or not updated[0]:
            logger.error(f"[update_user_by_session] User not found for user_id={user_id}, rolling back.")
            conn.rollback()
            raise HTTPException(status_code=404, detail="User not found for session")
        else:
            conn.commit()
            logger.info(f"[update_user_by_session] User updated successfully for user_id={user_id}")

        # Always set user_details_known = True in session state
        try:
            from app.api.deps import get_follow_up_manager
            follow_up_manager = get_follow_up_manager()
            session_data = follow_up_manager.get_session_data(str(sid))
            if session_data is not None:
                session_data.setdefault("state", {})["user_details_known"] = True
                try:
                    follow_up_manager.set_session_data(str(sid), session_data)
                    logger.info(f"[update_user_by_session] Session state updated: user_details_known=True for session_id={session_id}")
                except Exception as session_ex:
                    follow_up_manager.sessions[str(sid)] = session_data
                    logger.warning(f"[update_user_by_session] Fallback session state update for session_id={session_id}: {session_ex}")
        except Exception as e:
            logger.warning(f"[update_user_by_session] Could not set user_details_known in session: {e}")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"[update_user_by_session] Error updating user by session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating user")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logger.info(f"[update_user_by_session] DB connection closed.")


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
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
            RETURNING id
        """,
            (message_data.session_id,),
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
    logger.info(f"[send_message_stream] Called for session_id={getattr(req, 'session_id', None)}")

    # Extract required variables from req or context
    session_id = getattr(req, "session_id", None)
    conversation_history = getattr(req, "conversation_history", [])
    # Prefer an explicit query if provided by the client. This ensures the follow-up
    # flow has a usable latest_query for Redis context retrieval even when
    # conversation_history or prompt_context are empty.
    prompt_context = getattr(req, "query", None) or getattr(req, "prompt_context", None)

    # --- Unified streaming logic: yield processing event once, then stream follow-up or final response ---

    async def stream_response():
        yield "data: " + json.dumps({"status": "processing", "message": "Preparing response..."}) + "\n\n"
        if session_id is None or not isinstance(session_id, str):
            logger.error("[send_message_stream] session_id is missing or not a string.")
            raise HTTPException(status_code=400, detail="session_id is required and must be a string")

        # Save user message if present
        if req.query:
            user_msg = MessageCreate(
                session_id=session_id,
                content=req.query,
                role="user",
                reply_to=None,
                follow_up_to=None,
                follow_up_depth=0,
                metadata={}
            )
            try:
                await save_message(user_msg)
            except Exception as e:
                logger.warning(f"Failed to save user message: {e}")

        if not follow_up_manager.check_requirements(session_id):
            logger.info(f"[send_message_stream] Requirements not met, streaming follow up.")
            session_ended = False
            form_triggered = False
            # Buffer events to check for form_trigger
            buffered_events = []
            async for evt in build_chatbot_response(
                session_id=session_id,
                follow_up_manager=follow_up_manager,
                conversation_history=conversation_history,
                prompt_context=prompt_context,
                mode="follow_up",
            ):
                buffered_events.append(evt)
            # Iterate and yield, skipping last assistant chunk if next is form_trigger
            for i, evt in enumerate(buffered_events):
                assistant_content = None
                if isinstance(evt, dict):
                    status = evt.get("status", "unknown")
                    chunk = evt.get("chunk", "")
                    assistant_content = chunk
                    if status == "end_chat":
                        session_ended = True
                    if status == "form_trigger":
                        form_triggered = True
                else:
                    status = "unknown"
                    chunk = str(evt)
                    assistant_content = chunk
                # If next event is form_trigger, skip saving and yielding this assistant chunk
                if status == "chunk" and i + 1 < len(buffered_events):
                    next_evt = buffered_events[i + 1]
                    next_status = next_evt.get("status", "unknown") if isinstance(next_evt, dict) else "unknown"
                    if next_status == "form_trigger":
                        continue
                if assistant_content and status in ["chunk", "form_trigger"]:
                    assistant_msg = MessageCreate(
                        session_id=session_id,
                        content=assistant_content,
                        role="assistant",
                        reply_to=None,
                        follow_up_to=None,
                        follow_up_depth=0,
                        metadata={}
                    )
                    try:
                        await save_message(assistant_msg)
                    except Exception as e:
                        logger.warning(f"Failed to save assistant message: {e}")
                yield "data: " + json.dumps({"status": status, "chunk": chunk}) + "\n\n"
            # If session ended, update is_active flag in DB
            if session_ended:
                try:
                    conn = get_db_conn()
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE sessions SET is_active = FALSE WHERE session_id = %s
                        """,
                        (str(session_id),)
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                    logger.info(f"[send_message_stream] Updated is_active to FALSE for ended session_id={session_id}")
                except Exception as e:
                    logger.error(f"[send_message_stream] Failed to update is_active for ended session_id={session_id}: {e}")
        else:
            logger.info(f"[send_message_stream] Requirements met, streaming full response.")
            from app.core.services.chatbot_optimizer import OptimizedChatbot
            chatbot = OptimizedChatbot()
            query = req.query or ""
            chat_history = []
            for msg in conversation_history:
                if isinstance(msg, dict):
                    chat_history.append((msg.get('role', ''), msg.get('content', '')))
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    chat_history.append(tuple(msg))
            async for event in chatbot.get_detailed_response(query=query, chat_history=chat_history, session_id=session_id, stream=True):
                assistant_content = event.get("chunk") if isinstance(event, dict) else str(event)
                if assistant_content:
                    assistant_msg = MessageCreate(
                        session_id=session_id,
                        content=assistant_content,
                        role="assistant",
                        reply_to=None,
                        follow_up_to=None,
                        follow_up_depth=0,
                        metadata={}
                    )
                    try:
                        await save_message(assistant_msg)
                    except Exception as e:
                        logger.warning(f"Failed to save assistant message: {e}")
                yield "data: " + json.dumps(event) + "\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream", headers=SSE_HEADERS)


async def fetch_root_prompts():
    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        greeting_text = "Hello! I’m **Dits AI** 👋 — your smart assistant from Ditstek Innovations.\n\n**What brings you here today?**"
        bottom_hint_text = "**Feel free to type if you’re looking for something else!**"
        desired_order = [
            "See our Work",
            "Start a Project",
            "Talk to Dits team",
            "Explore Dits Services",
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

def delete_last_user_message(session_id: str):
    """Delete the last user message for a session from the database."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            DELETE FROM messages
            WHERE id = (
                SELECT id FROM messages
                WHERE session_id = %s AND role = 'user'
                ORDER BY created_at DESC
                LIMIT 1
            )
            RETURNING id
            """,
            (str(session_id),)
        )
        deleted = cursor.fetchone()
        conn.commit()
        logger.info(f"[delete_last_user_message] Deleted message id: {deleted[0] if deleted else None} for session_id={session_id}")
        return deleted[0] if deleted else None
    except Exception as e:
        logger.error(f"[delete_last_user_message] Error deleting last user message for session_id={session_id}: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()
        
async def end_session_helper(session_id: str):
    """End a session and trigger closure email in the background if conditions are met."""
    import threading
    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        # 1. Check is_active for the session
        cursor.execute(
            """
            SELECT is_active, user_id FROM sessions WHERE session_id = %s
            """,
            (str(session_id),)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        is_active, user_id = row
        # 2. If is_active is True, set to False
        if is_active:
            cursor.execute(
                """
                UPDATE sessions SET is_active = FALSE WHERE session_id = %s
                """,
                (str(session_id),)
            )
            conn.commit()
            is_active = False
        # 3. Check user_details_known for the user
        cursor.execute(
            """
            SELECT user_details_known FROM users WHERE id = %s
            """,
            (str(user_id),)
        )
        user_row = cursor.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")
        user_details_known = user_row[0]
        # 4. If is_active is False and user_details_known is True, trigger closure email in background
        def trigger_email():
            try:
                send_closure_email(session_id)
            except Exception as e:
                logger.error(f"Error sending closure email: {e}")
        if (is_active is False) and (user_details_known is True):
            threading.Thread(target=trigger_email, daemon=True).start()
        # Return minimal response
        return {
            "status": "success",
            "session_id": session_id,
            "is_active": is_active,
            "user_details_known": user_details_known
        }
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error in end_session_helper: {e}")
        raise HTTPException(status_code=500, detail="Error ending session")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()