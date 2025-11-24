import re
import os
import json
import asyncio
import functools
from typing import Dict, Any, List, Optional, Any as AnyType
from fastapi.responses import StreamingResponse
from uuid import UUID
from app.utils.llm_utils import generate_llm_response
from app.logger import get_logger
from fastapi import HTTPException, Depends
from app.api.models import (
    MessageCreate,
    Message,
    SentMessage,
)
from app.api.models import UserCreate
from app.core.email_sender import send_closure_email
from fastapi import HTTPException
from datetime import datetime
from app.api.models import PromptType
from app.db.base import get_db_conn
from app.utils.redis_context import append_message_to_chat_history


logger = get_logger(__name__)
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
}

# --- Form Trigger Logic (no longer used, logic moved to prompt) ---


def get_user_details_known_from_db(session_id: str) -> bool:
    """Fetch user_details_known from the database for the user associated with the session."""
    logger.info(
        f"[get_user_details_known_from_db] Fetching user_details_known for session_id={session_id}"
    )
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.user_details_known FROM users u
            JOIN sessions s ON u.id = s.user_id
            WHERE s.session_id = %s
            """,
            (str(session_id),),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        logger.debug(f"[get_user_details_known_from_db] DB row: {row}")
        if row and row[0]:
            logger.info(
                f"[get_user_details_known_from_db] user_details_known=True for session_id={session_id}"
            )
            return bool(row[0])
    except Exception as e:
        logger.warning(
            f"[get_user_details_known_from_db] DB error for session {session_id}: {e}"
        )
    logger.info(
        f"[get_user_details_known_from_db] user_details_known=False for session_id={session_id}"
    )
    return False



def get_user_details_from_db(session_id: str) -> Dict[str, Any]:
    """Fetch user details (username, email, mobile) from the database for the user associated with the session.

    Returns a dictionary with user details if available, empty dict otherwise.
    """
    logger.info(
        f"[get_user_details_from_db] Fetching user details for session_id={session_id}"
    )
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.username, u.email, u.mobile, u.user_details_known 
            FROM users u
            JOIN sessions s ON u.id = s.user_id
            WHERE s.session_id = %s
            """,
            (str(session_id),),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            username, email, mobile, user_details_known = row
            user_details = {}
            if username:
                user_details["username"] = username
            if email:
                user_details["email"] = email
            if mobile:
                user_details["mobile"] = mobile
            if user_details_known:
                user_details["user_details_known"] = True

            if user_details:
                logger.info(
                    f"[get_user_details_from_db] Found user details for session_id={session_id}: {list(user_details.keys())}"
                )
                return user_details
    except Exception as e:
        logger.warning(
            f"[get_user_details_from_db] DB error for session {session_id}: {e}"
        )

    logger.info(
        f"[get_user_details_from_db] No user details found for session_id={session_id}"
    )
    return {}


def mark_form_shown(session_data: dict):
    logger.info(
        f"[mark_form_shown] Marking form_shown=True for session_id={session_data.get('session_id')}"
    )
    session_data["form_shown"] = True
    return session_data


def is_valid_ip(ip_str: str) -> bool:
    # Very small regex-based validation for IPv4 and IPv6-like patterns (best-effort)
    logger.info(f"[is_valid_ip] Validating IP: {ip_str}")
    if not ip_str:
        logger.debug("[is_valid_ip] IP string is empty.")
        return False
    ip_str = str(ip_str).strip()
    ipv4_re = r"^((25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(25[0-5]|2[0-4]\\d|[01]?\\d?\\d)$"
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


def _update_user_by_session_sync(session_id: str, user: UserCreate):
    """Update a user's fields using the session_id (Synchronous).

    Accepts a `UserCreate` model and performs a partial update of only the
    provided fields (username, email, mobile). Also keeps the
    sessions table in sync for browser/ip when provided.

    Returns the session_id on success.
    """
    try:
        logger.info(
            f"[update_user_by_session] Called with session_id={session_id}, payload={user.dict()}"
        )
        sid = UUID(str(session_id))
    except Exception as ex:
        logger.error(
            f"[update_user_by_session] Invalid session_id format: {session_id} | Exception: {ex}"
        )
        raise HTTPException(status_code=422, detail="Invalid session_id format")

    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        logger.info("[update_user_by_session] DB connection established.")

        # Find the user for this session
        cursor.execute(
            """
            SELECT user_id::text FROM sessions WHERE session_id = %s
            """,
            (str(sid),),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            logger.error(
                f"[update_user_by_session] Session not found for session_id={session_id}"
            )
            raise HTTPException(status_code=404, detail="Session not found")

        user_id = row[0]
        logger.info(
            f"[update_user_by_session] Found user_id={user_id} for session_id={session_id}"
        )

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
            logger.error(
                f"[update_user_by_session] No user fields provided to update for user_id={user_id}"
            )
            raise HTTPException(
                status_code=400, detail="No user fields provided to update"
            )

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
            logger.error(
                f"[update_user_by_session] User not found for user_id={user_id}, rolling back."
            )
            conn.rollback()
            raise HTTPException(status_code=404, detail="User not found for session")
        else:
            conn.commit()
            logger.info(
                f"[update_user_by_session] User updated successfully for user_id={user_id}"
            )

        # Session state management removed - using optimized flow only
        logger.info(
            f"[update_user_by_session] User details updated successfully for session_id={session_id}"
        )
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(
            f"[update_user_by_session] Error updating user by session {session_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Error updating user")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logger.info(f"[update_user_by_session] DB connection closed.")


async def update_user_by_session(session_id: str, user: UserCreate):
    """Update a user's fields using the session_id (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _update_user_by_session_sync, session_id, user)


def _initialize_session_with_prompt_sync(
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


async def initialize_session_with_prompt(
    session_id: UUID, prompt_id: UUID
) -> Dict[str, Any]:
    """Initialize session with prompt (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _initialize_session_with_prompt_sync, session_id, prompt_id)


def _save_message_sync(message_data: MessageCreate) -> Message:
    """Synchronous implementation of save_message."""
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
            raise HTTPException(
                status_code=500, detail="Failed to save message to database"
            )
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


async def save_message(message_data: MessageCreate) -> Message:
    """Save a message to the database (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _save_message_sync, message_data)


def _get_messages_for_session_sync(session_id: UUID) -> List[Message]:
    """Retrieve all messages for a given session ordered by creation time (Synchronous).

    Returns a list of `Message` objects expected by the router.
    """
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, session_id, content, role, reply_to, follow_up_to, 
                   follow_up_depth, metadata, created_at, updated_at
            FROM messages 
            WHERE session_id = %s 
            ORDER BY created_at ASC
            """,
            (str(session_id),)
        )
        
        rows = cursor.fetchall()
        messages = []
        
        for row in rows:
            message = Message(
                id=str(row[0]),
                session_id=str(row[1]),
                content=row[2],
                role=row[3],
                reply_to=str(row[4]) if row[4] else None,
                follow_up_to=str(row[5]) if row[5] else None,
                follow_up_depth=row[6],
                metadata=row[7] or {},
                created_at=row[8],
                updated_at=row[9]
            )
            messages.append(message)
        
        return messages
        
    finally:
        cursor.close()
        conn.close()


async def get_messages_for_session(session_id: UUID) -> List[Message]:
    """Retrieve all messages for a given session (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_messages_for_session_sync, session_id)


def _fetch_root_prompts_sync():
    conn = None
    cursor = None
    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        greeting_text = "Hello! I'm **DITS AI** 👋 — your smart assistant from Ditstek Innovations.\n\n**What brings you here today?**"
        bottom_hint_text = "**Feel free to type if you're looking for something else!**"
        desired_order = [
            "See our Work",
            "Start a Project",
            "Talk to DITS team",
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


async def fetch_root_prompts():
    """Fetch root prompts (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_root_prompts_sync)


async def send_message_stream(req: SentMessage):
    # Simplified logging
    # logger.info(f"[send_message_stream] Called for session_id={getattr(req, 'session_id', None)}")

    # Extract required variables from req or context
    session_id = getattr(req, "session_id", None)
    conversation_history = getattr(req, "conversation_history", [])
    # Prefer an explicit query if provided by the client. This ensures the follow-up
    # flow has a usable latest_query for Redis context retrieval even when
    # conversation_history or prompt_context are empty.
    prompt_context = getattr(req, "query", None) or getattr(req, "prompt_context", None)

    # --- Unified streaming logic: yield processing event once, then stream follow-up or final response ---

    async def stream_response():
        # Only yield 'processing' once, from here (not from LLM)
        yield "data: " + json.dumps({"status": "processing", "message": "Preparing response..."}) + "\n\n"
        if session_id is None or not isinstance(session_id, str):
            logger.error("[send_message_stream] session_id is missing or not a string.")
            raise HTTPException(status_code=400, detail="session_id is required and must be a string")

        # Prepare user message for later saving (defer to improve response time)
        user_msg_to_save = None
        if req.query:
            user_msg_to_save = MessageCreate(
                session_id=session_id,
                content=req.query,
                role="user",
                reply_to=None,
                follow_up_to=None,
                follow_up_depth=0,
                metadata={}
            )

        # Save user message asynchronously to improve response time
        # Start user message saving in background
        if user_msg_to_save:
            asyncio.create_task(save_message(user_msg_to_save))
            try:
                # also persist to Redis chat_history asynchronously
                asyncio.create_task(asyncio.to_thread(append_message_to_chat_history, session_id, {"role": "user", "content": req.query, "timestamp": None}))
            except Exception:
                # if asyncio.to_thread not available, fallback to run_in_executor
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, append_message_to_chat_history, session_id, {"role": "user", "content": req.query, "timestamp": None})

        # Always use the optimized chatbot flow
        # logger.info("[send_message_stream] Using optimized chatbot flow.")
        from app.core.chatbot_optimizer import OptimizedChatbot
        chatbot = OptimizedChatbot()
        query = req.query or ""
        
        # Use provided conversation history or get from database only if needed
        if conversation_history:
            chat_history = [(msg.get("role"), msg.get("content")) for msg in conversation_history if msg.get("role") in ["user", "assistant"]]
            logger.info(f"[OPTIMIZED_FLOW] Using provided conversation history: {len(chat_history)} messages")
        else:
            try:
                from uuid import UUID
                session_uuid = UUID(session_id)
                messages = await get_messages_for_session(session_uuid)
                chat_history = [(msg.role, msg.content) for msg in messages if msg.role in ["user", "assistant"]]
                # logger.info(f"[OPTIMIZED_FLOW] Using conversation history from database: {len(chat_history)} messages")
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
                chat_history = []
        
        full_assistant_response = ""
        session_ended = False

        async for event in chatbot.get_detailed_response(query=query, chat_history=chat_history, session_id=session_id, stream=True):
            # Track session ending
            if isinstance(event, dict) and event.get("status") == "end_chat":
                session_ended = True
            # Build complete response from chunks only (not meta events)
            if isinstance(event, dict) and event.get("status") == "chunk":
                chunk_content = event.get("chunk", "")
                if chunk_content:
                    chunk_str = str(chunk_content) if not isinstance(chunk_content, str) else chunk_content
                    full_assistant_response += chunk_str  # Preserve all characters including markdown formatting
            elif not isinstance(event, dict):
                # Handle non-dict events
                full_assistant_response += str(event)
            # Do NOT flatten or clean newlines; preserve markdown as-is
            yield "data: " + json.dumps(event) + "\n\n"
        
        # After streaming is complete, save complete message and handle session ending
        if full_assistant_response:
            try:
                # Debug markdown preservation in optimized flow
                has_markdown = any(marker in full_assistant_response for marker in ["**", "*", "_", "#", "-", "`", "```"])
                if has_markdown:
                    pass # logger.info(f"[OPTIMIZED_FLOW] Response contains markdown: {full_assistant_response[:200]}...")
                
                # Save complete message to database (only once)
                assistant_msg = MessageCreate(
                    session_id=session_id,
                    content=full_assistant_response,
                    role="assistant",
                    reply_to=None,
                    follow_up_to=None,
                    follow_up_depth=0,
                    metadata={}
                )
                # Fire and forget save to close stream immediately
                asyncio.create_task(save_message(assistant_msg))
                try:
                    asyncio.create_task(asyncio.to_thread(append_message_to_chat_history, session_id, {"role": "assistant", "content": full_assistant_response, "timestamp": None}))
                except Exception:
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(None, append_message_to_chat_history, session_id, {"role": "assistant", "content": full_assistant_response, "timestamp": None})
                # logger.info(f"[OPTIMIZED_FLOW] Saved complete response: {len(full_assistant_response)} chars")
                        
            except Exception as e:
                logger.warning(f"Failed to save complete response: {e}")
        
        # Handle session ending asynchronously
        if session_ended:
            async def end_session_async():
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
                    logger.info(f"[OPTIMIZED_FLOW] Updated is_active to FALSE for ended session_id={session_id}")
                except Exception as e:
                    logger.error(f"[OPTIMIZED_FLOW] Failed to update is_active for ended session_id={session_id}: {e}")
            
            asyncio.create_task(end_session_async())

    return StreamingResponse(stream_response(), media_type="text/event-stream", headers=SSE_HEADERS)


def _delete_last_user_message_sync(session_id: str):
    """Delete the last user message for a session from the database (Synchronous)."""
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


async def delete_last_user_message(session_id: str):
    """Delete the last user message (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _delete_last_user_message_sync, session_id)


def _get_session_is_active_sync(session_id: UUID) -> bool:
    """Check if session is active (Synchronous)."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT is_active FROM sessions WHERE session_id = %s",
            (str(session_id),)
        )
        row = cursor.fetchone()
        return bool(row[0]) if row and row[0] is not None else False
    except Exception as e:
        logger.warning(f"Could not fetch is_active for session {session_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


async def get_session_is_active(session_id: UUID) -> bool:
    """Check if session is active (non-blocking wrapper)."""
    import asyncio
    import functools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_session_is_active_sync, session_id)


async def end_session_helper(session_id: str):
    """
    End a session and trigger closure email in the background ONLY IF:
    - user_details_known flag is True
    - user session has been ended (is_active is False)
    If both conditions are true, then and only then send the email.
    """
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
        # 4. Trigger closure email ONLY if BOTH conditions are met
        if (is_active is False) and (user_details_known is True):
            def trigger_email():
                try:
                    send_closure_email(session_id)
                except Exception as e:
                    logger.error(f"Error sending closure email: {e}")
            threading.Thread(target=trigger_email, daemon=True).start()
        else:
            logger.info(f"[end_session_helper] Email NOT triggered: is_active={is_active}, user_details_known={user_details_known}")
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
            


