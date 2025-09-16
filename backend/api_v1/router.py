from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
import json
from datetime import datetime
import logging
from typing import Optional, List
from uuid import UUID, uuid4
import psycopg2

from .models import (
    UserCreate, UserRegisterResponse, SentMessage, 
    HistoryResponse, StreamingChatResponse,
    SessionState, Prompt, FollowUp, MessageCreate,
    PromptType, FollowUpType
)
from backend.nested_follow_up_manager import FollowUpManager
from .helpers import (
    initialize_session_with_prompt, save_message, 
    get_messages_for_session, generate_follow_up
)
from backend.db_utils import get_db_conn

# Get logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["v1"])

# Dependencies
_follow_up_manager_instance: Optional[FollowUpManager] = None

def get_follow_up_manager():
    """Return a singleton FollowUpManager so in-memory session state persists across requests."""
    global _follow_up_manager_instance
    if _follow_up_manager_instance is None:
        from backend.llm_client import llm
        _follow_up_manager_instance = FollowUpManager(llm=llm)
    return _follow_up_manager_instance

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
        cursor.execute("""
            INSERT INTO users (browser, ip)
            VALUES (%s, %s)
            RETURNING id::text
        """, (
            user.browser,
            user.ip
        ))
        user_id = cursor.fetchone()[0]  # Get the UUID as string
        
        # Create new session - let Postgres handle both UUIDs
        cursor.execute("""
            INSERT INTO sessions (user_id, browser, ip, is_active)
            VALUES (%s, %s, %s, TRUE)
            RETURNING session_id::text
        """, (user_id, user.browser, user.ip))
        
        session_id = cursor.fetchone()[0]  # Get the UUID as string
        conn.commit()
        
        return UserRegisterResponse(
            status="success",
            message="Session created successfully",
            session_id=session_id
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
        conn.close()

def save_user_to_db(
    *, 
    username: Optional[str] = None,
    email: Optional[str] = None,
    mobile: Optional[str] = None,
    browser: str,
    ip: str
) -> str:
    """Insert a new user if not exists, return the user ID."""
    conn = get_db_conn()
    cursor = conn.cursor()

    try:
        if email:
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()
            if existing_user:
                # Update existing user's browser and IP
                cursor.execute(
                    """
                    UPDATE users 
                    SET browser = %s, ip = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE email = %s
                    RETURNING id
                    """,
                    (browser, ip, email)
                )
                return cursor.fetchone()[0]
        
        # Create new user
        cursor.execute(
            """
            INSERT INTO users (
                username, email, mobile, browser, ip,
                email_opt_in, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id
            """,
            (username, email, mobile, browser, ip, False)
        )
        user_id = cursor.fetchone()[0]
        conn.commit()
        return user_id
    except Exception as e:
        logger.error(f"Error in user registration: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create session")
    finally:
        cursor.close()
        conn.close()

# Message History Route
@router.get("/chat/{session_id}/messages", response_model=HistoryResponse)
async def get_chat_messages(session_id: str):
    """Get all messages for a chat session."""
    try:
        messages = await get_messages_for_session(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for this session")

        formatted_messages = [
            {
                "role": role, 
                "message": msg,
                "timestamp": ts.isoformat() if ts else None
            }
            for (role, msg, ts) in messages
        ]
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
    return {
        "status": "healthy",
        "streaming": "enabled",
        "chatbot": "optimized"
    }

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
        cursor.execute("""
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
        """)
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
                updated_at=row[5]
            )
            for row in rows
        ]
        
        return prompts

    except Exception as e:
        logger.error(f"Error fetching root prompts: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching prompts: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@router.post("/chat/send-stream")
async def send_message_stream(
    req: SentMessage,
    follow_up_manager = Depends(get_follow_up_manager)
):
    """
    Enhanced streaming chat endpoint that handles:
    1. Initial prompt selection and follow-up generation
    2. Follow-up conversation flow
    3. Final detailed response with suggestions
    """
    try:
        # 1. Validate session id
        session_id = (req.session_id or '').strip()
        if not session_id:
            raise HTTPException(status_code=422, detail="Invalid session_id provided")

        # 2. Fetch existing session (in-memory) – persists via singleton manager
        session_data = follow_up_manager.get_session_data(session_id)
        is_new = not session_data or not session_data.get("prompt_context")

        # 3. If new: establish prompt context from prompt_id OR first free-text query
        if is_new:
            prompt_context = None
            prompt_id_str = None
            if req.prompt_id:
                try:
                    prompt_db = await initialize_session_with_prompt(session_id, req.prompt_id)
                    prompt_context = prompt_db["prompt_text"]
                    prompt_id_str = str(req.prompt_id)
                except HTTPException:
                    # Fallback to free-text if DB prompt fetch fails and query exists
                    if req.query:
                        prompt_context = req.query.strip()
                    else:
                        raise
            else:
                prompt_context = (req.query or '').strip() or "General assistance"
            follow_up_manager.initialize_session(
                session_id=session_id,
                prompt_id=prompt_id_str,
                prompt_context=prompt_context
            )
            session_data = follow_up_manager.get_session_data(session_id)
            # If the initial free-text query started the session, store it as first user message only once
            if req.query:
                await save_message(MessageCreate(
                    content=req.query.strip(),
                    role="user",
                    session_id=session_id
                ))
                follow_up_manager.add_to_conversation_history(session_id, "user", req.query.strip())
        else:
            # Continuing session; ignore new prompt_id attempts
            if req.prompt_id and str(req.prompt_id) != str(session_data.get("prompt_id")):
                logger.warning(
                    f"Ignoring new prompt_id {req.prompt_id} for existing session {session_id}; "
                    f"continuing with original {session_data.get('prompt_id')}"
                )
            # Append user reply if provided
            if req.query:
                await save_message(MessageCreate(
                    content=req.query.strip(),
                    role="user",
                    session_id=session_id
                ))
                follow_up_manager.add_to_conversation_history(session_id, "user", req.query.strip())

        conversation_history = follow_up_manager.get_conversation_history(session_id)

        # Check if we need more follow-ups
        if not follow_up_manager.check_requirements(session_id):
            async def stream_follow_up():
                from backend.chat_logic import build_chatbot_response
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Preparing follow-up question...'})}\n\n"
                # Reuse unified streaming generator (follow_up mode)
                async for evt in build_chatbot_response(
                    session_id=session_id,
                    follow_up_manager=follow_up_manager,
                    conversation_history=conversation_history,
                    prompt_context=session_data.get('prompt_context'),
                    mode="follow_up"
                ):
                    yield evt
            return StreamingResponse(stream_follow_up(), media_type="text/event-stream", headers=SSE_HEADERS)

        # Requirements met: stream full response chunks using build_chatbot_response(mode='complete')
        async def generate_full_response_stream():
            from backend.chat_logic import build_chatbot_response
            async for evt in build_chatbot_response(
                session_id=session_id,
                follow_up_manager=follow_up_manager,
                conversation_history=conversation_history,
                prompt_context=session_data.get('prompt_context'),
                mode='complete'
            ):
                yield evt

        return StreamingResponse(generate_full_response_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error in send_message_stream: {str(e)}")
        
        async def generate_error_stream():
            # Initial error notification
            yield f"data: {json.dumps({'status': 'processing', 'message': 'An error occurred'})}\n\n"
            
            # Detailed error message
            error_response = {
                "status": "error",
                "message": "An unexpected error occurred",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(
            generate_error_stream(),
            media_type="text/event-stream",
            headers=SSE_HEADERS
        )

async def handle_follow_up_phase(
    session_id: str,
    query: str,
    state: SessionState,
    timestamp: str,
    llm,
    follow_up_manager
) -> StreamingResponse:
    """Handle the follow-up conversation phase."""
    
    if query:
        await save_message(MessageCreate(
            content=query.strip(),
            role="user",
            session_id=session_id
        ))
        follow_up_manager.add_to_conversation_history(session_id, "user", query.strip())
        state.follow_up_count += 1
        
        if state.follow_up_count >= 10:
            state.requirements_met = True
            return await handle_completion_phase(session_id, state, llm, follow_up_manager)
    
    # Generate next follow-up
    follow_up = await generate_follow_up(
        prompt_text=state.prompt_text,
        conversation_history=state.conversation_history,
        llm=llm
    )
    
    # Save system's follow-up
    await save_message(MessageCreate(
        content=follow_up.question,
        role="assistant",
        session_id=session_id
    ))
    
    # Update session state
    state.current_follow_up = follow_up
    follow_up_manager.get_session_data(session_id)["state"] = state.model_dump()
    
    response = StreamingChatResponse(
        status="follow_up",
        message="Gathering requirements",
        follow_up=follow_up,
        conversation_history=state.conversation_history
    )
    
    return StreamingResponse(
        iter([f"data: {json.dumps(response.model_dump())}\n\n"]),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )

async def handle_completion_phase(
    session_id: str,
    state: SessionState,
    llm,
    follow_up_manager
) -> StreamingResponse:
    """Handle the completion phase with detailed response."""
    state.requirements_met = True
    follow_up_manager.get_session_data(session_id)["state"] = state.model_dump()
    
    conversation_history = state.conversation_history
    messages = [
        {
            "role": "system",
            "content": f"""Generate a detailed response based on:
            
            Original Prompt: {state.prompt_text}
            
            Conversation History:
            {follow_up_manager.format_conversation_history(conversation_history)}
            
            Provide:
            1. Summary of requirements
            2. Detailed recommendations
            3. Next steps or suggestions
            4. Any relevant knowledge base references
            """
        }
    ]
    
    async def response_generator():
        completion_message = {
            "status": "complete",
            "message": "Generating detailed response...",
            "conversation_history": conversation_history
        }
        yield f"data: {json.dumps(completion_message)}\n\n"
        
        detailed_response = ""
        async for chunk in llm.stream(messages):
            if chunk:
                detailed_response += chunk
                yield f"data: {json.dumps({'status': 'complete', 'content': chunk})}\n\n"
        
        suggestions_prompt = f"""Based on the detailed response and conversation,
        suggest 3-5 next steps or follow-up actions."""
        
        suggestions = await llm.invoke(suggestions_prompt)
        final_message = {
            "status": "complete",
            "content": detailed_response,
            "suggestions": suggestions.split('\n'),
            "conversation_history": conversation_history
        }
        yield f"data: {json.dumps(final_message)}\n\n"
    
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )
