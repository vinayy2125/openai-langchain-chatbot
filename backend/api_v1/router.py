from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import json
import logging
from typing import Optional, List
from backend.services.thread_router import ChatRouter, detect_company_intent, handle_company_query

from .models import (
    UserCreate, UserRegisterResponse, SentMessage, 
    HistoryResponse, StreamingChatResponse,
    SessionState, Prompt, MessageCreate,
    PromptType
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

# Message History Route
@router.get("/chat/{session_id}/messages", response_model=HistoryResponse)
async def get_chat_messages(session_id: str):
    """Get all messages for a chat session."""
    try:
        messages = await get_messages_for_session(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for this session")

        formatted_messages = []
        for (role, msg, ts) in messages:
            formatted_messages.append({
                "role": role, 
                "message": msg,
                "timestamp": ts.isoformat() if ts else None
            })
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

chat_router: ChatRouter = None

@router.post("/chat/send-stream")
async def send_message_stream(
    req: SentMessage,
    follow_up_manager=Depends(get_follow_up_manager)
):
    """
    Enhanced streaming chat endpoint:
    - Thread/topic management
    - Company-intent override
    - Follow-up or complete response handling
    - Context-switch detection with suggestions
    """
    try:
        session_id = (req.session_id or '').strip()
        if not session_id:
            raise HTTPException(status_code=422, detail="Invalid session_id provided")

        # Lazy init ChatRouter
        global chat_router
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
                    prompt_db = await initialize_sessvion_with_prompt(session_id, req.prompt_id)
                    prompt_context = prompt_db["prompt_text"]
                    prompt_id_str = str(req.prompt_id)
                except HTTPException:
                    prompt_context = (req.query or '').strip()
            else:
                prompt_context = (req.query or '').strip() or "General assistance"

            follow_up_manager.initialize_session(
                session_id=session_id,
                prompt_id=prompt_id_str,
                prompt_context=prompt_context
            )

            if prompt_id_str and hasattr(follow_up_manager.chatbot, 'reset_follow_up_count'):
                follow_up_manager.chatbot.reset_follow_up_count(session_id)

            session_data = follow_up_manager.get_session_data(session_id)

            if req.query:
                await save_message(MessageCreate(
                    content=req.query.strip(),
                    role="user",
                    session_id=session_id
                ))
                follow_up_manager.add_to_conversation_history(session_id, "user", req.query.strip())
        else:
            if req.prompt_id and str(req.prompt_id) != str(session_data.get("prompt_id")):
                logger.warning(
                    f"Ignoring new prompt_id {req.prompt_id} for existing session {session_id}; "
                    f"continuing with original {session_data.get('prompt_id')}"
                )
            if req.query:
                await save_message(MessageCreate(
                    content=req.query.strip(),
                    role="user",
                    session_id=session_id
                ))
                follow_up_manager.add_to_conversation_history(session_id, "user", req.query.strip())

        conversation_history = follow_up_manager.get_conversation_history(session_id)
        prompt_context = session_data.get('prompt_context')

        # --- Company intent override ---
        if req.query and detect_company_intent(req.query):
            async def company_stream():
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Fetching company info...'})}\n\n"
                answer_and_suggestions = handle_company_query(req.query)
                for msg in answer_and_suggestions:
                    yield f"data: {json.dumps({'status': 'company', 'message': msg})}\n\n"
            return StreamingResponse(company_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

        # --- Decide if more follow-ups are needed ---
        if not follow_up_manager.check_requirements(session_id):
            async def stream_follow_up():
                from backend.chat_logic import build_chatbot_response
                # Add the `data:` prefix to the JSON-encoded string
                yield "data: " + json.dumps({'status': 'processing', 'message': 'Preparing response...'}) + "\n\n"

                # Reuse streaming follow-up generator with context switch & suggestions
                async for evt in build_chatbot_response(
                    session_id=session_id,
                    follow_up_manager=follow_up_manager,
                    conversation_history=conversation_history,
                    prompt_context=prompt_context,
                    mode="follow_up"
                ):
                    # Handle different event types with proper formatting
                    if isinstance(evt, dict):
                        status = evt.get('status', 'unknown')
                        chunk = evt.get('chunk', '')
                        
                        if status == 'complete_chunk':
                            # Main response content - keep as chunks for streaming
                            yield "data: " + json.dumps({'status': 'complete_chunk', 'chunk': chunk}) + "\n\n"
                        elif status == 'separator':
                            # Separator before final suggestions
                            yield "data: " + json.dumps({'status': 'separator', 'chunk': chunk}) + "\n\n"
                        elif status == 'followup_header':
                            # Header for follow-up section
                            yield "data: " + json.dumps({'status': 'followup_header', 'chunk': chunk}) + "\n\n"
                        elif status == 'followup_question':
                            # Individual follow-up question
                            yield "data: " + json.dumps({'status': 'followup', 'chunk': chunk}) + "\n\n"
                        elif status == 'suggestion':
                            # Suggestion
                            yield "data: " + json.dumps({'status': 'suggestion', 'chunk': chunk}) + "\n\n"
                        elif status == 'error':
                            # Error handling
                            yield "data: " + json.dumps({'status': 'error', 'message': evt.get('message', 'Unknown error')}) + "\n\n"
                        else:
                            # Default handling - preserve original structure
                            yield "data: " + json.dumps(evt) + "\n\n"
                    else:
                        # Fallback for non-dict events
                        yield "data: " + json.dumps({'status': 'complete_chunk', 'chunk': str(evt)}) + "\n\n"

            return StreamingResponse(stream_follow_up(), media_type="text/event-stream", headers=SSE_HEADERS)

        # --- Requirements captured: full response streaming ---
        async def generate_full_response_stream():
            from backend.chat_logic import build_chatbot_response
            yield "data: " + json.dumps({'status': 'processing', 'message': 'Generating comprehensive response...'}) + "\n\n"
            
            async for evt in build_chatbot_response(
                session_id=session_id,
                follow_up_manager=follow_up_manager,
                conversation_history=conversation_history,
                prompt_context=prompt_context,
                mode='complete'
            ):
                # Handle comprehensive response events
                if isinstance(evt, dict):
                    status = evt.get('status', 'unknown')
                    chunk = evt.get('chunk', '')
                    
                    if status == 'complete_chunk':
                        # Main comprehensive response content - keep as chunks for consistent streaming
                        yield "data: " + json.dumps({'status': 'complete_chunk', 'chunk': chunk}) + "\n\n"
                    elif status == 'separator':
                        # Separator before final suggestions
                        yield "data: " + json.dumps({'status': 'separator', 'chunk': chunk}) + "\n\n"
                    elif status == 'followup_header':
                        # Header for exploration suggestions (keeping for compatibility)
                        yield "data: " + json.dumps({'status': 'followup_header', 'chunk': chunk}) + "\n\n"
                    elif status == 'followup_question':
                        # Follow-up question
                        yield "data: " + json.dumps({'status': 'followup', 'chunk': chunk}) + "\n\n"
                    elif status == 'suggestion':
                        # Suggestion
                        yield "data: " + json.dumps({'status': 'suggestion', 'chunk': chunk}) + "\n\n"
                    elif status == 'error':
                        # Error handling
                        yield "data: " + json.dumps({'status': 'error', 'message': evt.get('message', 'Unknown error')}) + "\n\n"
                    else:
                        # Default handling - preserve original structure
                        yield "data: " + json.dumps(evt) + "\n\n"
                else:
                    # Fallback for non-dict events
                    yield "data: " + json.dumps({'status': 'complete_chunk', 'chunk': str(evt)}) + "\n\n"

        return StreamingResponse(generate_full_response_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error in send_message_stream: {str(e)}")

        async def generate_error_stream(e=e):
            yield f"data: {json.dumps({'status': 'processing', 'message': 'An error occurred'})}\n\n"
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
