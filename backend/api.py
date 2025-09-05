import markdown2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any, Generator
from uuid import uuid4
from datetime import datetime
import psycopg2
import json
import os
from backend.chat_logic import build_chatbot_response
from .services.chatbot_optimizer import OptimizedChatbot
from backend.llm_client import llm  # Make sure llm is initialized before importing chat_logic

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read env vars
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Make sure llm is already initialized
optimized_chatbot = OptimizedChatbot(llm, model="gpt-4o-mini")
print("✅ Optimized chatbot initialized successfully")

app = FastAPI()
# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Pydantic Models
# ----------------------------
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    mobile: str
    browser: str
    ip: str

class UserRegisterResponse(BaseModel):
    status: str
    message: str
    session_id: str

class SentMessage(BaseModel):
    query: str
    session_id: str
    stream: Optional[bool] = False  # Add streaming option
    detailed: Optional[bool] = False  # 👈 New flag for concise/detailed

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    source: Optional[str] = None
    matched: bool = True

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

class StreamingChatResponse(BaseModel):
    session_id: str
    chunk: str
    source: Optional[str] = None
    matched: bool = True
    done: bool = False

# ----------------------------
# DB Helpers
# ----------------------------
def _get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT,
        options="-c client_encoding=UTF8"
    )

def save_user_and_new_session(*, username, email, mobile, browser, ip) -> str:
    conn = _get_conn()
    cursor = conn.cursor()
    # 1. Ensure user exists
    cursor.execute("""
        INSERT INTO users (username, email, mobile, browser, ip)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (email) DO UPDATE
        SET username = EXCLUDED.username,
            mobile   = EXCLUDED.mobile,
            browser  = EXCLUDED.browser,
            ip       = EXCLUDED.ip
        RETURNING id
    """, (username, email, mobile, browser, ip))
    user_id = cursor.fetchone()[0]
    # 2. Deactivate old sessions
    cursor.execute("UPDATE sessions SET is_active = FALSE WHERE user_id = %s", (user_id,))
    # 3. Create new session
    session_id = str(uuid4())
    cursor.execute("""
        INSERT INTO sessions (user_id, session_id, title, browser, ip, is_active)
        VALUES (%s, %s, %s, %s, %s, TRUE)
    """, (user_id, session_id, "New Chat", browser, ip))
    
    # 4. Insert default welcome message from bot
    cursor.execute("""
        INSERT INTO messages (session_id, role, message, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (session_id, "bot", f"Hello {username} 👋! How can I assist you today?", datetime.now().isoformat()))
    conn.commit()
    cursor.close()
    conn.close()
    return session_id

def save_message(*, session_id, role, message, timestamp):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, role, message, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (session_id, role, message, timestamp))
    conn.commit()
    cursor.close()
    conn.close()

def get_messages_for_session(session_id):
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message, timestamp
        FROM messages
        WHERE session_id = %s
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# ----------------------------
# Streaming Helpers
# ----------------------------
def stream_response_generator(query: str, session_id: str) -> Generator[str, None, None]:
    """Generator for streaming responses"""
    timestamp = datetime.now().isoformat()
    
    # Get chat history
    rows = get_messages_for_session(session_id)
    history = [(r, m) for (r, m, _) in rows]
    
    # Get streaming response from chatbot
    try:
        response_generator = optimized_chatbot.get_detailed_response(
            query=query,
            chat_history=history,
            stream=True
        )
        
        full_response = ""
        source_flag = None
        matched = True
        
        # Stream chunks
        for chunk in response_generator:
            full_response += chunk
            
            # Convert chunk to HTML
            #chunk_html = markdown2.markdown(chunk, extras=["fenced-code-blocks", "tables"]).strip()
            
            # Send chunk as SSE
            data = {
                "session_id": session_id,
                "chunk": chunk,
                "source": source_flag,
                "matched": matched,
                "done": False
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Save complete response to database
        save_message(session_id=session_id, role="user", message=query.strip(), timestamp=timestamp)
        save_message(session_id=session_id, role="bot", message=full_response, timestamp=timestamp)
        
        # Send completion signal
        completion_data = {
            "session_id": session_id,
            "chunk": "",
            "source": source_flag,
            "matched": matched,
            "done": True
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        # Handle streaming errors
        error_data = {
            "session_id": session_id,
            "chunk": f"I apologize, but I'm experiencing technical difficulties: {str(e)}",
            "source": None,
            "matched": False,
            "done": True
        }
        yield f"data: {json.dumps(error_data)}\n\n"

# ----------------------------
# API Routes
# ----------------------------
@app.post("/user/register", response_model=UserRegisterResponse)
def register_user(user: UserCreate):
    try:
        session_id = save_user_and_new_session(
            username=user.username,
            email=user.email,
            mobile=user.mobile,
            browser=user.browser,
            ip=user.ip,
        )
        return UserRegisterResponse(
            status="success",
            message="User registered successfully",
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/send", response_model=ChatResponse)
def send_message(req: SentMessage):
    """Non-streaming chat endpoint (backward compatibility)"""
    timestamp = datetime.now().isoformat()
    
    # Get chat history
    rows = get_messages_for_session(req.session_id)
    history = [(r, m) for (r, m, _) in rows]
    
    # Build response
    result = build_chatbot_response(req.query, history)
    try:
        answer, matched, meta = result
    except ValueError:
        answer, matched = result
        meta = {}
    
    # Clean LLM answer
    answer = answer.strip()
    # Convert answer to HTML using markdown2
    answer_html = markdown2.markdown(answer, extras=["fenced-code-blocks", "tables"]).strip()
    
    # Save messages
    save_message(session_id=req.session_id, role="user", message=req.query.strip(), timestamp=timestamp)
    save_message(session_id=req.session_id, role="bot", message=answer, timestamp=timestamp)
    
    # Source info
    source_flag = None
    if isinstance(meta, dict):
        if meta.get("used_web"):
            source_flag = "internet"
        elif meta.get("used_kb"):
            source_flag = "knowledge_base"
    
    return ChatResponse(
        session_id=req.session_id,
        answer=answer_html,
        source=source_flag,
        matched=matched,
    )

@app.post("/chat/send-stream")
async def send_message_stream(req: SentMessage):
    """Streaming chat endpoint"""
    return StreamingResponse(
        stream_response_generator(req.query, req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/chat/{session_id}/messages", response_model=HistoryResponse)
def get_chat_messages(session_id: str):
    rows = get_messages_for_session(session_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No messages found for this session")
    messages = [
        {"role": role, 
         "message": msg,
         "timestamp": ts.isoformat() if ts else None
         }
        for (role, msg, ts) in rows
    ]
    return HistoryResponse(session_id=session_id, messages=messages)

# ----------------------------
# Health Check
# ----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "streaming": "enabled",
        "chatbot": "optimized"
    }