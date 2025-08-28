import markdown2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from uuid import uuid4
from datetime import datetime
import psycopg2

from database_setup import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from backend.chat_logic import build_chatbot_response


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


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    source: Optional[str] = None
    matched: bool = True


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]


# ----------------------------
# DB Helpers
# ----------------------------

def _get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
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
    """, (session_id, "bot", "Hello 👋 How can I assist you today?", datetime.now().isoformat()))

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
    timestamp = datetime.now().isoformat()

    # Build response
    rows = get_messages_for_session(req.session_id)
    history = [(r, m) for (r, m, _) in rows]
    result = build_chatbot_response(req.query, history)

    try:
        answer, matched, meta = result
    except ValueError:
        answer, matched = result
        meta = {}

    # Convert answer to HTML using markdown2
    answer_html = markdown2.markdown(answer, extras=["fenced-code-blocks", "tables"])
    
    # Save messages
    save_message(session_id=req.session_id, role="user", message=req.query, timestamp=timestamp)
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
        matched=matched
    )


@app.get("/chat/{session_id}/messages", response_model=HistoryResponse)
def get_chat_messages(session_id: str):
    rows = get_messages_for_session(session_id)
    if not rows:
        raise HTTPException(status_code=404, detail="No messages found for this session")

    messages = [
        {"role": role, 
        #  "message": msg,
         "message": markdown2.markdown(msg, extras=["fenced-code-blocks", "tables"]),
         "timestamp": ts.isoformat() if ts else None
         }
        for (role, msg, ts) in rows
    ]

    return HistoryResponse(session_id=session_id, messages=messages)
