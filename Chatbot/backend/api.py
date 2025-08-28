from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from backend.chat_logic import build_chatbot_response
from backend.history import chat_sessions
from datetime import datetime
import psycopg2
from database_setup import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

app = FastAPI()

# Allow frontend to access the Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    source_url: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    spource: Optional[str] = None
    matched: bool = True
    
class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]
    
    
# --- Helper Functions ---
def save_session_to_db(session_id, title, timestamp):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sessions (session_id, title, timestamp)
        VALUES (%s, %s, %s)
        ON CONFLICT (session_id) DO UPDATE SET title = EXCLUDED.title, timestamp = EXCLUDED.timestamp
        """,
        (session_id, title, timestamp)
    )
    conn.commit()
    cursor.close()
    conn.close()

def save_message_to_db(session_id, role, message, timestamp):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, message, timestamp)
        VALUES (%s, %s, %s, %s)
        """,
        (session_id, role, message, timestamp)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_history_from_db(session_id):
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, message, timestamp FROM messages WHERE session_id = %s ORDER BY id
        """,
        (session_id,)
    )
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history


# --- API Routes ---
@app.post("/chat/new", response_model=ChatResponse)
def new_chat(req: ChatRequest):
    session_id = str(uuid4())
    timestamp = datetime.now().isoformat()  # Ensure timestamp is defined
    answer, matched = build_chatbot_response(req.query, [])

    # Save session and first message to the database
    save_session_to_db(session_id, req.query[:30], timestamp)
    save_message_to_db(session_id, "user", req.query, timestamp)
    save_message_to_db(session_id, "bot", answer, timestamp)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        source=req.source_url,
        matched=matched
    )


@app.post("/chat/continue", response_model=ChatResponse)
def continue_chat(req: ChatRequest):
    if req.session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    answer, matched = build_chatbot_response(req.query)
    timestamp = datetime.now().isoformat()  # Ensure timestamp is defined

    # Save new messages to the database
    save_message_to_db(req.session_id, "user", req.query, timestamp)
    save_message_to_db(req.session_id, "bot", answer, timestamp)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        matched=matched
    )


@app.get("/chat/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    history = get_chat_history_from_db(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    return HistoryResponse(
        session_id=session_id,
        messages=[{"role": role, "message": message, "timestamp": timestamp} for role, message, timestamp in history]
    )
    
    
@app.get("/ping")
def health_check():
    return {"status": "ok"}
