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
from backend.llm_client import llm
from backend.logger import log_event
from backend.db_utils import _get_conn
import logging

from dotenv import load_dotenv
from functools import lru_cache

# ----------------------------
# Setup
# ----------------------------
load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

optimized_chatbot = OptimizedChatbot(llm, model="gpt-4o-mini")
print("✅ Optimized chatbot initialized successfully")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api")

# ----------------------------
# Pydantic Models
# ----------------------------
class UserCreate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    mobile: Optional[str] = None
    browser: str
    ip: str

class UserRegisterResponse(BaseModel):
    status: str
    message: str
    session_id: str

class SentMessage(BaseModel):
    query: Optional[str] = None
    session_id: str
    stream: Optional[bool] = False
    detailed: Optional[bool] = False
    selected_prompt_id: Optional[int] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    source: Optional[str] = None
    matched: bool = True

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

class Prompt(BaseModel):
    id: int
    prompt_text: str
    response_text: Optional[str] = None
    display_order: Optional[int] = None
    type: str  # root | category | leaf | followup

# ----------------------------
# DB Helpers
# ----------------------------
def save_user_and_new_session(*, username=None, email=None, mobile=None, browser, ip) -> str:
    conn = _get_conn()
    cursor = conn.cursor()

    # Check if the user already exists
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    if user:
        user_id = user[0]
        # Update existing user details
        cursor.execute(
            """
            UPDATE users
            SET username = COALESCE(%s, username),
                mobile = COALESCE(%s, mobile),
                browser = %s,
                ip = %s
            WHERE id = %s
            """,
            (username, mobile, browser, ip, user_id)
        )
    else:
        # Insert new user
        cursor.execute(
            """
            INSERT INTO users (username, email, mobile, browser, ip)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (username, email, mobile, browser, ip)
        )
        user_id = cursor.fetchone()[0]

    # Deactivate existing sessions for the user
    cursor.execute("UPDATE sessions SET is_active = FALSE WHERE user_id = %s", (user_id,))

    # Create a new session
    session_id = str(uuid4())
    cursor.execute(
        """
        INSERT INTO sessions (user_id, session_id, title, browser, ip, is_active)
        VALUES (%s, %s, %s, %s, %s, TRUE)
        """,
        (user_id, session_id, "New Chat", browser, ip)
    )

    # Insert a default bot message
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, message, timestamp)
        VALUES (%s, %s, %s, %s)
        """,
        (session_id, "bot", "Hello! How can I assist you today?", datetime.now().isoformat())
    )

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

@lru_cache(maxsize=128)
def _prompt_type(prompt_id, parent_id) -> str:
    if parent_id is None:
        return "root"
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prompts WHERE parent_id = %s", (prompt_id,))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return "category" if count > 0 else "leaf"

def _seed_root_prompts():
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prompts")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.executemany("""
            INSERT INTO prompts (prompt_text, response_text, display_order, parent_id)
            VALUES (%s, %s, %s, NULL)
        """, [
            ("AI/ML", None, 1),
            ("Do you want to explore our custom development services?", None, 2),
            ("Can you share your business goals or challenges?", None, 3),
            ("Would you like to discuss a tailored digital solution?", None, 4)
        ])
        conn.commit()
        print("✅ Seeded default root prompts")
    cursor.close()
    conn.close()

_seed_root_prompts()

def save_followups(parent_id: int, followups: List[str]) -> List[int]:
    """Persist generated follow-ups into prompts table as children."""
    conn = _get_conn()
    cursor = conn.cursor()
    ids = []
    for idx, text in enumerate(followups, start=1):
        # Check if the follow-up already exists
        cursor.execute("SELECT id FROM prompts WHERE parent_id = %s AND prompt_text = %s", (parent_id, text))
        existing_prompt = cursor.fetchone()
        if existing_prompt:
            logger.debug("Follow-up already exists: %s", text)
            ids.append(existing_prompt[0])
        else:
            cursor.execute(
                """
                INSERT INTO prompts (prompt_text, response_text, parent_id, display_order)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (text, None, parent_id, idx)
            )
            ids.append(cursor.fetchone()[0])
    conn.commit()
    cursor.close()
    conn.close()
    return ids

# ----------------------------
# Streaming Generator
# ----------------------------
def stream_response_generator(query: Optional[str], session_id: str, selected_prompt_id: Optional[int] = None):
    user_timestamp = datetime.now().isoformat()
    rows = get_messages_for_session(session_id)
    history = [(r, m) for (r, m, _) in rows]

    try:
        if selected_prompt_id:
            # Fetch prompt text
            conn = _get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT prompt_text FROM prompts WHERE id = %s", (selected_prompt_id,))
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if not row:
                raise ValueError("Prompt not found")
            prompt_text = row[0]

            # Combine prompt + user query if present
            combined_query = prompt_text
            if query:
                combined_query += f"\n\nUser additional input: {query}"

            # Generate main response (streaming)
            response_generator = optimized_chatbot.get_detailed_response(
                query=combined_query, chat_history=history, stream=True
            )
            full_response = ""
            for chunk in response_generator:
                full_response += chunk
                yield f"data: {json.dumps({'session_id': session_id,'chunk': chunk,'done': False})}\n\n"

            # Save messages
            save_message(session_id=session_id, role="user", message=query or prompt_text, timestamp=user_timestamp)
            bot_timestamp = datetime.now().isoformat()
            save_message(session_id=session_id, role="bot", message=full_response, timestamp=bot_timestamp)

            # Check if follow-ups already exist for this prompt
            existing_children = get_child_prompts(selected_prompt_id)
            if not existing_children:
                followup_prompt = f"Based on: '{prompt_text}', suggest 5 follow-up questions for deeper discussion."
                followups = optimized_chatbot.generate_followups(followup_prompt, num=5)
                followup_ids = save_followups(selected_prompt_id, followups)

                for fid, ftext in zip(followup_ids, followups):
                    yield f"data: {json.dumps({'session_id': session_id,'chunk': ftext,'prompt_id': fid,'parent_prompt_id': selected_prompt_id,'type': 'followup','done': False})}\n\n"
            else:
                # Return existing follow-ups
                for child in existing_children:
                    yield f"data: {json.dumps({'session_id': session_id,'chunk': child.prompt_text,'prompt_id': child.id,'parent_prompt_id': selected_prompt_id,'type': 'followup','done': False})}\n\n"

        else:
            # Free chat flow
            response_generator = optimized_chatbot.get_detailed_response(
                query=query, chat_history=history, stream=True
            )
            full_response = ""
            for chunk in response_generator:
                full_response += chunk
                yield f"data: {json.dumps({'session_id': session_id,'chunk': chunk,'done': False})}\n\n"

            save_message(session_id=session_id, role="user", message=query.strip(), timestamp=user_timestamp)
            bot_timestamp = datetime.now().isoformat()
            save_message(session_id=session_id, role="bot", message=full_response, timestamp=bot_timestamp)

        yield f"data: {json.dumps({'session_id': session_id,'chunk': '', 'done': True})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'session_id': session_id,'chunk': str(e),'done': True})}\n\n"


# ----------------------------
# API Routes
# ----------------------------
@app.post("/user/register", response_model=UserRegisterResponse)
def register_user(user: UserCreate):
    try:
        session_id = save_user_and_new_session(
            username=user.username, email=user.email, mobile=user.mobile,
            browser=user.browser, ip=user.ip,
        )
        return UserRegisterResponse(status="success", message="User registered successfully", session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/send-stream")
async def send_message_stream(req: SentMessage):
    """Streaming chat endpoint"""
    log_event("LLM request initiated", session_id=req.session_id, query=req.query[:200])

    # Save the selected root prompt if clicked
    if req.selected_prompt_id:
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT prompt_text FROM prompts WHERE id = %s", (req.selected_prompt_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            save_message(session_id=req.session_id, role="user", message=row[0], timestamp=datetime.now().isoformat())

    return StreamingResponse(
        stream_response_generator(req.query, req.session_id, req.selected_prompt_id),
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
        {
            "role": role,
            "message": msg.strip(),  # Ensure message is clean and includes selected prompt
            "timestamp": ts.isoformat() if ts else None
        }
        for (role, msg, ts) in rows
    ]
    return HistoryResponse(session_id=session_id, messages=messages)

@app.get("/prompts/root", response_model=List[Prompt])
def get_root_prompts():
    """Fetch top-level prompts with updated greeting message."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, prompt_text, response_text, display_order
        FROM prompts
        WHERE parent_id IS NULL
        ORDER BY display_order ASC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    log_event("Root prompts fetched", prompt_ids=[row[0] for row in rows])

    # Define the default message
    default_message = "👋 Hey there! Glad you stopped by.\nI can point you in the right direction — just tell me what’s on your mind."

    # Build the response
    prompts = []
    for row in rows:
        prompt_text = default_message if row[1] == "How can I assist you today?" else row[1]
        prompts.append({
            "id": row[0],
            "prompt_text": prompt_text,
            "response_text": row[2],
            "display_order": row[3],
            "type": "root"  # Root prompts always have type "root"
        })

    return prompts

@app.get("/prompts/{prompt_id}/children", response_model=List[Prompt])
def get_child_prompts(prompt_id: int):
    """Always generate follow-up questions dynamically for a given prompt ID."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT prompt_text FROM prompts WHERE id = %s", (prompt_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prompt not found")

    prompt_text = row[0]
    followup_prompt = f"Based on: '{prompt_text}', suggest 5 follow-up questions for deeper discussion."
    followups = optimized_chatbot.generate_followups(followup_prompt, num=5)
    followup_ids = save_followups(prompt_id, followups)

    rows = [
        (fid, ftext, None, idx + 1)
        for idx, (fid, ftext) in enumerate(zip(followup_ids, followups))
    ]

    log_event("Follow-up questions dynamically generated", parent_prompt_id=prompt_id, follow_up_ids=[row[0] for row in rows])
    return [
        {
            "id": row[0],
            "prompt_text": row[1].strip(),  # Ensure no serial numbers or extra formatting
            "response_text": None,  # Exclude response_text
            "display_order": row[3],
            "type": _prompt_type(row[0], prompt_id)  # Dynamically determine type
        }
        for row in rows
    ]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "streaming": "enabled", "chatbot": "optimized"}
