from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from backend.chat_logic import get_answer_from_context
from backend.history import chat_sessions
import json
from datetime import datetime

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
    
    
# --- API Routes ---
@app.post("/chat/new", response_model=ChatResponse)
def new_chat(req: ChatRequest):
    session_id = str(uuid4())
    answer, matched = get_answer_from_context(req.query, req.source_url)
    chat_sessions[session_id] = [(req.query, answer)]

    # Save the new session to chat_history.json
    try:
        with open("c:\\Users\\ditsd\\Project\\Chatbot\\session_data\\chat_history.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    data[session_id] = {
        "title": req.query[:30],
        "timestamp": datetime.now().isoformat(),  # Add timestamp here
        "history": [{"query": req.query, "answer": answer}]
    }

    with open("c:\\Users\\ditsd\\Project\\Chatbot\\session_data\\chat_history.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        source=req.source_url,
        matched=matched
    )


@app.post("/chat/continue", response_model=ChatResponse)
def continue_chat(req:ChatRequest):
    if req.session_id not in chat_sessions:
        print(f"DEBUG: Available session IDs: {list(chat_sessions.keys())}")  # Log all session IDs
        print(f"DEBUG: Requested session ID: {req.session_id}")  # Log the requested session ID
        raise HTTPException(status_code=404, detail="Session not found")
    answer, matched = get_answer_from_context(req.query)
    chat_sessions[req.session_id].append((req.query, answer))

    # Update the session in chat_history.json
    try:
        with open("c:\\Users\\ditsd\\Project\\Chatbot\\session_data\\chat_history.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    if req.session_id in data:
        data[req.session_id]["history"].append({"query": req.query, "answer": answer})

    with open("c:\\Users\\ditsd\\Project\\Chatbot\\session_data\\chat_history.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        matched=matched
    )
    
    
@app.get("/chat/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return HistoryResponse(
        session_id=session_id,
        messages=[{"query": q, "answer": a} for q, a in chat_sessions[session_id]]
    )
    
    
@app.get("/ping")
def health_check():
    return {"status": "ok"}
