import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from uuid import UUID
import logging
import psycopg2
from fastapi import HTTPException

from .models import (
    SessionState, FollowUp, FollowUpType,
    MessageCreate, Message, SessionCreate, Session
)

logger = logging.getLogger(__name__)

# Database connection helper
def get_db_conn():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        options="-c client_encoding=UTF8"
    )

async def initialize_session_with_prompt(session_id: UUID, prompt_id: UUID) -> Dict[str, Any]:
    """Initialize a new session with the selected prompt."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        # Get prompt details
        cursor.execute("""
            SELECT prompt_text, response_text, type
            FROM prompts
            WHERE id = %s
        """, (prompt_id,))
        prompt = cursor.fetchone()
        
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        prompt_text, response_text, prompt_type = prompt
        
        # Update session with prompt
        cursor.execute("""
            UPDATE sessions
            SET current_prompt_id = %s,
                last_interaction_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
            RETURNING id
        """, (prompt_id, session_id))
        
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Initialize session state
        return {
            "prompt_id": str(prompt_id),
            "prompt_text": prompt_text,
            "response_text": response_text,
            "prompt_type": prompt_type
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
        cursor.execute("""
            UPDATE sessions
            SET last_interaction_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP,
                follow_up_count = CASE 
                    WHEN %s IS NOT NULL THEN follow_up_count + 1 
                    ELSE follow_up_count 
                END
            WHERE session_id = %s
            RETURNING id
        """, (message_data.follow_up_to, message_data.session_id))

        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Session not found")

        # Insert the message
        cursor.execute("""
            INSERT INTO messages (
                session_id, role, content, reply_to, follow_up_to, 
                follow_up_depth, metadata, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING *
        """, (
            message_data.session_id,
            message_data.role,
            message_data.content,
            message_data.reply_to,
            message_data.follow_up_to,
            message_data.follow_up_depth,
            json.dumps(message_data.metadata)
        ))
        
        message_row = cursor.fetchone()
        conn.commit()

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
            updated_at=message_row[9]
        )
    finally:
        cursor.close()
        conn.close()

async def get_messages_for_session(session_id: UUID) -> List[Message]:
    """Retrieve all messages for a given session with their relationships and metadata."""
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
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
        """, (session_id,))
        
        messages = []
        for row in cursor.fetchall():
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
                updated_at=row[9]
            )
            messages.append(message)
            
        return messages
    finally:
        cursor.close()
        conn.close()

async def generate_follow_up(prompt_text: str, conversation_history: list, llm) -> FollowUp:
    """Generate a follow-up question using LLM."""
    messages = [
        {
            "role": "system",
            "content": f"""You are an AI assistant helping to gather requirements through follow-up questions.
            Original Prompt: {prompt_text}
            
            Generate a follow-up question that:
            1. Is relevant to the original prompt
            2. Builds on previous responses
            3. Helps gather complete requirements
            
            Return a JSON object with:
            - type: "yes_no", "nested", "expansion", or "clarification"
            - question: The follow-up question
            - context: Why you're asking this
            - options: Array of choices (for nested type only)
            """
        }
    ]
    
    follow_up_response = ""
    async for chunk in llm.stream(messages):
        if chunk:
            follow_up_response += chunk
    
    try:
        follow_up_data = json.loads(follow_up_response)
        return FollowUp(**follow_up_data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing follow-up response: {str(e)}")
        return FollowUp(
            type=FollowUpType.EXPANSION,
            question="Could you provide more details about your requirements?",
            context="Ensuring we understand your needs correctly",
        )
