from datetime import datetime
import psycopg2
import uuid

DB_NAME = "chatbot_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def load_sessions():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, title, timestamp FROM sessions")
    sessions = cursor.fetchall()
    conn.close()
    return [
        {
            "session_id": session_id,
            "title": title,
            "timestamp": timestamp.isoformat(),
            "history": get_chat_history(session_id)
        }
        for session_id, title, timestamp in sessions
    ]

def save_session_metadata(session_id: str, title: str, query: str, answer: str):
    try:
        # Validate session_id as a UUID
        session_id = str(uuid.UUID(session_id))
    except (ValueError, TypeError):
        # Generate a new UUID if validation fails
        session_id = str(uuid.uuid4())

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    # Save session metadata
    cursor.execute(
        """
        INSERT INTO sessions (session_id, title, timestamp)
        VALUES (%s, %s, NOW())
        ON CONFLICT (session_id) DO UPDATE SET title = EXCLUDED.title, timestamp = EXCLUDED.timestamp
        """,
        (session_id, title)
    )

    # Save chat history
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, message, timestamp)
        VALUES (%s, %s, %s, NOW())
        """,
        (session_id, "user", query)
    )
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, message, timestamp)
        VALUES (%s, %s, %s, NOW())
        """,
        (session_id, "bot", answer)
    )

    conn.commit()
    cursor.close()
    conn.close()

def load_chat_history_from_session(session):
    """
    Convert session['history'] (list of dicts with 'query' & 'answer') into
    a flat list of (role, message, timestamp) tuples for Streamlit session_state.
    """
    return get_chat_history(session["session_id"])

def get_chat_history(session_id):
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
    conn.close()
    return [(role, message, timestamp.strftime("%H:%M")) for role, message, timestamp in history]

def append_message_to_chat_history(role, message, chat_history):
    """
    Append a single message tuple (role, message, timestamp) to chat_history list.
    """
    timestamp = datetime.now().strftime("%H:%M")
    chat_history.append((role, message, timestamp))
