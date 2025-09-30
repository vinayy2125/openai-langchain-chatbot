from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from datetime import datetime
import psycopg2
import uuid
from database_setup import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

router = APIRouter()

# --- Pydantic Model for User ---
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    mobile: str
    browser: str
    ip: str


# --- Helper Functions ---
def get_or_create_session(*, email: str, browser: str, ip: str, discard_previous: bool = False) -> str:
    """Return an active session_id for the user, or create one if none exists."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    # 1. Get user_id from users table
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    user_row = cursor.fetchone()
    if not user_row:
        conn.close()
        raise Exception("User not found while creating session")
    user_id = user_row[0]

    # 2. Check if a session exists
    cursor.execute(
        "SELECT session_id FROM sessions WHERE user_id = %s AND is_active = TRUE",
        (user_id,)
    )
    result = cursor.fetchone()

    if result and not discard_previous:
        session_id = result[0]  # session_id
    else:
        # Deactivate previous session if discard_previous is True
        if result:
            cursor.execute(
                "UPDATE sessions SET is_active = FALSE WHERE session_id = %s",
                (result[0],)
            )

        # Create a new session
        session_id = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO sessions (user_id, session_id, created_at, browser, ip, is_active)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            """,
            (user_id, session_id, datetime.now(), browser, ip)
        )
        conn.commit()

    cursor.close()
    conn.close()
    return session_id


def save_user_to_db(*, username: str, email: str, mobile: str, browser: str, ip: str) -> None:
    """Insert a new user if not exists."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    # Check if user already exists
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cursor.fetchone() is None:
        cursor.execute(
            """
            INSERT INTO users (username, email, mobile, browser, ip, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (username, email, mobile, browser, ip, datetime.now())
        )
        conn.commit()

    cursor.close()
    conn.close()


# --- API Route ---
@router.post("/user/register")
def register_user(user: UserCreate):
    try:
        # Save user if new
        save_user_to_db(
            username=user.username,
            email=user.email,
            mobile=user.mobile,
            browser=user.browser,
            ip=user.ip
        )

        # Always create or fetch a session
        session_id = get_or_create_session(
            email=user.email,
            browser=user.browser,
            ip=user.ip
        )

        return {
            "status": "success",
            "message": "User registered successfully",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
