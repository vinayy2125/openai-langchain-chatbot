import os
import psycopg2
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read env vars
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# --- Database Setup ---
def setup_database():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    # Enable UUID extension
    cursor.execute("""CREATE EXTENSION IF NOT EXISTS "uuid-ossp";""")

    # Create users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            username VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            mobile VARCHAR(20),
            browser VARCHAR(255),
            ip VARCHAR(45),
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Create sessions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT,
            browser TEXT,
            ip TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + interval '30 days'),
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )
        """
    )

    cursor.execute(
        """CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);"""
    )

    # Create messages table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
            role VARCHAR(50) NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);"""
    )

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    setup_database()
    print("✅ PostgreSQL database setup complete.")
