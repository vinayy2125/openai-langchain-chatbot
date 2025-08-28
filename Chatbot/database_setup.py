import psycopg2
from psycopg2 import sql

DB_NAME = "chatbot_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Database Setup ---
def setup_database():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    cursor = conn.cursor()

    # Create sessions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id UUID PRIMARY KEY,
            title TEXT,
            timestamp TIMESTAMP
        )
        """
    )

    # Create messages table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id UUID,
            role TEXT,
            message TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
        """
    )

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup_database()
    print("✅ PostgreSQL database setup complete.")
