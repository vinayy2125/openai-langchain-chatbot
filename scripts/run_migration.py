import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.db.base import get_db_conn

MIGRATIONS_DIR = "migrations"

def run_migrations():
    conn = get_db_conn()
    cursor = conn.cursor()

    # Ensure migration history exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS migration_history (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Already applied migrations
    cursor.execute("SELECT filename FROM migration_history")
    applied = {row[0] for row in cursor.fetchall()}

    # Always enforce correct order
    ordered_files = [
        "001_create_users_table.sql",
        "002_create_sessions_table.sql",
        "003_create_messages_table.sql",
    ]

    # Add any other files found in dir (e.g. 004, 005…)
    for f in sorted(os.listdir(MIGRATIONS_DIR)):
        if f.endswith(".sql") and f not in ordered_files:
            ordered_files.append(f)

    for filename in ordered_files:
        if filename not in applied:
            path = os.path.join(MIGRATIONS_DIR, filename)
            if not os.path.exists(path):
                print(f"⚠️ Skipping missing migration: {filename}")
                continue

            print(f"🚀 Running migration: {filename}")
            with open(path, "r") as f:
                sql = f.read()
                try:
                    cursor.execute(sql)
                    cursor.execute(
                        "INSERT INTO migration_history (filename) VALUES (%s)",
                        (filename,)
                    )
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"❌ Failed migration {filename}: {e}")
                    raise

    cursor.close()
    conn.close()
    print("✅ All migrations applied successfully.")

if __name__ == "__main__":
    run_migrations()
