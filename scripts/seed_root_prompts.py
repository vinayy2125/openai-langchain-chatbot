"""Idempotent seeding script for root prompts.

Run with: python scripts/seed_root_prompts.py

This will insert the greeting, four primary prompts, and the typed hint if
they don't already exist (checks by prompt_text at top-level).
"""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so `from app...` imports work when running
# this script directly (Python sets sys.path[0] to the scripts directory).
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.db.base import get_db_conn


DEFAULT_PROMPTS = [
    {"prompt_text": "Start a Project", "response_text": "", "display_order": 1, "type": "root"},
    {"prompt_text": "Explore DITS Services", "response_text": "", "display_order": 2, "type": "root"},
    {"prompt_text": "See our Work", "response_text": "", "display_order": 3, "type": "root"},
    {"prompt_text": "Talk to our team", "response_text": "", "display_order": 4, "type": "root"},
]


def seed_root_prompts():
    conn = None
    cur = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()

        inserted = 0
        for p in DEFAULT_PROMPTS:
            # Only insert if a top-level prompt with the same text doesn't exist
            cur.execute("SELECT 1 FROM prompts WHERE prompt_text = %s AND parent_id IS NULL", (p["prompt_text"],))
            if cur.fetchone():
                continue

            cur.execute(
                """
                INSERT INTO prompts (prompt_text, response_text, display_order, type, created_at, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (p["prompt_text"], p["response_text"], p["display_order"], p["type"]),
            )
            inserted += 1

        if inserted:
            conn.commit()
        print(f"Seed complete. Inserted {inserted} new prompts.")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error seeding prompts: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    seed_root_prompts()
