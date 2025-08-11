import os
import json
from datetime import datetime

SESSION_FILE = "session_data/chat_history.json"

os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)

def load_sessions():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure data is a dictionary
            if not isinstance(data, dict):
                data = {}
            return [
                {
                    "session_id": key,
                    "title": value.get("title", key),  # Use title if available, else fallback to session_id
                    "timestamp": datetime.now().isoformat(),  # Only here!
                    "history": value.get("history", [])  # Include history in the session metadata
                } for key, value in data.items()
            ]
    return []

def save_session_metadata(session_id: str, title: str, query: str, answer: str):
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure data is a dictionary
            if not isinstance(data, dict):
                data = {}
    else:
        data = {}

    if session_id not in data:
        data[session_id] = {"title": title, "history": []}  # Save title and initialize history

    # Append the new query and answer to the session's history
    data[session_id]["history"].append({"query": query, "answer": answer})

    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_session_metadata():
    return load_sessions()
