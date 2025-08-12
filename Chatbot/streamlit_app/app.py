# streamlit_app/app.py
import streamlit as st
from datetime import datetime
import sys
import os
import psycopg2
import uuid  # Import uuid for generating session IDs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app.api_calls import start_new_chat, continue_chat
from streamlit_app.session_store import load_sessions, save_session_metadata, load_chat_history_from_session, append_message_to_chat_history
from backend.api import save_message_to_db

from backend.chat_logic import get_answer_from_context  # ✅ Import patched function

# Database connection parameters
DB_NAME = "chatbot_db"  # Replace with your actual database name
DB_USER = "postgres"  # Replace with your actual database username
DB_PASSWORD = "postgres"  # Replace with your actual database password
DB_HOST = "localhost"  # Use localhost for local PostgreSQL server
DB_PORT = 5432  # Correct port number for PostgreSQL

# --- Page Config ---
st.set_page_config(page_title="Chatbot", layout="wide", initial_sidebar_state="expanded")

# --- Session State Init ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_sessions()

# --- Sidebar ---
st.sidebar.title("💬 Chatbot")
if st.sidebar.button("➕ New Chat"):
    with st.spinner("Starting a new chat..."):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Previous Sessions**")
for session in st.session_state.chat_sessions:
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        truncated_title = session['title'][:20] + ("..." if len(session['title']) > 20 else "")
        formatted_timestamp = datetime.fromisoformat(session['timestamp']).strftime("%b %d, %Y %I:%M %p")
        if st.button(f"🗂 {truncated_title} ({formatted_timestamp})", key=session["session_id"]):
            with st.spinner("Loading session..."):
                st.session_state.session_id = session["session_id"]
                st.session_state.chat_history = load_chat_history_from_session(session)
                st.rerun()
    with col2:
        delete_button = st.button("🗑️", key=f"delete_{session['session_id']}")
        if delete_button:
            try:
                conn = psycopg2.connect(
                    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
                )
                cursor = conn.cursor()
                # Delete messages and session from the database
                cursor.execute("DELETE FROM messages WHERE session_id = %s", (session["session_id"],))
                cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session["session_id"],))
                conn.commit()
                cursor.close()
                conn.close()

                # Reload sessions after deletion
                st.session_state.chat_sessions = load_sessions()
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Failed to delete session: {str(e)}")

# --- Main Chat Display ---
st.markdown("<h1 style='text-align: center; color: #FF5C8D;'>🤖 Chat Bot</h1>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for role, message, timestamp in st.session_state.chat_history:
        avatar = "🧑" if role == "user" else "🤖"
        st.markdown(
            f"**{avatar} {'You' if role == 'user' else 'Bot'}:**\n\n{message}\n\n"
            f"<span style='font-size: 0.75rem; color: #888;'>{timestamp}</span>",
            unsafe_allow_html=True
        )

# --- Input Box ---
user_input = st.chat_input("Send a message")
if user_input:
    now = datetime.now().strftime("%I:%M %p")
    # Append user input to chat history
    append_message_to_chat_history("user", user_input, st.session_state.chat_history)

    try:
        # Prepare plain history for your context function (ignore timestamps)
        plain_history = [(r, m) for r, m, _ in st.session_state.chat_history if r in ("user", "bot")]
        bot_msg, _ = get_answer_from_context(user_input, plain_history)

        append_message_to_chat_history("bot", bot_msg, st.session_state.chat_history)

        # Save session data
        if st.session_state.session_id:
            timestamp = datetime.now().isoformat()  # Define timestamp
            save_message_to_db(st.session_state.session_id, "user", user_input, timestamp)
            save_message_to_db(st.session_state.session_id, "bot", bot_msg, timestamp)
        else:
            st.session_state.session_id = str(uuid.uuid4())  # Generate a valid UUID for the new session
            timestamp = datetime.now().isoformat()  # Define timestamp for new session
            save_session_metadata(
                st.session_state.session_id,
                user_input[:30], user_input, bot_msg
            )
            st.session_state.chat_sessions = load_sessions()

    except Exception as e:
        append_message_to_chat_history("bot", f"⚠️ Unexpected Error: {str(e)}", st.session_state.chat_history)

    st.rerun()
