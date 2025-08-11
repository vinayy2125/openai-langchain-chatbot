# streamlit_app/app.py
import streamlit as st
from datetime import datetime
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app.api_calls import start_new_chat, continue_chat
from streamlit_app.session_store import load_sessions, save_session_metadata, SESSION_FILE
from backend.chat_logic import get_answer_from_context  # ✅ Import patched function

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
        formatted_timestamp = datetime.fromisoformat(session['timestamp']).strftime("%b %d, %Y %I:%M %p") if session['timestamp'] != "NA" else "NA"
        if st.button(f"🗂 {truncated_title} ({formatted_timestamp})", key=session["session_id"]):
            with st.spinner("Loading session..."):
                st.session_state.session_id = session["session_id"]
                try:
                    st.session_state.chat_history = [
                        ("user", entry["query"], "") if i % 2 == 0 else ("bot", entry["answer"], "")
                        for i, entry in enumerate(session["history"])
                    ]
                except KeyError:
                    st.session_state.chat_history = []
                    st.error("⚠️ No history found for this session.")
                st.rerun()
    with col2:
        delete_button = st.button("🗑️", key=f"delete_{session['session_id']}")
        if delete_button:
            st.session_state.chat_sessions = [s for s in st.session_state.chat_sessions if s["session_id"] != session["session_id"]]
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if session["session_id"] in data:
                del data[session["session_id"]]
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            st.rerun()

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
    st.session_state.chat_history.append(("user", user_input, now))

    try:
        # Call our new context+history-aware function
        plain_history = [(r, m) for r, m, _ in st.session_state.chat_history if r in ("user", "bot")]
        bot_msg, _ = get_answer_from_context(user_input, plain_history)

        st.session_state.chat_history.append(("bot", bot_msg, now))

        # Save session data
        if st.session_state.session_id:
            save_session_metadata(
                st.session_state.session_id,
                st.session_state.chat_sessions[-1]['title'],
                user_input, bot_msg
            )
        else:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
            save_session_metadata(
                st.session_state.session_id,
                user_input[:30], user_input, bot_msg
            )
            st.session_state.chat_sessions = load_sessions()

    except Exception as e:
        st.session_state.chat_history.append(("bot", f"⚠️ Unexpected Error: {str(e)}", now))

    st.rerun()
