import streamlit as st
from datetime import datetime
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app.api_calls import start_new_chat, continue_chat
from streamlit_app.session_store import load_sessions, save_session_metadata, SESSION_FILE

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
        # Limit the session title to the first 20 characters
        truncated_title = session['title'][:20] + ("..." if len(session['title']) > 20 else "")
        # Format the timestamp for better readability
        formatted_timestamp = datetime.fromisoformat(session['timestamp']).strftime("%b %d, %Y %I:%M %p") if session['timestamp'] != "NA" else "NA"
        if st.button(f"🗂 {truncated_title} ({formatted_timestamp})", key=session["session_id"]):
            with st.spinner("Loading session..."):
                st.session_state.session_id = session["session_id"]
                print(f"DEBUG: Selected session_id: {st.session_state.session_id}")  # Log session_id for debugging
                try:
                    st.session_state.chat_history = [
                        ("user", entry["query"], "") if i % 2 == 0 else ("bot", entry["answer"], "")
                        for i, entry in enumerate(session["history"])
                    ]
                except KeyError:
                    st.session_state.chat_history = []  # Handle missing history gracefully
                    st.error("⚠️ No history found for this session.")
                st.rerun()
    with col2:
        # Update delete button styling
        delete_button = st.button("🗑️", key=f"delete_{session['session_id']}")
        if delete_button:
            st.session_state.chat_sessions = [s for s in st.session_state.chat_sessions if s["session_id"] != session["session_id"]]
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if session["session_id"] in data:
                del data[session["session_id"]]  # Completely remove the session from the JSON file
            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            st.rerun()

# --- Main Title ---
st.markdown("""
<h1 style='text-align: center; color: #FF5C8D;'>🤖 Chat Bot</h1>
""", unsafe_allow_html=True)

# --- Chat Container ---
chat_container = st.container()
with chat_container:
    for entry in st.session_state.chat_history:
        role, message, timestamp = entry
        avatar = "🧑" if role == "user" else "🤖"
        st.markdown(f"""
        <div>
            <strong>{avatar} {'You' if role == 'user' else 'Bot'}:</strong><br>{message}
            <div style='font-size: 0.75rem; color: #888;'>{timestamp}</div>
        </div>
        <hr style='border: none; border-top: 1px solid #333;'>
        """, unsafe_allow_html=True)

# --- Input Box at Bottom ---
user_input = st.chat_input("Send a message")
if user_input:
    now = datetime.now().strftime("%I:%M %p")
    st.session_state.chat_history.append(("user", user_input, now))

    try:
        if st.session_state.session_id:
            # Include chat history in the payload for continue_chat
            context = [
                {"query": entry[1], "answer": "" if entry[0] == "user" else entry[1]} for entry in st.session_state.chat_history
            ]
            res = continue_chat(user_input, st.session_state.session_id, context)
            save_session_metadata(
                st.session_state.session_id, st.session_state.chat_sessions[-1]['title'], user_input, res.get("answer", "(No response)")
            )
        else:
            # Include chat history in the payload for start_new_chat
            context = [
                {"query": entry[1], "answer": "" if entry[0] == "user" else entry[1]} for entry in st.session_state.chat_history
            ]
            res = start_new_chat(user_input, context)
            st.session_state.session_id = res.get("session_id")
            save_session_metadata(
                st.session_state.session_id, user_input[:30], user_input, res.get("answer", "(No response)")
            )
            st.session_state.chat_sessions = load_sessions()

        bot_msg = res.get("answer", "(No response)")
        st.session_state.chat_history.append(("bot", bot_msg, now))

    except ValueError as e:
        st.session_state.chat_history.append(("bot", f"⚠️ Error: {str(e)}", now))
    except KeyError as e:
        st.session_state.chat_history.append(("bot", f"⚠️ Missing Key: {str(e)}", now))
    except json.JSONDecodeError as e:
        st.session_state.chat_history.append(("bot", f"⚠️ JSON Error: {str(e)}", now))
    except Exception as e:
        st.session_state.chat_history.append(("bot", f"⚠️ Unexpected Error: {str(e)}", now))

    st.rerun()
