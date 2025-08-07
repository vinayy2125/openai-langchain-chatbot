from langchain_google_genai import ChatGoogleGenerativeAI
from modules.pdf_chat_handler import handle_pdf_chat
from langchain_openai import ChatOpenAI
from modules.tools import web_search, calculator, current_datetime, image_generation
from modules.vector_utils import search_similar_chunks
from modules.file_utils import extract_text_from_file, chunk_text
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
import os
import openai
import streamlit as st
import uuid

from modules.db_models import SessionLocal, ChatSession, ChatMessage, UploadedFile
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
genai.configure(api_key="GOOGLE_API_KEY")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# if not openai.api_key:
#     raise ValueError(
#         "OPENAI_API_KEY not found in .env file. Please add it to your .env file."
#     )


try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    st.warning("PyPDF2 is not installed. PDF extraction will not work.")

try:
    import docx
except ImportError:
    docx = None
    st.warning("python-docx is not installed. DOCX extraction will not work.")


prompt_template_openai = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""
You are OPENAI, a helpful AI assistant.
{context}
User: {user_input}
OPENAI:
""",
)

prompt_template_gemini = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""
You are Gemini, a helpful assistant powered by Google.
{context}
User: {user_input}
Gemini:
""",
)

def convert_history_for_gemini(chat_history):
    converted = []
    for msg in chat_history:
       if msg["role"] == "user":
           converted.append({"role": "user", "parts": [msg["content"]]})
       else:
           converted.append({"role": "model", "parts": [msg["content"]]})
    return converted

# --- DB Session Setup ---
db = SessionLocal()
if "session_id" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    db_session = ChatSession(id=session_id)
    db.add(db_session)
    db.commit()
else:
    session_id = st.session_state.session_id

# --- Load chat history from DB ---
messages = (
    db.query(ChatMessage)
    .filter_by(session_id=session_id)
    .order_by(ChatMessage.timestamp)
    .all()
)
#st.session_state.chat_history = [(msg.sender, msg.content) for msg in messages]

# --- Initialize OPENAI Chat Model ---
# llm = ChatOpenAI(
#     model_name="gpt-4",  # or "gpt-4" if you have access
#     temperature=0,
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  # or "gemini-1.5-pro" if you're in preview
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agent and Tools Setup ---
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# --- UI and Main Logic ---
user_input = st.chat_input("Type your message...")

# --- Conversational memory for OPENAI Chat mode ---
if "openai_chat_history" not in st.session_state:
    past_msgs = db.query(ChatMessage).filter_by(
        session_id=session_id).order_by(ChatMessage.timestamp).all()
    st.session_state.openai_chat_history = [
        {"role": "user" if m.sender == "user" else "assistant", "content": m.content} for m in past_msgs
    ]

# --- Main chat area (centered) ---
with st.container():
    st.markdown(
        """
        <div style='background:#f0f4ff; border-radius:12px; padding:24px 12px 12px 12px; margin-bottom:24px; box-shadow:0 2px 8px #e0e7ef;'>
            <h2 style='text-align:center; margin-bottom:18px;'>🤖 Chat Bot</h2>
            <div style='text-align:center; color:#888; font-size:1em; margin-bottom:10px;'>Active Chat</div>
    """,
        unsafe_allow_html=True,
    )
    # Display chat history in the center as chat bubbles
    for msg in st.session_state.openai_chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='text-align:right; background:#e6f7ff; padding:8px; border-radius:8px; margin:4px 0;'>🧑 {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; background:#f6f6f6; padding:8px; border-radius:8px; margin:4px 0;'>🤖 {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)


# --- Input and response handling ---
if user_input:
        st.markdown(
        f"<div style='text-align:right; background:#e6f7ff; padding:8px; border-radius:8px; margin:4px 0;'>🧑 {user_input}</div>",
        unsafe_allow_html=True,
        )
        # st.session_state.openai_chat_history.append(
        #     {"role": "user", "content": user_input}
        # )
       # Convert OpenAI-style history to Gemini-compatible format
        try:
            with st.spinner("OPENAI is thinking..."):
                # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                # full_prompt = prompt_template_openai.format(context=context, user_input=user_input)
                # response = client.chat.completions.create(
                #     model="gpt-4", messages=st.session_state.openai_chat_history
                # )
                # assistant_reply = response.choices[0].message.content
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

                # Start the Gemini chat model
                # or "gemini-1.5-pro" if whitelisted
                model = genai.GenerativeModel("gemini-2.5-pro")
                # chat = model.start_chat(
                #     history=st.session_state.openai_chat_history)
                st.session_state.openai_chat_history.append(
                    {"role": "user", "content": user_input}
            )
                gemini_history = convert_history_for_gemini(st.session_state.openai_chat_history)
                chat = model.start_chat(history=gemini_history)


                # Send user message
                response = chat.send_message(user_input)

                # Defensive check for valid response
                if response.candidates and response.candidates[0].content.parts:
                    assistant_reply = response.text
                else:
                    assistant_reply = "⚠️ Sorry, I didn’t receive a valid response. Try rephrasing."

                # Append and display
                st.session_state.openai_chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )
                db.add_all([
                    ChatMessage(session_id=session_id, sender="user", content=user_input),
                    ChatMessage(session_id=session_id, sender="assistant", content=assistant_reply)
                ])
                db.commit()

            st.markdown(
                f"<div style='text-align:left; background:#f6f6f6; padding:8px; border-radius:8px; margin:4px 0;'>🤖 {assistant_reply}</div>",
                unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"OPENAI error: {e}")

# --- Chat History Display with Expander ---
with st.container():
    with st.expander("📚 Show Previous Chat History", expanded=False):
            for msg in st.session_state.openai_chat_history:
                who = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>{who}:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
                
# 📄 File Upload and Interactive PDF Chat Module
handle_pdf_chat(st, db, session_id, user_input)
