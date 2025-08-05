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


# --- Prompt Template ---
template = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""
You are OPENAI, a helpful AI assistant.
{context}
User: {user_input}
OPENAI:
""",
)

WINDOW_SIZE = 1


def get_context(user_input=None):
    history = st.session_state.chat_history[-WINDOW_SIZE:]
    context = "\n".join(
        [
            f"You: {msg}" if speaker == "You" else f"OPENAI: {msg}"
            for speaker, msg in history
        ]
    )
    # Only search for relevant chunks if explicitly requested to save API calls
    relevant_chunks = []
    if user_input and ("search" in user_input.lower() or "find" in user_input.lower()):
        relevant_chunks = search_similar_chunks(
            user_input, top_k=1
        )  # Reduced from 2 to 1
    if relevant_chunks:
        return (
            f"[Relevant file context]:\n"
            + "\n---\n".join(relevant_chunks)
            + "\n---\n"
            + context
        )
    return context

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
st.session_state.chat_history = [(msg.sender, msg.content) for msg in messages]

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
tools = [
    Tool(
        name="Web Search",
        func=web_search,
        description="Useful for answering questions about current events or general knowledge",
    ),
    Tool(
        name="Calculator", func=calculator, description="Useful for math calculations."
    ),
    Tool(
        name="Current DateTime",
        func=current_datetime,
        description="Returns the current date and time. Useful for answering questions about today's date or time.",
    ),
    Tool(
        name="Image Generation",
        func=image_generation,
        return_direct=True,
        description="Use this tool to generate an image from a text prompt. It returns a direct image URL...",
    ),
]
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False,
    # memory=memory,
    handle_parsing_errors=True,
)

# --- UI and Main Logic ---
mode = st.radio("Choose mode:", ["OPENAI Chat",
                "Agent with Tools", "Efficient Mode"])
user_input = st.chat_input("Type your message...")

# --- Conversational memory for OPENAI Chat mode ---
if "openai_chat_history" not in st.session_state:
    past_msgs = db.query(ChatMessage).filter_by(
        session_id=session_id).order_by(ChatMessage.timestamp).all()
    st.session_state.openai_chat_history = [
        {"role": "user" if m.sender == "user" else "assistant", "content": m.content} for m in past_msgs
    ]

# --- Efficient mode memory ---
if "efficient_chat_history" not in st.session_state:
    st.session_state.efficient_chat_history = []

# --- Main chat area (centered) ---
with st.container():
    st.markdown(
        """
        <div style='background:#f0f4ff; border-radius:12px; padding:24px 12px 12px 12px; margin-bottom:24px; box-shadow:0 2px 8px #e0e7ef;'>
            <h2 style='text-align:center; margin-bottom:18px;'>🤖 Intelligent Chat Bot</h2>
            <div style='text-align:center; color:#888; font-size:1em; margin-bottom:10px;'>Active Chat</div>
    """,
        unsafe_allow_html=True,
    )
    # Display chat history in the center as chat bubbles
    if mode == "OPENAI Chat":
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
    elif mode == "Agent with Tools":
        pass
    elif mode == "Efficient Mode":
        for user_msg, ai_msg in st.session_state.efficient_chat_history:
            st.markdown(
                f"<div style='text-align:right; background:#e6f7ff; padding:8px; border-radius:8px; margin:4px 0;'>🧑 {user_msg}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:left; background:#f6f6f6; padding:8px; border-radius:8px; margin:4px 0;'>🤖 {ai_msg}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# --- Input and response handling ---
if user_input:
    if mode == "OPENAI Chat":
        # st.session_state.openai_chat_history.append(
        #     {"role": "user", "content": user_input}
        # )
       # Convert OpenAI-style history to Gemini-compatible format

        def convert_history_for_gemini(chat_history):
            converted = []
            for msg in chat_history:
                if msg["role"] == "user":
                    converted.append({"role": "user", "parts": [msg["content"]]})
                else:
                    # Gemini expects model's replies under 'model'
                    converted.append({"role": "model", "parts": [msg["content"]]})
            return converted

        try:
            with st.spinner("OPENAI is thinking..."):
                # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

                # Extract response text
                assistant_reply = response.text
                st.session_state.openai_chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )
                st.markdown(
                    f"<div style='text-align:left; background:#f6f6f6; padding:8px; border-radius:8px; margin:4px 0;'>🤖 {assistant_reply}</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"OPENAI error: {e}")
    elif mode == "Agent with Tools":
        try:
            with st.spinner("Agent is thinking..."):
                response = agent.run(user_input)

                #  Always print raw agent output
                print("[DEBUG] Raw agent response:", response)

                import re

                import os

                if isinstance(response, str) and response.endswith(".png") and os.path.exists(response):
                    st.image(response, caption="Generated Image")
                    with open(response, "rb") as file:
                        st.download_button(
                            "Download Image", data=file, file_name=os.path.basename(response))
                else:
                    st.markdown(response if isinstance(
                        response, str) else str(response))

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.error(
                    "You have exceeded your API quota. Please wait or upgrade your plan."
                )
            elif "iteration" in str(e).lower() or "time" in str(e).lower():
                st.error(
                    "Agent stopped due to complexity. Try breaking down your request into smaller parts or use 'Efficient Mode' for simpler queries."
                )
            else:
                st.error(f"Agent error: {e}")

    elif mode == "Efficient Mode":
        try:
            with st.spinner("OPENAI is thinking..."):
                # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                # response = client.chat.completions.create(
                #     model="gpt-4", messages=[{"role": "user", "content": user_input}],
                # )
                # ai_reply = response.choices[0].message.content
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

                # Start the Gemini chat model
                # or "gemini-1.5-pro" if whitelisted
                model = genai.GenerativeModel("gemini-2.5-pro")
                # chat = model.start_chat(
                #     history=st.session_state.openai_chat_history)
                
                gemini_history = convert_history_for_gemini(st.session_state.openai_chat_history)
                chat = model.start_chat(history=gemini_history)


                # Send user message
                response = chat.send_message(user_input)

                # Extract response text
                ai_reply = response.text
                st.session_state.efficient_chat_history.append(
                    (user_input, ai_reply))
                st.markdown(
                    f"<div style='text-align:left; background:#f6f6f6; padding:8px; border-radius:8px; margin:4px 0;'>🤖 {ai_reply}</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Efficient Mode error: {e}")

# --- Chat History Display with Expander ---
with st.container():
    with st.expander("📚 Show Previous Chat History", expanded=False):
        if mode == "OPENAI Chat":
            for msg in st.session_state.openai_chat_history:
                who = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>{who}:</b> {msg['content']}</div>",
                    unsafe_allow_html=True,
                )
        elif mode == "Efficient Mode":
            for user_msg, ai_msg in st.session_state.efficient_chat_history:
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>🧑 You:</b> {user_msg}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>🤖 Assistant:</b> {ai_msg}</div>",
                    unsafe_allow_html=True,
                )
        elif mode == "Agent with Tools":
            for entry in st.session_state.get("agent_chat_history", []):
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>🧑 You:</b> {entry['user']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>🤖 Assistant:</b> {entry['bot']}</div>",
                    unsafe_allow_html=True,
                )
        elif mode == "PDF Chat":
            for entry in st.session_state.get("pdf_chat_history", []):
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>🧑 You:</b> {entry['user']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.9em; margin-bottom:4px;'><b>🤖 Assistant:</b> {entry['bot']}</div>",
                    unsafe_allow_html=True,
                )

# 📄 File Upload and Interactive PDF Chat Module
handle_pdf_chat(st, db, session_id, user_input)
