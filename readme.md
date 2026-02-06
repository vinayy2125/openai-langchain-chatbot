# Backend Chatbot Application

This is a backend chatbot application built using **OpenAI APIs (Chat + DALL·E 3)** and **LangChain** with support for:

- Interactive AI chat on a particular knowledge base
- Semantic vector search using **FAISS**

---

## Workflow Overview

The backend processes user queries, retrieves relevant context, and generates responses using OpenAI's GPT models. It supports:

- **OPENAI Chat**: Full conversational assistant

---

## 🚀 Running the App

### 1. Running the Backend
From the project root, run:
```bash
python run.py
```
This starts the Backend API on `http://127.0.0.1:8000`.

### 2. Running the Utility Application
To run the LLM Instructions Utility (Streamlit application), use:
```powershell
./run_utility.ps1
```
Or via Python:
```bash
python run_utility.py
```
This will launch the utility app in your browser.

---

## 💡 Features

- 🔮 **OPENAI Chat** — Full conversational assistant
- 🧠 Memory + chat history logging
- 💃 Persistent DB using SQLAlchemy

---

## ⚖️ Requirements

- Python 3.8+
- OpenAI API Key

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔐 Environment Setup (`.env`)

```env
OPENAI_API_KEY=your_openai_api_key
```

```

---

## 📝 Prompt Source & Configuration

### Where Do Prompt Instructions Come From?

**The chatbot uses `app/utils/prompts.py` for ALL response generation.**

- ✅ **Primary Source**: `final_response_prompt()` function in `app/utils/prompts.py`
- ❌ **NOT Used**: Redis `chat_prompt_json` chunks (only for fallback/archival)
- 📊 **Current Version**: Check `PROMPT_VERSION` in `prompts.py`

### How to Update Chatbot Behavior:

1. Edit `app/utils/prompts.py`
2. Modify the `final_response_prompt()` function
3. Update `PROMPT_VERSION` to track changes
4. Restart the server

### Verification:

Check logs for these messages during chat interactions:
```
[PROMPT_SOURCE] ✓ Generating prompt instructions from prompts.py (Version: X.X.X)
[PROMPT_SOURCE] ✗ NOT loading from Redis chat_prompt_json chunks
```

📖 **For detailed documentation**, see: [`PROMPT_SOURCE_CLARITY.md`](./PROMPT_SOURCE_CLARITY.md)

---

## 🐛 Troubleshooting

- **No module named **``\
  → Run `pip install langchain openai`

- **Streamlit duplicate widget ID**\
  → Use `key=` parameter in widgets like `st.file_uploader`

- **OpenAI quota/rate limit**\
  → Upgrade your plan or reduce usage temporarily

---

## 📜 License

MIT

