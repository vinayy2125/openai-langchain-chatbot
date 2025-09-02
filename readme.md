# Backend Chatbot Application

This is a backend chatbot application built using **OpenAI APIs (Chat + DALL·E 3)** and **LangChain** with support for:

- Interactive AI chat on a particular knowledge base
- Semantic vector search using **FAISS**

---

## Workflow Overview

The backend processes user queries, retrieves relevant context, and generates responses using OpenAI's GPT models. It supports:

- **OPENAI Chat**: Full conversational assistant

---

## 🚀 Running the Backend

From the project root, run:

```bash
uvicorn backend.api:app --reload
```

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

