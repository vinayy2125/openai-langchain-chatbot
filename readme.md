# OpenAI + DALL·E 3 + LangChain PDF Chat Assistant

## 🧠 Overview

This is a powerful chatbot web app built using **Streamlit**, **OpenAI APIs (Chat + DALL·E 3)**, and **LangChain** with support for:

- Interactive AI chat
- Image generation from text
- PDF/document search and analysis
- Multi-mode agent tool orchestration
- Semantic vector search using **FAISS**

---

## Workflow Overview

```mermaid
graph TD
    A[User Input (Streamlit UI)] --> B{Mode Selection}
    B -- OPENAI Chat --> C[GPT-3.5/4 Chat Model]
    B -- Agent with Tools --> D[LangChain Agent (DALL·E 3, Web Search, DateTime)]
    B -- Efficient Mode --> E[GPT Direct Completion]
    B -- PDF Chat --> F[Upload + Extract + Vector Search + GPT Reasoning]
    C --> G[Show Response]
    D --> G
    E --> G
    F --> G
    G --> H[Update Session State & DB]
```

---

## 💡 Features

- 🔮 **OPENAI Chat** — Full conversational assistant
- 🛠 **Agent with Tools** — Image generation, tools, dynamic routing
- ⚡ **Efficient Mode** — Lightweight direct GPT chat
- 📄 **PDF Chat Assistant**:
  - Upload PDF, DOCX, or TXT files
  - Extract, chunk, and embed text
  - Perform keyword and semantic search (FAISS)
  - Chat with the document like ChatGPT
  - View document content in expandable preview
- 📷 **Image Generation** via **DALL·E 3**
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

## 🚀 Running the App

From the project root, run:

```bash
streamlit run app.py
```

---

## 🗄️ Image Generation

- Use natural prompts like:
  - "Generate an image of a futuristic city"
  - "Draw a lion standing on a mountain"
- Agent identifies intent and routes to **DALL·E 3**

---

## 📄 PDF Chat Mode

- Upload documents (PDF, DOCX, TXT)
- Extracted text is token-limited or chunked
- Uses **FAISS + OpenAIEmbeddings** for vector search
- Responds to your questions contextually

---

## 🧹 Modules

- `app.py` – Main app logic + routing
- `modules/pdf_chat_handler.py` – PDF upload + chat logic
- `modules/tools.py` – Agent tools (image generation, datetime, etc.)
- `modules/vector_utils.py` – Vector search utilities (FAISS)
- `modules/db_models.py` – SQLAlchemy models

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

