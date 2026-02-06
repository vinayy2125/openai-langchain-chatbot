# DITSTEK Chatbot Backend

A lead-generation chatbot with **UC-1 Conversation Orchestrator** — a deterministic state machine controlling conversation flow while an LLM generates all user-facing responses.

---

## 🎯 Core Architecture

### UC-1 Conversation Flow
```
ENTRY → CAPABILITY_SELECTION → CONTEXT_QUESTION → NAME_CAPTURE 
      → AI_SYNTHESIS → EXPLORATION_LAYER → CONSULTATIVE_ALTERNATIVES 
      → RECOMMENDATION → EMAIL_CAPTURE → EXIT
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

### 6 Capability Buckets
| ID | Trigger | Goal |
|----|---------|------|
| UC1-A | Product development & engineering | Orient product |
| UC1-B | Application Modernization | Modernize system |
| UC1-C | Staff Augmentation & Talent | Build team |
| UC1-D | AI/ML & Automation | Add intelligence |
| UC1-E | Cloud, DevOps & Scalability | Prepare for scale |
| UC1-F | Not sure yet / need guidance | Reduce confusion |

### Key Principles
- **LLM-First**: Fine-tuned model generates 100% of user-visible text
- **Text-Blind Orchestrator**: Controls FLOW, not LANGUAGE
- **Policy Engine**: Validates user intent and enforces guardrails

📖 See [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md) for detailed technical documentation.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **API Server** | FastAPI (SSE streaming) |
| **Database** | PostgreSQL 16 |
| **Cache** | Redis Stack (JSON + Vector Search) |
| **Embeddings** | ChromaDB + Sentence Transformers |
| **LLM** | OpenAI GPT / Groq (fine-tuned) |
| **Orchestration** | LangChain + LangGraph |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 16
- Redis Stack
- OpenAI or Groq API Key

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd Chatbot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Running the App

```bash
# Start the API server (development)
python run.py
# Server runs at http://127.0.0.1:8000

# Run the LLM Instructions Utility (Streamlit)
python run_utility.py
```

---

## ⚙️ Environment Variables

### Required
```env
# Database
DB_NAME=chatbot
DB_USER=postgres
DB_PASSWORD=<your-password>
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<your-password>

# LLM API (choose one)
GROQ_API_KEY=<your-groq-key>
# or OPENAI_API_KEY=<your-openai-key>
```

### Optional
```env
# ChromaDB
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=website_embeddings
CHROMA_SERVER_URL=              # Use HttpClient if set

# Email (SMTP)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=<user>
SMTP_PASSWORD=<password>

# Web Scraper
SCRAPER_BASE_URLS=https://example.com
SCRAPER_MAX_DEPTH=3
SCRAPER_MAX_PAGES=100
```

See [.env.example](./.env.example) for complete list.

**The chatbot uses `app/utils/prompts.py` for ALL response generation.**

## 📡 API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/send-stream` | Send message (SSE streaming) |
| `GET` | `/api/chat/{session_id}/messages` | Get chat history |

### User
| Method | Endpoint | Description |
|--------|----------|-------------|
| `PATCH` | `/api/user/{session_id}` | Update user details |
| `GET` | `/api/user/{session_id}` | Get user info |

### Prompts
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/prompts/root` | Get initial greeting prompts |

### SSE Response Schema
```json
{"status": "processing", "message": "Preparing response..."}
{"status": "chunk", "chunk": "<markdown text>"}
{"status": "meta", "chunk": {"user_details_known": true, "uc1_state": "exploration_layer", "uc1_options": ["Option 1", "Option 2"]}}
{"status": "form_trigger", "chunk": ""}
{"status": "end_chat", "chunk": "<closure message>"}
```

1. Edit `app/utils/prompts.py`
2. Modify the `final_response_prompt()` function
3. Update `PROMPT_VERSION` to track changes
4. Restart the server

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# UC-1 E2E flow tests
pytest tests/test_uc1_e2e.py -v

# ACK logic tests
pytest tests/test_ack_logic_e2e.py -v

# Input classifier tests
pytest tests/test_input_classifier.py -v

# State machine tests
pytest tests/test_state_machine.py -v
```

### Test Structure
```
tests/
├── test_uc1_e2e.py          # Full UC-1 flow integration tests
├── test_ack_logic_e2e.py    # ACK/confirmation handling
├── test_input_classifier.py  # Input classification (ACK, negation, questions)
├── test_state_machine.py     # State transition validation
├── test_slot_manager.py      # Slot filling logic
├── test_policy_*.py          # Policy engine tests
└── test_output_sanitizer.py  # Response sanitization
```

---

## 📁 Project Structure

```
Chatbot/
├── app/
│   ├── main.py              # FastAPI application
│   ├── orchestrator/        # UC-1 state machine & flow control
│   │   ├── orchestrator.py  # Central controller
│   │   ├── state_machine.py # State definitions & transitions
│   │   ├── llm_adapter.py   # LLM response generation
│   │   ├── policy_engine.py # Intent validation
│   │   └── uc1_config.yaml  # Capability buckets & CTAs
│   ├── api/                 # FastAPI routers
│   ├── db/                  # Database models & helpers
│   ├── utils/               # Prompts, formatters, helpers
│   └── agents/              # LangGraph agents
├── tests/                   # Pytest test suites
├── scripts/                 # Utility scripts
├── fine_tuning_data/        # Training data for LLM
└── scraped_data/            # Web scraper output
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Redis connection failed | Ensure Redis Stack is running: `docker-compose up redis` |
| OpenAI quota/rate limit | Check API key or upgrade plan |
| ChromaDB errors | Verify `CHROMA_DB_PATH` exists and is writable |

---

## 📜 License

MIT
