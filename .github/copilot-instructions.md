# Copilot Instructions for openai-langchain-chatbot

## Project Architecture
- **Backend**: Python FastAPI app (see `app/`), orchestrates chat, context retrieval, and response generation.
- **AI/LLM**: Uses OpenAI APIs (Chat, DALL·E 3) via LangChain (`app/utils/llm_client.py`, `app/utils/llm_utils.py`).
- **Vector Search**: Semantic search with FAISS, Redis integration (`app/db/redis_vector_helper.py`, `app/api/redis_endpoint.py`).
- **Persistence**: SQLAlchemy for DB, Redis for context/memory.
- **Ingestion**: Scripts for scraping and populating Redis (`app/ingestion/scrape_to_redis.py`).

## Key Workflows
- **Run backend**: `uvicorn backend.api:app --reload` (see README)
- **Install deps**: `pip install -r requirements.txt`
- **Set env**: `.env` with `OPENAI_API_KEY`
- **Ingest data**: Use scripts in `scripts/` or `app/ingestion/`

## Patterns & Conventions
- **Routers**: API endpoints in `app/api/router.py`, dependencies in `app/api/deps.py`.
- **Helpers**: Shared logic in `app/api/helpers.py`, `app/utils/`.
- **Config**: Centralized in `app/config.py`.
- **Logging**: Use `app/logger.py` for all logs.
- **Testing**: Place tests in `tests/`.
- **Migrations**: SQL in `migrations/`, run via `scripts/run_migration.py`.
- **Embeddings**: Models and blobs in `embeddings/`.

## Integration Points
- **OpenAI**: API key required, see `.env`.
- **Redis**: Used for vector search and chat memory.
- **FAISS**: For semantic search, see `app/db/redis_vector_helper.py`.

## Examples
- Add new API: Create router in `app/api/`, register in `main.py`.
- Add ingestion: Script in `app/ingestion/`, call from CLI or workflow.
- Add embedding model: Place in `embeddings/`, update vector helper.

## Non-Obvious Details
- **Semantic search**: Uses both Redis and FAISS, see helper classes for details.
- **Chat memory**: Persisted in Redis, not just in-process.
- **Migrations**: Use provided SQL scripts, not ORM autogeneration.

## References
- See `README.md` for setup, troubleshooting, and requirements.
- See `app/` for main backend logic and API structure.
- See `scripts/` for DB and ingestion utilities.
