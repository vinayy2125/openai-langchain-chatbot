# Copilot Instructions for OpenAI-LangChain Chatbot

## Project Overview
This is a FastAPI-based chatbot that uses OpenAI's GPT models and LangChain for context-aware conversations, with Redis for vector storage and PostgreSQL for chat history.

## Architecture & Data Flow

### Core Components
- `app/core/` - Core chatbot logic and services
  - `chat_logic.py` - Main conversation handling and response generation
  - `llm_client.py` - OpenAI model configuration and streaming
  - `nested_follow_up_manager.py` - Manages conversation context and follow-ups
  - `redis_context.py` - Context retrieval and vector similarity search

### API Layer
- `app/api/v1/` - REST API endpoints with standardized patterns:
  ```python
  @router.post("/user/register", response_model=UserRegisterResponse)
  async def register_user(user: UserCreate):
      # Database transaction pattern with proper cleanup
      try:
          # Transaction logic
          conn.commit()
      except Exception as e:
          conn.rollback()
          raise HTTPException(status_code=500)
      finally:
          cursor.close()
          conn.close()
  ```

### Data Layer
- `app/db/` - Database operations
  - `redis_operations.py` - Vector store operations with RediSearch
  - `base.py` - Database connection pool management

## Key Patterns

### Chat Management
- Conversation Flow:
  1. User message received via API
  2. Context retrieved from Redis vector store
  3. History managed by FollowUpManager
  4. Streaming response generated with follow-up suggestions
  5. Messages stored in PostgreSQL

### Vector Storage (Redis)
- Uses RediSearch with HNSW algorithm for similarity search
- Configuration:
  ```python
  INDEX_NAME = "dits_chat_idx"
  PREFIX = "dits_chatbot:"
  EMBED_DIM = 786  # E5-large embedding dimension
  DISTANCE = "COSINE"
  ```
- Query example:
  ```python
  q_str = f'(@user_id:[{user_id} {user_id}])=>[KNN {top_k} @embedding $vec_param AS score]'
  query_obj = Query(q_str).return_fields("text", "chat_id").dialect(2)
  ```

## Deployment

### Docker Services
1. API Service:
   - Python 3.13 base image
   - Uses uv package manager
   - Exposes port 8006
   - Auto-reloads in development

2. Redis Stack:
   - Custom redis-stack.conf for vector search
   - Requires password configuration
   - Ports: 6379 (Redis), 8001 (RedisInsight)

3. PostgreSQL:
   - Version 16
   - Persistent volume for data
   - Port 5432

### Environment Setup
```env
OPENAI_API_KEY=required
POSTGRES_USER=required
POSTGRES_PASSWORD=required
POSTGRES_DB=required
REDIS_PASSWORD=required
```

### Development Workflow
1. Start all services: `docker compose up`
2. API available at http://localhost:8006
3. RedisInsight at http://localhost:8001
4. Postgres on port 5432

## Common Operations
- New API endpoint: Extend `app/api/v1/router.py` following transaction pattern
- Chat logic changes: Update `app/core/chat_logic.py` and test streaming
- Vector operations: Use helper functions in `app/db/redis_operations.py`

## Testing & Debugging
- Run tests: `pytest tests/`
- API logs: `logs/backend.log` (debug level enabled)
- Monitor Redis: RedisInsight dashboard
- Check embeddings: `describe_index_stats()` in redis_operations.py