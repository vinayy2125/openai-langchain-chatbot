# CHATBOT SYSTEM ARCHITECTURE
**Machine-Readable System Description**

---

## SYSTEM TOPOLOGY

### Components
```
API_SERVER: FastAPI (main.py) - HTTP/SSE endpoints on port 8000
WORKER_INFERENCE: OptimizedChatbot (chatbot_optimizer.py) - LLM response generation
WORKER_EMBEDDING: ThreadPoolExecutor (redis_context.py) - Background embedding computation
DATABASE_PRIMARY: PostgreSQL 16 - users, sessions, messages, prompts tables
CACHE_PRIMARY: Redis Stack (redis-stack:latest) - Key-value, JSON, vector search
CACHE_VECTOR: Redis RediSearch - Vector similarity (FLAT/COSINE, 768 dims)
CACHE_SORTED: Redis Sorted Sets - Prompt ordering by timestamp
LLM_CLIENT: OpenAI openai/gpt-oss-120b via ChatOpenAI - 128k token context
FRONTEND_PROTOCOL: Server-Sent Events (SSE) - Streaming responses
UTILITY_SCRIPTS: refresh_prompts_to_redis.py, seed_root_prompts.py
QUEUE_ASYNC: asyncio.create_task - Fire-and-forget DB writes
```

---

## DATA CONTRACTS

### Session Object
```python
{
  "session_id": "UUID (text)",           # Primary key
  "user_id": "UUID (text)",              # Foreign key to users.id
  "browser": "string | null",            # User agent
  "ip": "string | null",                 # IP address
  "is_active": "boolean",                # Session state
  "current_prompt_id": "UUID | null",    # Active prompt reference
  "last_interaction_at": "timestamp",    # Last activity
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

### Prompt Object
```python
{
  "id": "UUID (text)",                   # Primary key
  "prompt_text": "string",               # Display text
  "response_text": "string | null",     # Default response
  "display_order": "integer",            # Sort order
  "type": "PromptType (ROOT|FOLLOW_UP)", # Enum
  "parent_id": "UUID | null",            # Hierarchy reference
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

### Instruction Record (Redis JSON)
```python
{
  "id": "string",                        # Prompt identifier
  "prompt": "string",                    # Instruction text
  "created_at": "float (unix_ts)",       # Timestamp
  "source": "string | null",             # Origin tag
  "lang": "string | null",               # Language code
  "assistant_name": "string | null",    # AI persona name
  "assistant_instruction": "string | null" # Behavior rules
}
# Redis key: chat_prompt_json:{id}
# Index: chat_prompts_json (JSON RediSearch)
```

### Retrieval Chunk (Redis JSON)
```python
{
  "session_id": "string",                # Session identifier
  "messages": [                          # Chat history array
    {
      "role": "user | assistant",
      "content": "string",
      "timestamp": "ISO8601 | null"
    }
  ],
  "messages_text": "string",            # Concatenated text for search
  "embedding": "float32[768] | null"    # Vector representation
}
# Redis key: chat_history:{session_id}
# Index: chat_history_index (JSON RediSearch, VECTOR FLAT COSINE 768)
```

### Cache Structure (Redis)
```
HASH: chat_prompt:{id}                  # Legacy prompt storage
  - id: string
  - prompt: string
  - created_at: string (numeric)
  - source: string
  - lang: string

JSON: chat_prompt_json:{id}             # Current prompt storage (see Instruction Record)

JSON: chat_history:{session_id}         # Session chat history (see Retrieval Chunk)

SORTED_SET: chat_prompts:z              # Prompt ordering
  - member: key (chat_prompt:{id} or chat_prompt_json:{id})
  - score: created_at (float)
```

### Request Envelope (API)
```python
# POST /api/chat/send-stream
{
  "query": "string | null",              # User message (max 4000 chars)
  "session_id": "string (UUID)",         # Required
  "prompt_id": "string | null",          # Optional prompt reference
  "stream": "boolean (default: true)",   # Enable SSE streaming
  "detailed": "boolean (default: false)" # Unused legacy field
}
```

### Response Envelope (SSE Stream)
```python
# Event 1: Processing
{"status": "processing", "message": "Preparing response..."}

# Event 2-N: Content chunks
{"status": "chunk", "chunk": "string (markdown)"}

# Event N+1: Metadata
{"status": "meta", "chunk": {"user_details_known": "boolean", "user_network_id": "string | null"}}

# Event N+2: Form trigger (conditional)
{"status": "form_trigger", "chunk": ""}

# Event N+3: Session end (conditional, count >= 50)
{"status": "end_chat", "chunk": "string (closure message)"}
```

### Message Record (PostgreSQL)
```python
{
  "id": "UUID (text)",                   # Primary key
  "session_id": "UUID (text)",           # Foreign key to sessions
  "role": "string (user|assistant)",     # Sender type
  "content": "string",                    # Message text (markdown)
  "reply_to": "UUID | null",             # Reply chain reference
  "follow_up_to": "UUID | null",         # Follow-up chain reference
  "follow_up_depth": "integer (default: 0)", # Nesting level
  "metadata": "jsonb (default: {})",     # Extensible data
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

---

## EXECUTION PATHS

### 1. Request Handling
```
1. Client sends POST /api/chat/send-stream with SentMessage payload
2. router.py:post_send_message_stream receives request
3. helpers.py:send_message_stream extracts session_id, query
4. Yield SSE event: {"status": "processing"}
5. Fire asyncio.create_task to save user message to PostgreSQL (non-blocking)
6. Fire asyncio.create_task to append user message to Redis chat_history (non-blocking)
7. Fetch conversation history from PostgreSQL via get_messages_for_session
8. Instantiate OptimizedChatbot and call get_detailed_response
9. Stream response events back to client via StreamingResponse
10. Fire asyncio.create_task to save assistant message to PostgreSQL (non-blocking)
11. Fire asyncio.create_task to append assistant message to Redis chat_history (non-blocking)
12. If session_ended flag, fire asyncio.create_task to update sessions.is_active=FALSE
```

### 2. Retrieval Pipeline
```
1. chatbot_optimizer.py:get_detailed_response receives query, chat_history, session_id
2. Calculate dynamic top_n (4-30) based on query length, question patterns, conversation context
3. Call get_redis_context_chunks via loop.run_in_executor (offload blocking I/O)
4. redis_context.py:get_redis_context_chunks derives search query from chat history
5. Execute similarity_search on chunk_index (KB vector search, top_n results)
6. Execute similarity_search_chat_history on chat_history_index (session vector search, top_n results)
7. Normalize results: extract text from response/messages_text fields
8. Deduplicate chunks (history prioritized over KB)
9. Return merged list of context chunks
10. Join chunks with "\n\n---\n\n" separator
```

### 3. Inference Assembly
```
1. Build LLM context from chat history via build_llm_context_from_history
2. Summarize all messages except latest user message using summarize_chunks_with_llm
3. Fetch user_details from PostgreSQL via get_user_details_from_db (cached for request)
4. Fetch user_details_known flag from PostgreSQL via get_user_details_known_from_db
5. Call final_response_prompt with: context, conversation_summary, query, count, user_details_known, user_details
6. Construct prompt with sections: Role & Mission, Critical Rules, Conversational Intelligence, Lead Capture, Funnel Logic, Output Schema, Context Block, Reminders
7. Call generate_llm_response (OpenAI openai/gpt-oss-120b) with assembled prompt
8. Parse JSON response: extract "response" (markdown text), "funnel_stage" (Awareness|Interest|Intent|Action)
9. Format response via format_response: validate URLs, normalize spacing
10. If user_details_known=True, remove bold follow-up questions via regex
11. Yield {"status": "chunk", "chunk": formatted_text}
```

### 4. Post-Processing
```
1. Evaluate form trigger logic based on funnel_stage and message count
2. If funnel_stage="action" AND count >= 2 AND user_details_known=False: trigger_form=True
3. If funnel_stage="intent" AND count >= 2 AND user_details_known=False: trigger_form=True
4. If funnel_stage="interest" AND count >= 3 AND user_details_known=False: trigger_form=True
5. If count >= 10 AND user_details_known=False: trigger_form=True (fallback)
6. If trigger_form=True: yield {"status": "form_trigger", "chunk": ""}
7. Else: yield {"status": "meta", "chunk": {"user_details_known": boolean}}
8. If count >= 50: yield {"status": "end_chat", "chunk": closure_message}
```

### 5. Update Propagation (User Details Capture)
```
1. Client sends PATCH /api/user/{session_id} with UserCreate payload
2. router.py:update_user calls helpers.py:update_user_by_session
3. Fetch user_id from sessions table via session_id
4. Update users table: set username, email, mobile, user_details_known=TRUE
5. Commit transaction to PostgreSQL
6. Call delete_last_user_message to remove form trigger message from DB
7. Return success response with session_id and deleted_message_id
8. No Redis cache invalidation required (user_details fetched from DB on each request)
```

### 6. Sync Propagation (Prompt Updates)
```
1. Developer edits prompts.py: modify final_response_prompt function or PROMPT_VERSION
2. Run script: python scripts/refresh_prompts_to_redis.py --as-json
3. Script calls redis_prompts.py:refresh_prompts
4. Load prompts from prompts.py module via _fallback_to_prompts_module
5. For each prompt: create JSON document with id, prompt, created_at, source, lang
6. Enrich with assistant_name and assistant_instruction from assistant_instructions_helper
7. Store as RedisJSON at key chat_prompt_json:{id} via client.json().set
8. Add key to sorted set chat_prompts:z with score=created_at
9. Ensure RediSearch index chat_prompts_json exists (JSON ON JSON PREFIX)
10. Application reads prompts via redis_prompts.py:load_prompts (auto-detects JSON index)
11. No application restart required (prompts loaded dynamically from Redis)
```

---

## MODIFICATION RULES

### prompts.py → DB → Redis
```
RULE 1: prompts.py is SOURCE OF TRUTH for instruction logic
RULE 2: PostgreSQL prompts table stores UI-facing prompt options (greeting, buttons)
RULE 3: Redis chat_prompt_json:{id} stores LLM instruction templates
RULE 4: Changes to prompts.py require manual script execution: refresh_prompts_to_redis.py
RULE 5: PROMPT_VERSION in prompts.py tracks instruction changes (semantic versioning)
RULE 6: final_response_prompt function generates dynamic prompts (not stored in Redis)
RULE 7: Redis prompts are OPTIONAL fallback (app uses prompts.py directly for LLM calls)
RULE 8: PostgreSQL prompts are REQUIRED for /api/prompts/root endpoint
```

### Utility Edits → DB → Redis
```
RULE 1: Utility scripts (refresh_prompts_to_redis.py) are WRITE-ONLY to Redis
RULE 2: Scripts do NOT modify PostgreSQL prompts table
RULE 3: Scripts read from prompts.py module or JSON file (--file flag)
RULE 4: --as-json flag controls storage format (JSON vs HASH)
RULE 5: --ensure-index flag controls RediSearch index creation
RULE 6: Scripts are IDEMPOTENT (safe to re-run, overwrites existing keys)
RULE 7: No automatic sync mechanism (manual execution required)
```

### Runtime Cache Invalidation → Session Context Reconstruction
```
RULE 1: Redis chat_history:{session_id} is APPEND-ONLY during session
RULE 2: save_chat_history OVERWRITES entire chat_history key (not incremental)
RULE 3: Embedding computation is ASYNC (background thread via _EMBEDDING_EXECUTOR)
RULE 4: Embedding failures are SILENT (logged but not blocking)
RULE 5: get_chat_history falls back to PostgreSQL messages table on Redis miss
RULE 6: Session context is RECONSTRUCTED on each request (no persistent in-memory state)
RULE 7: user_details_known flag is ALWAYS fetched from PostgreSQL (no Redis cache)
RULE 8: Conversation summary is REGENERATED on each request via build_llm_context_from_history
RULE 9: Redis vector search results are MERGED with PostgreSQL history (deduplication)
RULE 10: No explicit cache invalidation API (keys expire naturally or overwritten)
```

---

## VERSIONING RULES

### Version Increment
```
RULE 1: PROMPT_VERSION in prompts.py follows semantic versioning (MAJOR.MINOR.PATCH)
RULE 2: MAJOR bump: Breaking changes to prompt structure or critical rules
RULE 3: MINOR bump: New features, sections, or non-breaking enhancements
RULE 4: PATCH bump: Bug fixes, typos, clarifications
RULE 5: Version change REQUIRES comment in prompts.py explaining modification
RULE 6: Current version: 2.1.0 (Fixed repeated questions issue with explicit examples)
```

### Conflict Resolution
```
RULE 1: PostgreSQL is SOURCE OF TRUTH for user data (users, sessions, messages)
RULE 2: prompts.py is SOURCE OF TRUTH for LLM instructions
RULE 3: Redis is CACHE LAYER (can be flushed without data loss)
RULE 4: On Redis-PostgreSQL conflict: PostgreSQL wins (fallback logic in get_chat_history)
RULE 5: On prompts.py-Redis conflict: prompts.py wins (app uses prompts.py directly)
RULE 6: No distributed locking (single-instance deployment assumed)
RULE 7: Concurrent writes to same session: last-write-wins (PostgreSQL MVCC)
```

### Stale Session Detection
```
RULE 1: sessions.last_interaction_at updated on every message save
RULE 2: sessions.is_active flag tracks session lifecycle (TRUE=active, FALSE=ended)
RULE 3: Session ends when: count >= 50 OR explicit /api/session/end/{session_id} call
RULE 4: Stale session = is_active=FALSE (no automatic expiration by time)
RULE 5: Redis chat_history keys have NO TTL (persist indefinitely)
RULE 6: Application checks is_active flag on /api/chat/{session_id}/messages
RULE 7: No automatic cleanup of old sessions (manual DB maintenance required)
RULE 8: PROMPT_VERSION mismatch detection: NONE (no version tracking per session)
RULE 9: Prompt changes apply IMMEDIATELY to all sessions (no migration logic)
RULE 10: User sees latest prompt behavior on next message (no session isolation)
```

---

## NOTES

- **No Workers**: No separate worker processes (async tasks via asyncio.create_task)
- **No Queues**: No message queue system (direct DB writes, fire-and-forget async)
- **No Caching Layer**: Redis serves dual purpose (cache + vector search)
- **No API Gateway**: Direct FastAPI exposure (CORS middleware for cross-origin)
- **No Load Balancer**: Single-instance deployment (no horizontal scaling)
- **No Service Mesh**: Direct component communication (no sidecar proxies)
- **No Monitoring**: Logging only (app.log file, no metrics/tracing infrastructure)
- **No CI/CD**: Manual deployment (Dockerfile + docker-compose.yml provided)
- **No Schema Migrations**: Manual SQL execution (migrations/ directory exists but unused)
- **No Feature Flags**: Prompt version is only toggle mechanism
