from __future__ import annotations
from datetime import datetime
from typing import List, Optional
import pdfplumber

import numpy as np


# redis / RediSearch imports
from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from core_services.generate_embeddings import get_embedding

# local config & helpers (must exist in your project)
from app.config import settings, get_redis_client

# module logger
logger = logging.getLogger("redis_service")

# -------------------------
# Configuration (override via settings)
# -------------------------
PREFIX = getattr(settings, "redis_prefix", "dits_chatbot:")

    q_emb = get_embedding(query)
    vec = np.array(q_emb, dtype=np.float32)

    # numeric range filter syntax: @user_id:[<min> <max>]
    # we restrict to user_id exactly by giving same min/max
    q_str = f'(@user_id:[{int(user_id)} {int(user_id)}])=>[KNN {top_k} @embedding $vec_param AS score]'

    # Build Query object (return text/chat_id/created_at/chunk_index for richer context)
    query_obj = Query(q_str).return_fields("text", "chat_id", "created_at", "chunk_index").dialect(2)

    try:
        res = r.ft(INDEX_NAME).search(query_obj, query_params={"vec_param": vec.tobytes()})
    except Exception as e:
        logger.exception("Redis search failed: %s", e)
        return "Error retrieving context."

    docs = getattr(res, "docs", []) or []
    if not docs:
        logger.debug("No vector matches found for user=%s", user_id)
        return ""

    parts = []
    # Attempt to fetch stored JSON for each doc.id to be robust across redis-py variations
    for d in docs:
        try:
            doc_id = getattr(d, "id", None)
            if doc_id:
                stored = r.json().get(doc_id)
                if stored:
                    t = stored.get("text")
                    if t:
                        parts.append(t)
                        continue
            # fallback: check the doc attributes returned by Query.return_fields(...)
            # Some redis clients attach returned fields as attributes (e.g., d.text)
            text_field = getattr(d, "text", None) or getattr(d, "payload", {}).get("text", None) if hasattr(d, "payload") else None
            if text_field:
                parts.append(text_field)
        except Exception as e:
            logger.debug("Failed to parse search doc: %s", e)

    # join and return
    context = "\n".join(parts)
    logger.debug("retrieve_context returned %d parts (len=%d)", len(parts), len(context))
    return context


# -------------------------
# Indexing helpers
# -------------------------
def index_transcript(user_id: int, transcript: str, chat_id: Optional[int] = None):
    """
    Chunk transcript and store as JSON docs in Redis.
    Uses per-user counter key to group all chunks for same user under:
      voicechat:user:{user_id}:chunk:{n}
    Each JSON contains user_id, chat_id, chunk_index, text, created_at, embedding.
    """

    logger.info("index_transcript user=%s chat_id=%s len=%d", user_id, chat_id, len(transcript or ""))
    if not transcript:
        return

    r = get_redis_client()
    ensure_index_exists(r)

    # naive chunking ~1800 chars per chunk
    lines = [l.strip() for l in transcript.replace("\r", "\n").split("\n") if l.strip()]
    chunks = []
    cur = []
    for line in lines:
        if sum(len(x) for x in cur) + len(line) + 1 > 1800:
            chunks.append(" ".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        chunks.append(" ".join(cur))

    counter_key = f"{PREFIX}user:{user_id}:counter"

    for chunk_text in chunks:
        emb = get_embedding(chunk_text)
        try:
            chunk_index = int(r.incr(counter_key))
        except Exception:
            # fallback to timestamp-based index if INCR fails (rare)
            chunk_index = int(datetime.utcnow().timestamp() * 1000)

        key = f"{PREFIX}user:{user_id}:chunk:{chunk_index}"
        doc = {
            "user_id": int(user_id),
            "chat_id": int(chat_id) if chat_id is not None else None,
            "chunk_index": int(chunk_index),
            "text": chunk_text,
            "created_at": datetime.utcnow().isoformat(),
            "embedding": np.array(emb, dtype=np.float32).tolist(),
        }
        try:
            r.json().set(key, "$", doc)
        except Exception as e:
            logger.exception("Failed to store transcript chunk to Redis (key=%s): %s", key, e)

    logger.info("Indexed %d chunks for user=%s", len(chunks), user_id)


def index_document(file_path: str, document_id: int, user_id: int, db=None):
    """
    Index a PDF document (page-level chunks) into Redis JSON docs grouped by user counter.
    If db (SQLAlchemy session) is provided, tries to mark DB document as indexed similarly to previous behavior.
    """
    from core_services.generate_embeddings import get_embedding

    logger.info("index_document file=%s user=%s document_id=%s", file_path, user_id, document_id)
    r = get_redis_client()
    ensure_index_exists(r)

    page_chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                if text.strip():
                    page_chunks.append(text.strip())
    except Exception:
        logger.exception("Failed to extract PDF text")

    if not page_chunks:
        logger.warning("No text extracted from document: %s", file_path)
        return

    counter_key = f"{PREFIX}user:{user_id}:counter"
    for i, chunk_text in enumerate(page_chunks):
        emb = get_embedding(chunk_text)
        try:
            chunk_index = int(r.incr(counter_key))
        except Exception:
            chunk_index = int(datetime.utcnow().timestamp() * 1000)

        key = f"{PREFIX}user:{user_id}:chunk:{chunk_index}"
        doc = {
            "document_id": int(document_id),
            "user_id": int(user_id),
            "chunk_index": int(chunk_index),
            "text": chunk_text,
            "created_at": datetime.utcnow().isoformat(),
            "embedding": np.array(emb, dtype=np.float32).tolist(),
        }
        try:
            r.json().set(key, "$", doc)
        except Exception:
            logger.exception("Failed to store document chunk to Redis: %s", key)

    # optionally update DB 'indexed' flag if db session supplied
    if db is not None:
        try:
            from db.database import Document as DBDocument
            doc_obj = db.query(DBDocument).filter(DBDocument.id == document_id).first()
            if doc_obj:
                doc_obj.indexed = True
                doc_obj.indexed_at = datetime.utcnow()
                db.commit()
        except Exception:
            logger.exception("Failed to mark DB document as indexed for document_id=%s", document_id)


# -------------------------
# (Optional) Utility: small function to scan per-user chunks (not required)
# -------------------------
def list_user_chunks(user_id: int, limit: int = 100) -> List[str]:
    """
    Return list of keys for user's chunks (useful for debugging).
    """
    r = get_redis_client()
    pattern = f"{PREFIX}user:{user_id}:chunk:*"
    cursor = 0
    out = []
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)
        for k in keys:
            out.append(k.decode() if isinstance(k, bytes) else k)
            if len(out) >= limit:
                return out
        if cursor == 0:
            break
    return out

def generate_and_store_embedding(r, session_id: int, query: str, response: str) -> str:
    """Generate embedding for the given text and store it in Redis.
    Groups queries and embeddings by session_id.

    Args:
        r: Redis client instance.
        session_id: The session ID to group queries under.
        query: The query text to generate an embedding for.
        response: The response text associated with the query.

    Returns:
        The Redis key under which the document is stored.
    """

    # Generate embedding for the new query
    query_embedding = get_embedding(query)
    
    # Create a session-based key
    session_key = f"{PREFIX}session:{session_id}"
    
    try:
        # Try to get existing session data
        existing_data = r.json().get(session_key)
        
        if existing_data:
            # Session exists, append new query data
            existing_data["queries"].append({
                "query": query,
                "query_embedding": query_embedding,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Update the existing session
            r.json().set(session_key, '$', existing_data)
        else:
            # Create new session entry
            new_session_data = {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "queries": [{
                    "query": query,
                    "query_embedding": query_embedding,
                    "response": response,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            r.json().set(session_key, '$', new_session_data)
    
    except Exception as e:
        logger.error(f"Error storing embeddings for session {session_id}: {str(e)}")
        raise
    
    return session_key