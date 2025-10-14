import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Union, Optional
from app.config import get_redis
from app.db.redis_vector_helper import store_text, similarity_search 
router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisContextRequest(BaseModel):
    session_id: str
    text: str  # Input text or query
    fetch_context: bool = False



class RedisContextStoreResponse(BaseModel):
    status: str
    message: str
    response_time: Optional[float]


class RedisContextDataResponse(BaseModel):
    status: str
    session_id: str
    created_at: str
    queries: Optional[List[dict]]
    response_time: Optional[float]
    


@router.post("/redis-context", response_model=Union[RedisContextDataResponse, RedisContextStoreResponse])
async def redis_context_endpoint(payload: RedisContextRequest):
    if payload.fetch_context:
        strt_time = datetime.now(timezone.utc)
        # Use similarity_search to get queries with similarity scores
        key = f"session:{payload.session_id}"
        stored_results = get_redis.json().get(key)
        if stored_results and "created_at" in stored_results:
            created_at = stored_results["created_at"]
        else:
            created_at = datetime.now(timezone.utc).isoformat()

        queries = similarity_search(payload.session_id, payload.text)
        logging.info(f"Similarity search returned {payload.text} results.")
        logging.info(f"Retrieved {len(queries)} similar queries from Redis.")
        # Remove embeddings from each query dict
        cleaned_queries = []
        for q in queries:
            # make a shallow copy of q excluding 'query_embedding'
            cleaned_q = {k: v for k, v in q.items() if k != "query_embedding"}
            # Ensure similarity key exists
            if "similarity" not in cleaned_q:
                cleaned_q["similarity"] = None
            cleaned_queries.append(cleaned_q)

        response_obj = {
            "status": "success",
            "session_id": payload.session_id,
            "created_at": created_at,
            "queries": cleaned_queries,
            "response_time": (datetime.now(timezone.utc) - strt_time).total_seconds(),
        }
        return RedisContextDataResponse(**response_obj)

    else:
        # Store the text after vectorizing
        strt_time = datetime.now(timezone.utc)
        success = store_text(payload.session_id, payload.text)
        response_time = (datetime.now(timezone.utc) - strt_time).total_seconds()       
        if success:
            return RedisContextStoreResponse(status="success", message="Text stored successfully.", response_time=response_time)
        else:
            
            raise HTTPException(status_code=500, detail="Failed to store text.")
