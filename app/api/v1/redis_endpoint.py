from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Union, Optional

from app.db.redis_vector_helper import store_text, similarity_search, r

router = APIRouter()


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
        strt_time = datetime.utcnow()
        # Use similarity_search to get queries with similarity scores
        key = f"session:{payload.session_id}"
        stored_results = r.json().get(key)
        if stored_results and "created_at" in stored_results:
            created_at = stored_results["created_at"]
        else:
            created_at = datetime.utcnow().isoformat()

        queries = similarity_search(payload.session_id, payload.text)

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
            "response_time": (datetime.utcnow() - strt_time).total_seconds(),
        }
        return RedisContextDataResponse(**response_obj)

    else:
        # Store the text after vectorizing
        success = store_text(payload.session_id, payload.text)
        if success:
            return RedisContextStoreResponse(status="success", message="Text stored successfully.")
        else:
            raise HTTPException(status_code=500, detail="Failed to store text.")
