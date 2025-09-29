from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import json
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


class RedisContextDataResponse(BaseModel):
    status: str
    session_id: str
    created_at: str
    queries: List[dict]


@router.post("/redis-context", response_model=Union[RedisContextDataResponse, RedisContextStoreResponse])
async def redis_context_endpoint(payload: RedisContextRequest):
    if payload.fetch_context:
        # Use similarity_search to get queries with similarity scores
        key = f"session:{payload.session_id}"
        stored_results = r.json().get(key)
        if stored_results and "created_at" in stored_results:
            created_at = stored_results["created_at"]
        else:
            created_at = datetime.utcnow().isoformat()

        queries = similarity_search(payload.session_id, payload.text)
        # Ensure every query object has a similarity field
        for q in queries:
            if "similarity" not in q:
                q["similarity"] = None

        response_obj = {
            "status": "success",
            "session_id": payload.session_id,
            "created_at": created_at,
            "queries": queries
        }
        return RedisContextDataResponse(**response_obj)

    else:
        # Store the text after vectorizing
        success = store_text(payload.session_id, payload.text)
        if success:
            return RedisContextStoreResponse(status="success", message="Text stored successfully.")
        else:
            raise HTTPException(status_code=500, detail="Failed to store text.")
