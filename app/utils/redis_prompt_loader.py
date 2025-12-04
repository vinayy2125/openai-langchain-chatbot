from app.config import get_redis
from app.db.redis_prompts import load_prompts

def get_system_prompt_from_redis(prompt_id=None):
    """
    Fetch a system prompt from Redis (chat_prompt_json) by id, or the latest if id is None.
    Returns the prompt text, or None if not found.
    """
    redis_client = get_redis
    prompts = load_prompts(redis_client, limit=1, json_index=True)
    if not prompts:
        return None
    if prompt_id:
        for p in prompts:
            if str(p.get("id")) == str(prompt_id):
                return p.get("prompt")
        return None
    return prompts[0].get("prompt")
