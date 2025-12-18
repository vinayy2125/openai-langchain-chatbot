from typing import Optional, Dict
import json
import logging
from app.utils.redis_context import get_redis_client

logger = logging.getLogger("redis_prompt_loader")

def get_prompt_sections_from_redis() -> Optional[Dict[str, str]]:
    """
    Fetch structured prompt sections from Redis.
    Returns a dict with keys: core, behavior, funnel_logic, output_schema, reminders
    Returns None if not found or on error.
    """
    try:
        r = get_redis_client()
        # Try fetching as JSON first if using RedisJSON, or string
        # The checklist says "All admin-editable prompt sections are stored in Redis as a single JSON object"
        # We can use .json().get() if the client supports it, or get() and parse.
        
        # Using json().get() for 'chat_prompt_json'
        try:
            sections = r.json().get("chat_prompt_json")
        except Exception:
            # Fallback for standard redis client or if key is string
            data = r.get("chat_prompt_json")
            if data:
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                sections = json.loads(data)
            else:
                sections = None

        if sections and isinstance(sections, dict):
            # Validate expected keys
            expected_keys = ["core", "behavior", "funnel_logic", "output_schema", "reminders"]
            # We don't enforce STRICT containment, but we check if we have what we need.
            # If some are missing, dynamic_prompts.py handles empty strings safely.
            # But logging a warning is good.
            missing = [k for k in expected_keys if k not in sections]
            if missing:
                logger.warning(f"Redis chat_prompt_json missing keys: {missing}")
            
            return sections
            
        return None
    except Exception as e:
        logger.error(f"Error fetching prompt sections from Redis: {e}")
        return None
