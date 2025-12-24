# app\config.py
import os
import redis
from dotenv import load_dotenv

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
 
class Settings:
    # Redis
    redis_host: str = REDIS_HOST
    redis_port: int = REDIS_PORT
    redis_password: str | None = REDIS_PASSWORD
 
settings = Settings()
 
 
def get_redis_client(connect_timeout: int = 5):
    """Return a connected redis.Redis client or raise an informative error.
 
    Returns None if the redis package is not installed or connection fails.
    """
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            socket_connect_timeout=connect_timeout,
            decode_responses=False, # We want bytes for embeddings
        )
        client.ping()
        return client
    except redis.AuthenticationError:
        raise RuntimeError("Redis authentication failed; check REDIS_PASSWORD")
    except redis.ConnectionError:
        raise RuntimeError("Could not connect to Redis; is the server running?")
    except Exception as exc:
        raise RuntimeError(f"Unexpected Redis error: {exc}")


# Lazy-loaded Redis client (initialized on first access)
_redis_client = None


def get_redis():
    """Get the lazily-initialized Redis client.
    
    Uses lazy initialization to:
    - Avoid failures at module import time if Redis is temporarily unavailable
    - Allow for connection retry on transient failures
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = get_redis_client()
    return _redis_client
