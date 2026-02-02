"""
Redis-based caching layer for improved performance.

This module provides a caching layer to reduce database load and improve
response times for frequently accessed data.
"""
import json
import pickle
from typing import Any, Optional, Callable
from functools import wraps
import hashlib
from app.config import get_redis_client
from app.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Manages caching operations using Redis.
    
    Provides methods for getting, setting, and invalidating cached data
    with support for TTL (time-to-live) and different serialization methods.
    """
    
    def __init__(self, prefix: str = "chatbot"):
        """
        Initialize the cache manager.
        
        Args:
            prefix: Prefix for all cache keys to avoid collisions
        """
        self.prefix = prefix
        try:
            self.redis_client = get_redis_client()
            logger.info("Cache manager initialized with Redis")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """
        Create a prefixed cache key.
        
        Args:
            key: The base key
            
        Returns:
            str: Prefixed cache key
        """
        return f"{self.prefix}:{key}"
    
    def get(self, key: str, deserialize: str = "json") -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            deserialize: Deserialization method ('json' or 'pickle')
            
        Returns:
            The cached value or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._make_key(key)
            value = self.redis_client.get(cache_key)
            
            if value is None:
                logger.debug(f"Cache miss: {key}")
                return None
            
            logger.debug(f"Cache hit: {key}")
            
            if deserialize == "json":
                return json.loads(value)
            elif deserialize == "pickle":
                return pickle.loads(value)
            else:
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        serialize: str = "json"
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiration)
            serialize: Serialization method ('json' or 'pickle')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(key)
            
            if serialize == "json":
                serialized_value = json.dumps(value)
            elif serialize == "pickle":
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = str(value)
            
            if ttl:
                self.redis_client.setex(cache_key, ttl, serialized_value)
            else:
                self.redis_client.set(cache_key, serialized_value)
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(key)
            self.redis_client.delete(cache_key)
            logger.debug(f"Cache deleted: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "user:*")
            
        Returns:
            int: Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            cache_pattern = self._make_key(pattern)
            keys = self.redis_client.keys(cache_pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cache pattern deleted: {pattern} ({deleted} keys)")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Error deleting cache pattern {pattern}: {e}")
            return 0
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._make_key(key)
            return bool(self.redis_client.exists(cache_key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get the TTL of a cached key.
        
        Args:
            key: Cache key
            
        Returns:
            int: TTL in seconds, -1 if no expiration, None if key doesn't exist
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._make_key(key)
            ttl = self.redis_client.ttl(cache_key)
            return ttl if ttl >= -1 else None
        except Exception as e:
            logger.error(f"Error getting TTL for {key}: {e}")
            return None


# Global cache manager instance
_cache_manager = CacheManager()


def cached(
    key_prefix: str,
    ttl: int = 300,
    key_builder: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        key_prefix: Prefix for the cache key
        ttl: Time-to-live in seconds (default: 5 minutes)
        key_builder: Optional function to build cache key from args/kwargs
        
    Usage:
        @cached(key_prefix="user_details", ttl=300)
        def get_user_details(user_id):
            # expensive operation
            return details
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = f"{key_prefix}:{key_builder(*args, **kwargs)}"
            else:
                # Default: hash the arguments
                args_str = str(args) + str(sorted(kwargs.items()))
                args_hash = hashlib.md5(args_str.encode()).hexdigest()
                cache_key = f"{key_prefix}:{args_hash}"
            
            # Try to get from cache
            cached_value = _cache_manager.get(cache_key, deserialize="pickle")
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss for {func.__name__}, executing function")
            result = func(*args, **kwargs)
            
            # Cache the result
            _cache_manager.set(cache_key, result, ttl=ttl, serialize="pickle")
            
            return result
        
        # Add cache control methods to the wrapper
        wrapper.cache_clear = lambda: _cache_manager.delete_pattern(f"{key_prefix}:*")
        wrapper.cache_key_prefix = key_prefix
        
        return wrapper
    return decorator


def invalidate_cache(key_or_pattern: str):
    """
    Invalidate a specific cache key or pattern.
    
    Args:
        key_or_pattern: Cache key or pattern to invalidate
    """
    if "*" in key_or_pattern:
        _cache_manager.delete_pattern(key_or_pattern)
    else:
        _cache_manager.delete(key_or_pattern)


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager: The global cache manager
    """
    return _cache_manager


# Specific cache helpers for common operations

def cache_user_details(session_id: str, details: dict, ttl: int = 300):
    """Cache user details for a session."""
    _cache_manager.set(f"user_details:{session_id}", details, ttl=ttl)


def get_cached_user_details(session_id: str) -> Optional[dict]:
    """Get cached user details for a session."""
    return _cache_manager.get(f"user_details:{session_id}")


def invalidate_user_details(session_id: str):
    """Invalidate cached user details for a session."""
    _cache_manager.delete(f"user_details:{session_id}")


def cache_session_state(session_id: str, state: dict, ttl: int = 600):
    """Cache session state."""
    _cache_manager.set(f"session_state:{session_id}", state, ttl=ttl)


def get_cached_session_state(session_id: str) -> Optional[dict]:
    """Get cached session state."""
    return _cache_manager.get(f"session_state:{session_id}")


def invalidate_session_state(session_id: str):
    """Invalidate cached session state."""
    _cache_manager.delete(f"session_state:{session_id}")


def cache_root_prompts(prompts: dict, ttl: int = 3600):
    """Cache root prompts."""
    _cache_manager.set("root_prompts", prompts, ttl=ttl)


def get_cached_root_prompts() -> Optional[dict]:
    """Get cached root prompts."""
    return _cache_manager.get("root_prompts")


def invalidate_root_prompts():
    """Invalidate cached root prompts."""
    _cache_manager.delete("root_prompts")
