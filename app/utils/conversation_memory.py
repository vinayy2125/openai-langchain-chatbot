"""
LangChain-based Conversation Memory Manager

Provides efficient in-memory conversation context with automatic summarization.
Replaces Redis-dependent chat history for context building.

Key Features:
- ConversationSummaryBufferMemory: Summary + recent messages (token efficient)
- Per-session memory management (thread-safe)
- No Redis dependency for chat context
- Single LLM call instead of separate summary call
"""

import threading
from typing import Dict, Optional, List
from app.logger import get_logger

logger = get_logger("conversation_memory")

# Lazy imports to avoid circular dependencies
_memory_class = None
_llm_instance = None


def _get_memory_class():
    """Lazy load ConversationSummaryBufferMemory to avoid import issues."""
    global _memory_class
    if _memory_class is None:
        try:
            from langchain.memory import ConversationSummaryBufferMemory
            _memory_class = ConversationSummaryBufferMemory
        except ImportError:
            # Fallback to basic buffer if summary buffer not available
            from langchain.memory import ConversationBufferMemory
            _memory_class = ConversationBufferMemory
            logger.warning("ConversationSummaryBufferMemory not available, using basic buffer")
    return _memory_class


def _get_llm():
    """Get shared LLM instance for summarization."""
    global _llm_instance
    if _llm_instance is None:
        try:
            from app.utils.llm_client import llm
            _llm_instance = llm
        except Exception as e:
            logger.error(f"Failed to get LLM for memory summarization: {e}")
            return None
    return _llm_instance


class SessionMemoryManager:
    """
    Manages per-session conversation memory with automatic summarization.
    
    Uses LangChain's ConversationSummaryBufferMemory:
    - Keeps full content of recent messages (buffer)
    - Automatically summarizes older messages when buffer exceeds limit
    - Returns combined summary + buffer for prompt context
    
    Thread-safe for concurrent sessions.
    """
    
    _instance: Optional["SessionMemoryManager"] = None
    _lock = threading.Lock()
    
    def __init__(self, max_token_limit: int = 800):
        """
        Initialize memory manager.
        
        Args:
            max_token_limit: Token limit for the buffer before summarization kicks in.
                            Lower = more aggressive summarization, fewer tokens.
                            Default 800 keeps ~8-10 recent messages before summarizing.
        """
        self._sessions: Dict[str, object] = {}  # session_id -> Memory instance
        self._session_locks: Dict[str, threading.Lock] = {}
        self._max_token_limit = max_token_limit
        self._global_lock = threading.Lock()
        logger.info(f"SessionMemoryManager initialized with max_token_limit={max_token_limit}")
    
    @classmethod
    def get_instance(cls, max_token_limit: int = 800) -> "SessionMemoryManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_token_limit)
        return cls._instance
    
    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """Get or create a lock for a specific session."""
        with self._global_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]
    
    def get_or_create_memory(self, session_id: str):
        """
        Get or create memory for a session.
        
        Returns a ConversationSummaryBufferMemory instance configured with:
        - Automatic summarization when buffer exceeds token limit
        - Human/AI message formatting
        """
        lock = self._get_session_lock(session_id)
        with lock:
            if session_id not in self._sessions:
                llm = _get_llm()
                MemoryClass = _get_memory_class()
                
                if llm is not None and MemoryClass.__name__ == "ConversationSummaryBufferMemory":
                    # Use summary buffer memory (preferred)
                    memory = MemoryClass(
                        llm=llm,
                        max_token_limit=self._max_token_limit,
                        return_messages=False,  # Return as string for easy prompt integration
                        memory_key="history",
                        human_prefix="User",
                        ai_prefix="Assistant",
                    )
                    logger.debug(f"Created ConversationSummaryBufferMemory for session {session_id}")
                else:
                    # Fallback to basic buffer (no summarization)
                    memory = MemoryClass(
                        return_messages=False,
                        memory_key="history",
                        human_prefix="User",
                        ai_prefix="Assistant",
                    )
                    logger.debug(f"Created basic ConversationBufferMemory for session {session_id}")
                
                self._sessions[session_id] = memory
            
            return self._sessions[session_id]
    
    def add_user_message(self, session_id: str, content: str) -> None:
        """Add a user message to session memory."""
        if not content or not content.strip():
            return
        
        memory = self.get_or_create_memory(session_id)
        lock = self._get_session_lock(session_id)
        
        with lock:
            try:
                memory.chat_memory.add_user_message(content.strip())
                logger.debug(f"Added user message to session {session_id}")
            except Exception as e:
                logger.error(f"Failed to add user message to memory: {e}")
    
    def add_ai_message(self, session_id: str, content: str) -> None:
        """Add an AI/assistant message to session memory."""
        if not content or not content.strip():
            return
        
        memory = self.get_or_create_memory(session_id)
        lock = self._get_session_lock(session_id)
        
        with lock:
            try:
                memory.chat_memory.add_ai_message(content.strip())
                logger.debug(f"Added AI message to session {session_id}")
            except Exception as e:
                logger.error(f"Failed to add AI message to memory: {e}")
    
    def get_context(self, session_id: str) -> str:
        """
        Get conversation context for prompt injection.
        
        Returns formatted string with:
        - Summary of older messages (if any)
        - Recent messages in buffer
        
        This is the main method to call when building LLM prompts.
        """
        memory = self.get_or_create_memory(session_id)
        lock = self._get_session_lock(session_id)
        
        with lock:
            try:
                # load_memory_variables returns dict with memory_key
                result = memory.load_memory_variables({})
                context = result.get("history", "")
                
                if context:
                    logger.debug(f"Retrieved context for session {session_id} ({len(context)} chars)")
                
                return context if isinstance(context, str) else str(context)
            except Exception as e:
                logger.error(f"Failed to get context from memory: {e}")
                return ""
    
    def get_message_count(self, session_id: str) -> int:
        """Get number of messages in session memory."""
        if session_id not in self._sessions:
            return 0
        
        memory = self._sessions[session_id]
        lock = self._get_session_lock(session_id)
        
        with lock:
            try:
                return len(memory.chat_memory.messages)
            except Exception:
                return 0
    
    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session."""
        lock = self._get_session_lock(session_id)
        with lock:
            if session_id in self._sessions:
                try:
                    self._sessions[session_id].clear()
                except Exception:
                    pass
                del self._sessions[session_id]
                logger.info(f"Cleared memory for session {session_id}")
    
    def initialize_from_history(self, session_id: str, messages: List[Dict]) -> None:
        """
        Initialize memory from existing chat history (e.g., from DB on session recovery).
        
        Args:
            session_id: Session identifier
            messages: List of message dicts with 'role' and 'content' keys
        """
        if not messages:
            return
        
        memory = self.get_or_create_memory(session_id)
        lock = self._get_session_lock(session_id)
        
        with lock:
            try:
                # Clear existing and rebuild
                memory.clear()
                
                for msg in messages:
                    role = msg.get("role") or msg.get("sender", "")
                    content = msg.get("content", "")
                    
                    if not content:
                        continue
                    
                    if role.lower() == "user":
                        memory.chat_memory.add_user_message(content)
                    elif role.lower() in ("assistant", "ai", "bot"):
                        memory.chat_memory.add_ai_message(content)
                
                logger.info(f"Initialized memory for session {session_id} with {len(messages)} messages")
            except Exception as e:
                logger.error(f"Failed to initialize memory from history: {e}")


# Convenience function
def get_session_memory_manager() -> SessionMemoryManager:
    """Get the singleton SessionMemoryManager instance."""
    return SessionMemoryManager.get_instance()


def set_session_metadata(session_id: str, key: str, value) -> None:
    """Set arbitrary metadata for a session's memory object.

    This stores metadata on the memory instance associated with the session.
    It's useful for small per-session flags (like UC1 state) that should live
    alongside the conversation buffer memory.
    """
    try:
        mgr = get_session_memory_manager()
        memory = mgr.get_or_create_memory(session_id)
        lock = mgr._get_session_lock(session_id)
        with lock:
            if not hasattr(memory, "_session_metadata"):
                setattr(memory, "_session_metadata", {})
            memory._session_metadata[key] = value
            logger.debug(f"Set session metadata for {session_id}: {key}")
    except Exception as e:
        logger.error(f"Failed to set session metadata: {e}")


def get_session_metadata(session_id: str, key: str, default=None):
    """Get metadata previously stored for a session, or `default` if missing."""
    try:
        mgr = get_session_memory_manager()
        memory = mgr.get_or_create_memory(session_id)
        lock = mgr._get_session_lock(session_id)
        with lock:
            meta = getattr(memory, "_session_metadata", {})
            return meta.get(key, default)
    except Exception as e:
        logger.error(f"Failed to get session metadata: {e}")
        return default


def delete_session_metadata(session_id: str, key: str) -> None:
    """Delete a metadata key for a session if present."""
    try:
        mgr = get_session_memory_manager()
        memory = mgr.get_or_create_memory(session_id)
        lock = mgr._get_session_lock(session_id)
        with lock:
            if hasattr(memory, "_session_metadata") and key in memory._session_metadata:
                del memory._session_metadata[key]
                logger.debug(f"Deleted session metadata for {session_id}: {key}")
    except Exception as e:
        logger.error(f"Failed to delete session metadata: {e}")
