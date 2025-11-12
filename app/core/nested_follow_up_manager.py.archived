from typing import List, Dict
import logging
import time
from app.logger import get_logger
from app.core.services.chatbot_optimizer import OptimizedChatbot
from app.core.utils import generate_llm_response  
from app.core.prompts import SHARED_SYSTEM_PROMPT, assesment_prompt
logger = get_logger(__name__)


class FollowUpManager:
    def __init__(self, llm):
        self.llm = llm
        self.sessions = {}  # In-memory session cache for performance
        self.chatbot = OptimizedChatbot(llm=llm)
        self._processing_sessions = set()  # Track sessions being processed to prevent concurrency issues
        self._db_loaded_sessions = set()  # Track which sessions have been loaded from DB (hybrid optimization)
        self._last_followup_generation = {}  # Track when follow-ups were last generated per session

    def _load_conversation_history_from_db(self, session_id: str) -> List[Dict[str, str]]:
        """Load conversation history from database messages table"""
        try:
            from app.db.base import get_db_conn
            import psycopg2
            
            conn = get_db_conn()
            cursor = conn.cursor()
            
            # Get all messages for this session ordered by creation time
            cursor.execute("""
                SELECT role, content, created_at 
                FROM messages 
                WHERE session_id = %s 
                ORDER BY created_at ASC
            """, (str(session_id),))
            
            rows = cursor.fetchall()
            conversation_history = []
            
            for row in rows:
                role, content, created_at = row
                conversation_history.append({
                    "role": role,
                    "content": content,
                    "timestamp": created_at.isoformat() if created_at else ""
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"[DB_HISTORY] Loaded {len(conversation_history)} messages from database for session {session_id}")
            return conversation_history
            
        except Exception as e:
            logger.error(f"[DB_HISTORY] Failed to load conversation history from DB for session {session_id}: {e}")
            return []

    def resolve_complete_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Force resolution of complete conversation history, ignoring incoming_history.
        Priority: synced in-memory → database → empty (never uses incoming_history)
        
        Used by: Optimized flow (later conversation, 6+ exchanges) 
        Purpose: Accurate message counting for form trigger logic
        Critical: Always returns complete history regardless of client data
        """
        try:
            # Check if we have in-memory data that's already synced with DB
            session_data = self.get_session_data(session_id)
            memory_history = session_data.get("conversation_history", [])
            
            if memory_history and session_id in self._db_loaded_sessions:
                # We have memory data that's synced with DB - use it for performance
                logger.info(f"[RESOLVE_COMPLETE_HISTORY] Using synced in-memory history for session {session_id}: {len(memory_history)} messages")
                return memory_history
            
            # Load from database (first time or after server restart)
            db_history = self._load_conversation_history_from_db(session_id)
            
            if db_history:
                # Update memory with DB data and mark as synced
                session_data["conversation_history"] = db_history
                self.sessions[session_id] = session_data
                self._db_loaded_sessions.add(session_id)
                logger.info(f"[RESOLVE_COMPLETE_HISTORY] Loaded and cached {len(db_history)} messages from database for session {session_id}")
                return db_history
            
            # Fallback to existing memory (edge case - shouldn't happen in normal flow)
            if memory_history:
                logger.info(f"[RESOLVE_COMPLETE_HISTORY] Using existing in-memory history for session {session_id}: {len(memory_history)} messages")
                return memory_history
            
            # No history found anywhere
            logger.info(f"[RESOLVE_COMPLETE_HISTORY] No conversation history found for session {session_id}")
            return []
            
        except Exception:
            logger.exception(f"[FollowUpManager] Failed to resolve complete history for session {session_id}")
            return []

    def resolve_history(self, session_id: str, incoming_history: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
        """
        Optimized conversation history resolution with smart caching.
        Priority: incoming_history → synced in-memory → database → empty
        Uses hybrid approach for performance and persistence across restarts.
        
        Used by: Follow-up flow (early conversation, 1-5 exchanges)
        Purpose: Performance optimization, allows client-side caching
        Note: For form trigger accuracy, use resolve_complete_history() instead
        """
        try:
            if incoming_history:
                logger.info(f"[RESOLVE_HISTORY] Using incoming_history for session {session_id}: {len(incoming_history)} messages")
                return incoming_history
            
            # Check if we have in-memory data that's already synced with DB
            session_data = self.get_session_data(session_id)
            memory_history = session_data.get("conversation_history", [])
            
            if memory_history and session_id in self._db_loaded_sessions:
                # We have memory data that's synced with DB - use it for performance
                logger.info(f"[RESOLVE_HISTORY] Using synced in-memory history for session {session_id}: {len(memory_history)} messages")
                return memory_history
            
            # Load from database (first time or after server restart)
            db_history = self._load_conversation_history_from_db(session_id)
            
            if db_history:
                # Update memory with DB data and mark as synced
                session_data["conversation_history"] = db_history
                self.sessions[session_id] = session_data
                self._db_loaded_sessions.add(session_id)
                logger.info(f"[RESOLVE_HISTORY] Loaded and cached {len(db_history)} messages from database for session {session_id}")
                return db_history
            
            # Fallback to existing memory (edge case - shouldn't happen in normal flow)
            if memory_history:
                logger.info(f"[RESOLVE_HISTORY] Using existing in-memory history for session {session_id}: {len(memory_history)} messages")
                return memory_history
            
            # No history found anywhere
            logger.info(f"[RESOLVE_HISTORY] No conversation history found for session {session_id}")
            return []
            
        except Exception:
            logger.exception(f"[FollowUpManager] Failed to resolve history for session {session_id}")
            return []

    def initialize_session(self, session_id, prompt_id, prompt_context):
        """Initialize session with prompt context and conversation history"""
        self.sessions[session_id] = {
            "prompt_id": prompt_id,
            "prompt_context": prompt_context,
            "conversation_history": [],
            "state": {},
            "answers": {},
        }

    def add_to_conversation_history(self, session_id: str, role: str, content: str):
        """Optimized: Add to memory with one-time DB load and deduplication"""
        from datetime import datetime
        
        if session_id in self._processing_sessions:
            logger.warning(f"[CONCURRENT_ACCESS] Session {session_id} already processing, queuing: {role} - {content[:50]}...")
        
        self._processing_sessions.add(session_id)
        
        try:
            # Debug markdown preservation
            has_markdown = any(marker in content for marker in ["**", "*", "_", "#", "-", "`", "```", "\n"])
            if role == "assistant" and has_markdown:
                logger.info(f"[MARKDOWN_DEBUG] Adding assistant message with markdown: {content[:150]}...")
                markdown_count = content.count("**") + content.count("*") + content.count("#") + content.count("-")
                logger.info(f"[MARKDOWN_DEBUG] Markdown elements count: {markdown_count}")
            
            # Get current session data
            session_data = self.get_session_data(session_id)
            existing_history = session_data.get("conversation_history", [])
            
            # One-time DB load if memory is empty and session hasn't been DB-synced yet
            if not existing_history and session_id not in self._db_loaded_sessions:
                existing_history = self._load_conversation_history_from_db(session_id)
                session_data["conversation_history"] = existing_history
                self._db_loaded_sessions.add(session_id)
                logger.info(f"[ADD_MESSAGE] One-time DB load: {len(existing_history)} messages for session {session_id}")
            
            # Efficient deduplication check (only last 2 messages)
            if existing_history:
                for recent_msg in existing_history[-2:]:
                    if (recent_msg.get("role") == role and recent_msg.get("content") == content):
                        logger.warning(f"[DUPLICATE_PREVENTION] Skipping duplicate: {role} - {content[:50]}...")
                        return
            
            # Add to memory only - DB persistence handled by save_message() in helpers.py
            message_data = {
                "role": role, 
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            session_data["conversation_history"].append(message_data)
            self.sessions[session_id] = session_data
            
            # Debug markdown preservation after storage
            if role == "assistant" and has_markdown:
                stored_content = message_data["content"]
                stored_markdown_count = stored_content.count("**") + stored_content.count("*") + stored_content.count("#") + stored_content.count("-")
                logger.info(f"[MARKDOWN_DEBUG] Stored with markdown preserved: {stored_content[:150]}...")
                logger.info(f"[MARKDOWN_DEBUG] Stored markdown count: {stored_markdown_count}")
                
                if stored_markdown_count != markdown_count:
                    logger.warning(f"[MARKDOWN_WARNING] Markdown count changed! Original: {markdown_count}, Stored: {stored_markdown_count}")
            
            logger.info(f"[ADD_MESSAGE] Added to memory for session {session_id}: {role} - {content[:100]}...")
            logger.info(f"[COMPLETE_SESSION_HISTORY] Session {session_id} now has {len(session_data['conversation_history'])} messages in memory")
            
        finally:
            self._processing_sessions.discard(session_id)

    def log_conversation_entry(self, session_id: str, user_message: str, assistant_response: str):
        """Simplified logging of conversation entry for debugging"""
        from datetime import datetime
        conversation_entry = {
            "user_message": {"role": "user", "content": user_message, "timestamp": datetime.utcnow().isoformat()},
            "assistant_response": {"role": "assistant", "content": assistant_response, "timestamp": datetime.utcnow().isoformat()},
            "session_id": session_id
        }
        
        session_data = self.get_session_data(session_id)
        total_messages = len(session_data.get("conversation_history", []))
        logger.info(f"[CONVERSATION_ENTRY] Logged interaction for session {session_id}: {total_messages} total messages")
        
        return conversation_entry

    def _format_conversation_for_log(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for readable logging"""
        if not conversation_history:
            return "No conversation history"
        
        formatted = []
        formatted.append("=" * 50)
        formatted.append("CONVERSATION HISTORY")
        formatted.append("=" * 50)
        
        for i, msg in enumerate(conversation_history, 1):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]  # Limit content length for logs
            timestamp = msg.get('timestamp', 'no-timestamp')
            
            # Simple markdown indicator
            has_markdown = any(marker in content for marker in ["**", "*", "`", "```"])
            markdown_indicator = " [MD]" if has_markdown else ""
            
            formatted.append(f"{i}. {role.upper()}{markdown_indicator}: {content}")
            formatted.append("-" * 40)
        
        formatted.append("=" * 50)
        return "\n".join(formatted)

    def format_conversation_history(self, conversation_history):
        """Format conversation history for LLM prompts while preserving markdown"""
        formatted = ""
        for message in conversation_history:
            role = message.get("role", "")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            
            # Check for markdown content and log it
            has_markdown = any(marker in content for marker in ["**", "*", "_", "#", "-", "`", "```"])
            if has_markdown:
                logger.info(f"[MARKDOWN_PRESERVE] Formatting message with markdown: {role} - {content[:100]}...")
            
            # Preserve markdown formatting by using proper delimiters and spacing
            if role.lower() == "user":
                formatted += f"USER: {content}\n\n"
            elif role.lower() == "assistant":
                # Preserve assistant markdown by not interfering with formatting
                formatted += f"ASSISTANT: {content}\n\n"
            else:
                formatted += f"{role.upper()}: {content}\n\n"
        
        # Log final formatted result if it contains markdown
        if any(marker in formatted for marker in ["**", "*", "_", "#", "-", "`", "```"]):
            logger.info(f"[MARKDOWN_PRESERVE] Final formatted conversation contains markdown: {len(formatted)} chars")
        
        return formatted.strip()

    def check_requirements(self, session_id):
        """Enhanced requirements checking with smarter conversation analysis"""
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        conversation_history = session_data.get("conversation_history", [])

        user_messages = [
            msg for msg in conversation_history if msg.get("role") == "user"
        ]
        meaningful_exchanges = len(
            [msg for msg in user_messages if len(msg.get("content", "").strip()) > 10]
        )
        if meaningful_exchanges <= 1:
            logger.debug(
                "[check_requirements] Initial question - continue with follow-ups"
            )
            return False

        if 2 <= meaningful_exchanges <= 3:
            logger.debug(
                f"[check_requirements] Early conversation ({meaningful_exchanges} exchanges) - continue with follow-ups"
            )
            return False

        if 4 <= meaningful_exchanges <= 5:
            recent_conversation = self.format_conversation_history(
                conversation_history[-6:]
            )

            assessment_prompt = assesment_prompt(recent_conversation=recent_conversation, prompt_context=prompt_context)

            messages = [
                {"role": "system", "content": SHARED_SYSTEM_PROMPT},
                {"role": "user", "content": assessment_prompt},
            ]

            evaluation_raw = generate_llm_response(messages)
            evaluation = evaluation_raw.strip().upper() if isinstance(evaluation_raw, str) else ""
            is_complete = "COMPLETE" in evaluation

            logger.debug(
                f"[check_requirements] LLM evaluation after {meaningful_exchanges} exchanges: {evaluation}"
            )
            logger.debug(
                f"[check_requirements] Will use {'comprehensive response' if is_complete else 'optimized response'}"
            )
            return is_complete
        if meaningful_exchanges >= 6:
            logger.debug(
                f"[check_requirements] Extended conversation ({meaningful_exchanges} exchanges) - forcing completion"
            )
            return True

        return False

    def get_session_data(self, session_id):
        """
        Retrieve session data for a given session_id.
        Return a default structure if the session does not exist.
        """
        return self.sessions.get(
            session_id,
            {
                "prompt_context": "",
                "conversation_history": [],
                "state": {},
                "answers": {},
            },
        )

    def set_session_data(self, session_id, session_data: Dict):
        """Persist the given session_data for session_id into the in-memory store.

        This centralizes session writes so callers don't accidentally forget to
        assign back to self.sessions after mutating the dict.
        """
        self.sessions[session_id] = session_data
        return session_data

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a session.
        Returns a list of message dictionaries with role and content.
        """
        session_data = self.get_session_data(session_id)
        return session_data.get("conversation_history", [])

    def cleanup_inactive_sessions(self, max_sessions: int = 1000):
        """
        Clean up inactive sessions from memory to prevent memory bloat.
        Keeps the most recently used sessions up to max_sessions.
        """
        if len(self.sessions) <= max_sessions:
            return
        
        # Sort sessions by last activity (we'll use session creation order as proxy)
        session_items = list(self.sessions.items())
        # Keep the most recent max_sessions
        sessions_to_keep = dict(session_items[-max_sessions:])
        
        removed_count = len(self.sessions) - len(sessions_to_keep)
        self.sessions = sessions_to_keep
        
        # Also clean up tracking sets
        active_session_ids = set(sessions_to_keep.keys())
        self._db_loaded_sessions &= active_session_ids
        self._processing_sessions &= active_session_ids
        
        logger.info(f"[CLEANUP] Removed {removed_count} inactive sessions from memory. Active sessions: {len(self.sessions)}")

    def should_generate_followups(self, session_id: str) -> bool:
        """Check if we should generate new follow-ups for this session to prevent duplicates"""
        current_time = time.time()
        
        # Don't generate follow-ups if we generated them recently (within 30 seconds)
        if session_id in self._last_followup_generation:
            time_since_last = current_time - self._last_followup_generation[session_id]
            if time_since_last < 30:  # 30 seconds cooldown
                logger.info(f"[FOLLOWUP_COOLDOWN] Skipping follow-up generation for session {session_id} - {time_since_last:.1f}s since last generation")
                return False
        
        return True
        
    def mark_followup_generated(self, session_id: str):
        """Mark that follow-ups were generated for this session"""
        self._last_followup_generation[session_id] = time.time()
        
    def get_session_stats(self) -> Dict[str, int]:
        """Get statistics about current session state for monitoring"""
        return {
            "total_sessions": len(self.sessions),
            "db_loaded_sessions": len(self._db_loaded_sessions),
            "processing_sessions": len(self._processing_sessions),
            "total_messages": sum(len(session.get("conversation_history", [])) for session in self.sessions.values())
        }

