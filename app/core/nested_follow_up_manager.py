from typing import List, Dict
from app.logger import get_logger
from app.core.services.chatbot_optimizer import OptimizedChatbot
from app.core.utils import generate_llm_response  
from app.core.prompts import SHARED_SYSTEM_PROMPT, assesment_prompt
logger = get_logger(__name__)


class FollowUpManager:
	def __init__(self, llm):
		self.llm = llm
		self.sessions = {}  
		self.chatbot = OptimizedChatbot(llm=llm)
		self._processing_sessions = set()  # Track sessions being processed  

	def resolve_history(self, session_id: str, incoming_history: List[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
		"""
		Return the most appropriate conversation history to use for processing.
		Priority:
		1. incoming_history if provided and non-empty
		2. server-side stored session history
		3. empty list
		This centralizes the fallback logic so callers can always pass a
		`conversation_history` and expect consistent behavior.
		"""
		try:
			resolved_history = []
			if incoming_history:
				resolved_history = incoming_history
				logger.info(f"[RESOLVE_HISTORY] Using incoming_history for session {session_id}: {len(resolved_history)} messages")
			else:
				session = self.get_session_data(session_id)
				if session:
					resolved_history = session.get("conversation_history", []) or []
					logger.info(f"[RESOLVE_HISTORY] Using stored session history for session {session_id}: {len(resolved_history)} messages")
			
			# Log the complete resolved history
			logger.info(f"[RESOLVE_HISTORY] Complete resolved conversation history for session {session_id}: {resolved_history}")
			return resolved_history
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
		"""Add message to conversation history with deduplication"""
		from datetime import datetime
		
		# Check if session is already being processed
		if session_id in self._processing_sessions:
			logger.warning(f"[CONCURRENT_ACCESS] Session {session_id} is already being processed, queuing message: {role} - {content[:50]}...")
		
		self._processing_sessions.add(session_id)
		
		try:
			# Debug: Check if markdown is preserved in input
			has_markdown = any(marker in content for marker in ["**", "*", "_", "#", "-", "`", "```", "\n"])
			if role == "assistant" and has_markdown:
				logger.info(f"[MARKDOWN_DEBUG] Adding assistant message with markdown: {content[:150]}...")
				markdown_count = content.count("**") + content.count("*") + content.count("#") + content.count("-")
				logger.info(f"[MARKDOWN_DEBUG] Markdown elements count: {markdown_count}")
			
			session_data = self.get_session_data(session_id)
			
			# Check for duplicate messages (same role and content in last few messages)
			existing_history = session_data.get("conversation_history", [])
			if existing_history:
				# Check last 2 messages for duplicates
				for recent_msg in existing_history[-2:]:
					if (recent_msg.get("role") == role and 
						recent_msg.get("content") == content):
						logger.warning(f"[DUPLICATE_PREVENTION] Skipping duplicate message for session {session_id}: {role} - {content[:100]}...")
						return
			
			message_data = {
				"role": role, 
				"content": content,  # Ensure content is not modified during storage
				"timestamp": datetime.utcnow().isoformat()
			}
			session_data["conversation_history"].append(message_data)
			self.sessions[session_id] = session_data
			
			# Debug: Verify markdown is still there after storage
			if role == "assistant" and has_markdown:
				stored_content = message_data["content"]
				stored_markdown_count = stored_content.count("**") + stored_content.count("*") + stored_content.count("#") + stored_content.count("-")
				logger.info(f"[MARKDOWN_DEBUG] Stored assistant message with markdown preserved: {stored_content[:150]}...")
				logger.info(f"[MARKDOWN_DEBUG] Stored markdown elements count: {stored_markdown_count}")
				
				# Check if markdown was preserved
				if stored_markdown_count != markdown_count:
					logger.warning(f"[MARKDOWN_WARNING] Markdown count changed during storage! Original: {markdown_count}, Stored: {stored_markdown_count}")
			
			# Add stack trace info to identify the caller
			import traceback
			caller_info = traceback.format_stack()[-3:-1]  # Get the calling function info
			logger.info(f"[ADD_MESSAGE] Message added to conversation history for session {session_id}: {role} - {content[:100]}...")
			logger.info(f"[ADD_MESSAGE] Called from: {caller_info}")
			
			# Log complete conversation history after each addition
			complete_history = session_data.get("conversation_history", [])
			logger.info(f"[COMPLETE_SESSION_HISTORY] Session {session_id} now has {len(complete_history)} messages:")
			logger.info(f"[COMPLETE_SESSION_HISTORY] Full conversation: {complete_history}")
			
		finally:
			# Always remove from processing set
			self._processing_sessions.discard(session_id)

	def log_conversation_entry(self, session_id: str, user_message: str, assistant_response: str):
		"""Log both user and assistant messages together for confirmation"""
		from datetime import datetime
		conversation_entry = {
			"user_message": {
				"role": "user", 
				"content": user_message, 
				"timestamp": datetime.utcnow().isoformat()
			},
			"assistant_response": {
				"role": "assistant", 
				"content": assistant_response, 
				"timestamp": datetime.utcnow().isoformat()
			},
			"session_id": session_id
		}
		logger.info(f"[CONVERSATION_ENTRY] Single interaction logged: {conversation_entry}")
		
		# Also log the complete session state after this interaction
		session_data = self.get_session_data(session_id)
		complete_history = session_data.get("conversation_history", [])
		logger.info(f"[CONVERSATION_ENTRY] Complete session state after interaction - Session {session_id} total messages: {len(complete_history)}")
		logger.info(f"[CONVERSATION_ENTRY] Full session conversation history: {complete_history}")
		
		# Log formatted conversation for readability
		formatted_conversation = self._format_conversation_for_log(complete_history)
		logger.info(f"[CONVERSATION_ENTRY] Formatted conversation view:\n{formatted_conversation}")
		return conversation_entry

	def _format_conversation_for_log(self, conversation_history: List[Dict[str, str]]) -> str:
		"""Format conversation history for readable logging while preserving markdown"""
		if not conversation_history:
			return "No conversation history"
		
		formatted = []
		formatted.append("=" * 50)
		formatted.append("COMPLETE CONVERSATION HISTORY")
		formatted.append("=" * 50)
		
		for i, msg in enumerate(conversation_history, 1):
			role = msg.get('role', 'unknown')
			content = msg.get('content', '')
			timestamp = msg.get('timestamp', 'no-timestamp')
			
			# Check if message contains markdown and log it
			has_markdown = any(marker in content for marker in ["**", "*", "_", "#", "-", "`", "```"])
			markdown_indicator = " [MARKDOWN]" if has_markdown else ""
			
			formatted.append(f"{i}. {role.upper()}{markdown_indicator} [{timestamp}]: {content}")
			formatted.append("-" * 40)
		
		formatted.append("=" * 50)
		
		# Log summary of markdown preservation
		total_markdown_msgs = sum(1 for msg in conversation_history 
								 if any(marker in msg.get('content', '') for marker in ["**", "*", "_", "#", "-", "`", "```"]))
		if total_markdown_msgs > 0:
			formatted.append(f"MARKDOWN SUMMARY: {total_markdown_msgs} messages contain markdown formatting")
		
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

