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
			if incoming_history:
				return incoming_history
			session = self.get_session_data(session_id)
			if session:
				return session.get("conversation_history", []) or []
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
		"""Add message to conversation history"""
		session_data = self.get_session_data(session_id)
		session_data["conversation_history"].append({"role": role, "content": content})
		self.sessions[session_id] = session_data

	def format_conversation_history(self, conversation_history):
		"""Format conversation history for LLM prompts"""
		formatted = ""
		for message in conversation_history:
			role = message.get("role", "")
			content = message.get("content", "")

			formatted += f"{role}: {content}\n"
		return formatted

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

