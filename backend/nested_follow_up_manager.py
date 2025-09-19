from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging
from backend.services.chatbot_optimizer import OptimizedChatbot
from backend.utils import generate_llm_response  # Import from utils package

logger = logging.getLogger(__name__)

class FollowUpManager:
    def __init__(self, llm):
        self.llm = llm
        self.sessions = {}  # Existing session storage
        self.chatbot = OptimizedChatbot(llm=llm)  # Initialize optimized chatbot
    
    # NEW METHODS TO ADD
    def initialize_session(self, session_id, prompt_id, prompt_context):
        """Initialize session with prompt context and conversation history"""
        self.sessions[session_id] = {
            "prompt_id": prompt_id,
            "prompt_context": prompt_context,
            "conversation_history": [],
            # Keep existing fields if any
            "state": {},
            "answers": {}
        }
    
    def add_to_conversation_history(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        session_data = self.get_session_data(session_id)
        session_data["conversation_history"].append({
            "role": role,
            "content": content
        })
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
        """Check if requirements are met based on prompt context and conversation"""
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        conversation_history = session_data.get("conversation_history", [])

        # Minimum requirements check
        if len(conversation_history) < 2:  # At least one user response needed
            logger.debug("[check_requirements] Not enough conversation history")
            return False

        # Create messages for requirement evaluation
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant that evaluates conversation completeness.
                            You must determine if enough information has been gathered to provide a comprehensive response.
                            Consider: context understanding, specific details, and clarity of requirements.
                            Respond ONLY with 'YES' or 'NO'."""
            },
            {
                "role": "user",
                "content": f"""Evaluate if we have gathered sufficient information:

                Original Context: {prompt_context}

                Conversation History:
                {self.format_conversation_history(conversation_history)}

                Requirements Analysis:
                1. Is the user's primary need clearly understood? 
                2. Have all necessary details been provided?
                3. Are there any critical gaps in information?
                4. Can we provide a specific, actionable response?

                Based on these criteria, should we:
                - Return 'YES' if we can now provide a complete, accurate response
                - Return 'NO' if we need more specific information

                Response (YES/NO only):"""
            }
        ]

        logger.debug(f"[check_requirements] Evaluating completion with {len(conversation_history)} messages")
        
        # Get LLM evaluation
        evaluation = generate_llm_response(messages).strip().upper()
        is_complete = evaluation == "YES"
        
        logger.debug(f"[check_requirements] Requirements met: {is_complete}")
        
        if is_complete:
            # Update session state to mark completion
            self.sessions[session_id]["state"]["requirements_met"] = True
            
        return is_complete
    
    async def generate_next_follow_up(self, session_id):
        """Generate a follow-up question using streaming response"""
        from backend.chat_logic import build_chatbot_response
        
        conversation_history = self.get_conversation_history(session_id)
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        
        response = None
        async for message in build_chatbot_response(
            session_id=session_id,
            follow_up_manager=self,
            conversation_history=conversation_history,
            prompt_context=prompt_context,
            mode="follow_up"
        ):
            if message and "content" in json.loads(message.split("data: ")[1]):
                response = json.loads(message.split("data: ")[1])["content"]
        
        return response
    
    def get_session_data(self, session_id):
        """
        Retrieve session data for a given session_id.
        Return a default structure if the session does not exist.
        """
        return self.sessions.get(session_id, {
            "prompt_context": "",
            "conversation_history": [],
            "state": {},
            "answers": {}
        })
        
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a session.
        Returns a list of message dictionaries with role and content.
        """
        session_data = self.get_session_data(session_id)
        return session_data.get("conversation_history", [])
        
    # async def generate_complete_response(self, session_id: str, conversation_history: Optional[List[Dict[str, Any]]] = None):
    #     """Generate a complete response using streaming response.

    #     Accepts optional conversation_history (for callers that already fetched it)
    #     to avoid duplicative lookups. Falls back to internal storage if not provided.
    #     """
    #     from backend.chat_logic import build_chatbot_response

    #     if conversation_history is None:
    #         conversation_history = self.get_conversation_history(session_id)
    #     session_data = self.get_session_data(session_id)
    #     prompt_context = session_data.get("prompt_context", "")

    #     complete_response = ""
    #     async for message in build_chatbot_response(
    #         session_id=session_id,
    #         follow_up_manager=self,
    #         conversation_history=conversation_history,
    #         prompt_context=prompt_context,
    #         mode="complete"
    #     ):
    #         try:
    #             if "data: " in message:
    #                 payload = message.split("data: ", 1)[1]
    #             else:
    #                 payload = message
    #             message_data = json.loads(payload)
    #         except Exception:
    #             continue
    #         if message_data.get("content"):
    #             complete_response = message_data["content"]

    #     return complete_response

    # async def generate_suggestions(self, session_id: str) -> List[str]:
    #     """Return suggestion follow-up questions (pass-through to chatbot).

    #     Provides a graceful fallback to an empty list if underlying generation fails.
    #     """
    #     try:
    #         return await self.chatbot.generate_suggestions(session_id)
    #     except Exception as e:
    #         logger.error(f"[generate_suggestions] Failed: {e}")
    #         return []

