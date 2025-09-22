from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging
import re
from backend.services.chatbot_optimizer_new import OptimizedChatbot
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
        """Enhanced requirements checking with smarter conversation analysis"""
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        conversation_history = session_data.get("conversation_history", [])

        # Count meaningful exchanges (user messages that aren't just acknowledgments)
        user_messages = [msg for msg in conversation_history if msg.get("role") == "user"]
        meaningful_exchanges = len([msg for msg in user_messages if len(msg.get("content", "").strip()) > 10])
        
        # For initial questions, always continue with follow-ups
        if meaningful_exchanges <= 1:
            logger.debug("[check_requirements] Initial question - continue with follow-ups")
            return False
            
        # If we have 2-4 meaningful exchanges, use intelligent assessment
        if 2 <= meaningful_exchanges <= 4:
            # Use LLM to assess if we have enough information
            recent_conversation = self.format_conversation_history(conversation_history[-6:])
            
            assessment_prompt = f"""Analyze this conversation to determine if we have sufficient information to provide a comprehensive response.

Original Context: {prompt_context}

Recent Conversation:
{recent_conversation}

Evaluation Criteria:
1. Is the user's main need clearly understood?
2. Have we gathered key requirements/preferences?
3. Can we provide actionable recommendations now?
4. Are there obvious gaps that need clarification?

Rules:
- If this is a general question that can be answered immediately: COMPLETE
- If we need 1-2 more specific details for a good response: CONTINUE  
- If the user seems satisfied with current info: COMPLETE
- If we're just starting to understand their needs: CONTINUE

Respond with ONLY: COMPLETE or CONTINUE"""

            messages = [
                {"role": "system", "content": "You evaluate conversation completeness for optimal user experience."},
                {"role": "user", "content": assessment_prompt}
            ]
            
            evaluation = generate_llm_response(messages).strip().upper()
            is_complete = "COMPLETE" in evaluation
            
            logger.debug(f"[check_requirements] LLM evaluation ({meaningful_exchanges} exchanges): {evaluation} -> {'Complete' if is_complete else 'Continue'}")
            return is_complete
            
        # After 5+ exchanges, start leaning toward completion
        if meaningful_exchanges >= 5:
            logger.debug(f"[check_requirements] Extended conversation ({meaningful_exchanges} exchanges) - forcing completion")
            return True

        return False

    def generate_comprehensive_response(self, session_id: str) -> str:
        """Generate a comprehensive final response when requirements are complete"""
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        conversation_history = session_data.get("conversation_history", [])

        # Build a comprehensive prompt for final response
        conversation_summary = self.format_conversation_history(conversation_history)

        comprehensive_prompt = f"""Based on our conversation, provide a comprehensive, well-formatted response that addresses the user's needs.

Original Context: {prompt_context}

Full Conversation:
{conversation_summary}

FORMATTING INSTRUCTIONS:
1. Provide a thorough, well-structured response (3-5 paragraphs)
2. Use **bold text** for key concepts, technologies, and important terms
3. Use ONLY ### headings for sections - vary the heading text:
   - ### Implementation Strategy
   - ### Key Considerations
   - ### Recommended Approach
   - ### Technical Overview
   - ### Next Steps
   - ### Important Notes
4. Include specific recommendations with **bold highlights**
5. Address all key points discussed in the conversation
6. Make it actionable and practical with **clear next steps**
7. Use natural paragraph breaks for readability
8. Emphasize important frameworks, tools, or concepts with **bold**
9. Structure with clear Markdown sections if helpful
10. Conclude with **highlighted** next steps or recommendations
11. Use Markdown formatting naturally (lists, bold, italics, etc.)
12. Vary heading phrases to avoid repetition

CONTENT REQUIREMENTS:
- Comprehensive answer addressing the original question
- Specific recommendations based on gathered requirements
- Technical details with **bold keywords** for clarity
- Actionable next steps with **emphasis**
- Practical implementation guidance
- Use Markdown headings and formatting for structure

Create a complete, well-formatted response that would satisfy the user's original question while incorporating all the information gathered through our conversation.

Use Markdown formatting effectively:
- **Bold** for key concepts and technologies
- ### headings only for major sections (vary the phrases)
- Natural paragraph breaks for readability
- Emphasis where appropriate

Example structure:
**Technology/Solution** provides [comprehensive answer].

[Supporting details with **key points** highlighted]

### Implementation Strategy
[Specific recommendations based on requirements]

### Next Steps
- [Action item]
- [Action item]"""

        messages = [
            {"role": "system", "content": "You are an expert assistant providing comprehensive, well-formatted responses with proper Markdown formatting for clear presentation."},
            {"role": "user", "content": comprehensive_prompt}
        ]
        
        try:
            response = generate_llm_response(messages)
            logger.debug(f"[generate_comprehensive_response] Generated comprehensive response of {len(response)} characters")
            return response
        except Exception as e:
            logger.error(f"[generate_comprehensive_response] Failed: {e}")
            return "I apologize, but I encountered an issue generating a comprehensive response. Please try rephrasing your question."

    def generate_suggestions(self, session_id: str, context: str = "") -> List[str]:
        """Generate actionable suggestions based on conversation and context"""
        session_data = self.get_session_data(session_id)
        conversation_history = session_data.get("conversation_history", [])
        prompt_context = session_data.get("prompt_context", "")

        # Build suggestion prompt
        conversation_summary = self.format_conversation_history(conversation_history[-4:])  # Last 4 messages for context

        suggestion_prompt = f"""Based on this conversation, generate actionable suggestions or recommendations.

Original Context: {prompt_context}
Additional Context: {context}

Recent Conversation:
{conversation_summary}

FORMATTING INSTRUCTIONS:
1. Provide concise, actionable suggestions (1-2 sentences each).
2. Use **bold text** for key terms and technologies.
3. Structure suggestions with bullet points for clarity.
4. Ensure suggestions are practical and relevant.
5. Avoid numbering or excessive formatting.

Example:
- **Optimize front-end performance** by implementing lazy loading and code minification.
- **Enhance user experience** with responsive design and intuitive navigation.

Provide 2-3 actionable suggestions based on the conversation."""

        messages = [
            {"role": "system", "content": "You generate practical, actionable suggestions based on conversation context."},
            {"role": "user", "content": suggestion_prompt}
        ]

        try:
            response = generate_llm_response(messages)
            logger.debug(f"[generate_suggestions] Generated suggestions: {response}")
            return response.split("\n")  # Split suggestions into list
        except Exception as e:
            logger.error(f"[generate_suggestions] Failed: {e}")
            return ["Consider exploring related topics"]  # Single fallback suggestion
        
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
        
    def generate_comprehensive_response(self, session_id: str) -> str:
        """
        Generate a comprehensive response based on the conversation history and requirements.
        """
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        conversation_history = session_data.get("conversation_history", [])
        
        # Format the conversation history
        conversation_text = ""
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n"
        
        # Create comprehensive response prompt
        comprehensive_prompt = f"""Based on our conversation, provide a comprehensive, well-formatted response that addresses the user's needs.

Original Context: {prompt_context}

Full Conversation:
{conversation_text}

FORMATTING INSTRUCTIONS:
1. Provide a thorough, well-structured response (3-5 paragraphs)
2. Use **bold text** for key concepts, technologies, and important terms
3. Use ONLY ### headings for sections - vary the heading text:
   - ### Implementation Strategy
   - ### Key Considerations
   - ### Recommended Approach
   - ### Technical Overview
   - ### Next Steps
   - ### Important Notes
4. Include specific recommendations with **bold highlights**
5. Address all key points discussed in the conversation
6. Make it actionable and practical with **clear next steps**
7. Use natural paragraph breaks for readability
8. Emphasize important frameworks, tools, or concepts with **bold**
9. Structure with clear Markdown sections if helpful
10. Conclude with **highlighted** next steps or recommendations
11. Use Markdown formatting naturally (lists, bold, italics, etc.)
12. Vary heading phrases to avoid repetition

CONTENT REQUIREMENTS:
- Comprehensive answer addressing the original question
- Specific recommendations based on gathered requirements
- Technical details with **bold keywords** for clarity
- Actionable next steps with **emphasis**
- Practical implementation guidance
- Use Markdown headings and formatting for structure

Create a complete, well-formatted response that would satisfy the user's original question while incorporating all the information gathered through our conversation.

Use Markdown formatting effectively:
- **Bold** for key concepts and technologies
- ### headings only for major sections (vary the phrases)
- Natural paragraph breaks for readability
- Emphasis where appropriate

Example structure:
**Technology/Solution** provides [comprehensive answer].

[Supporting details with **key points** highlighted]

### Implementation Strategy
[Specific recommendations based on requirements]

### Next Steps
- [Action item]
- [Action item]"""

        # Generate the response using the LLM
        messages = [
            {"role": "system", "content": "You are an expert assistant providing comprehensive, well-formatted responses with proper Markdown formatting for clear presentation."},
            {"role": "user", "content": comprehensive_prompt}
        ]
        
        try:
            response = generate_llm_response(messages)
            return response
        except Exception as e:
            logger.error(f"Error generating comprehensive response: {e}")
            return f"**Summary:** Based on our conversation about {prompt_context}, here are the key recommendations we discussed. Please let me know if you'd like me to elaborate on any specific aspect."

    def generate_follow_ups(self, session_id: str, main_response: str) -> List[str]:
        """
        Generate follow-up questions or options based on the main response.

        Args:
            session_id: The session identifier
            main_response: The main response generated for the user query

        Returns:
            A list of follow-up questions or options
        """
        try:
            prompt = (
                f"Based on the following response, generate 3-5 follow-up questions or suggestions to continue the conversation:\n"
                f"Response:\n{main_response}\n\n"
                "Instructions:\n"
                "- Focus on clarifying user needs or providing actionable next steps\n"
                "- Ensure the follow-ups are concise and relevant\n"
                "- Format each follow-up as a single sentence"
            )

            # Use the LLM to generate follow-ups
            follow_up_response = self.llm.invoke(prompt)
            follow_ups = follow_up_response.split("\n")

            # Filter and clean up the follow-ups
            return [line.strip() for line in follow_ups if line.strip()]
        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return ["Could not generate follow-up questions at this time."]

    # Refactored response generation to use `chat_logic` and `chatbot_optimizer` methods for better flow and consistency.

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

