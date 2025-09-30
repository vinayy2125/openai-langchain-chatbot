from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import logging
import re
from app.core.services.chatbot_optimizer import OptimizedChatbot
from app.core.utils import generate_llm_response  # Import from utils package

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

        # Count meaningful exchanges (user messages that aren't just acknowledgments)
        user_messages = [
            msg for msg in conversation_history if msg.get("role") == "user"
        ]
        meaningful_exchanges = len(
            [msg for msg in user_messages if len(msg.get("content", "").strip()) > 10]
        )

        # For initial questions, always continue with follow-ups
        if meaningful_exchanges <= 1:
            logger.debug(
                "[check_requirements] Initial question - continue with follow-ups"
            )
            return False

        # FIX: Be more conservative with early completion to ensure structured follow-ups
        # For 2-3 exchanges, continue follow-ups to gather more information
        if 2 <= meaningful_exchanges <= 3:
            logger.debug(
                f"[check_requirements] Early conversation ({meaningful_exchanges} exchanges) - continue with follow-ups"
            )
            return False

        # For 4-5 exchanges, use intelligent assessment but with stricter criteria
        if 4 <= meaningful_exchanges <= 5:
            # Use LLM to assess if we have enough information, but be more conservative
            recent_conversation = self.format_conversation_history(
                conversation_history[-6:]
            )

            assessment_prompt = f"""Analyze this conversation to determine if we have sufficient information to provide a useful response.

Original Context: {prompt_context}

Recent Conversation:
{recent_conversation}

Evaluation Criteria:
1. Can we understand the main points of what the user wants?
2. Do we have enough context to provide a helpful response?
3. Can we offer actionable guidance based on what we know?

Rules:
- Mark COMPLETE if we can provide a meaningful, focused response
- Mark COMPLETE if we have 2-3 clear points to address
- Mark COMPLETE if the user has expressed their core needs
- Mark CONTINUE only if we're missing critical information
- When in doubt and we have sufficient context, mark COMPLETE

Respond with ONLY: COMPLETE or CONTINUE"""

            messages = [
                {
                    "role": "system",
                    "content": "You are a conservative evaluator. Only mark conversations COMPLETE when you have comprehensive information for a detailed response.",
                },
                {"role": "user", "content": assessment_prompt},
            ]

            evaluation = generate_llm_response(messages).strip().upper()
            is_complete = "COMPLETE" in evaluation

            logger.debug(
                f"[check_requirements] LLM evaluation after {meaningful_exchanges} exchanges: {evaluation}"
            )
            logger.debug(
                f"[check_requirements] Will use {'comprehensive response' if is_complete else 'optimized response'}"
            )
            return is_complete

        # After 6+ exchanges, force completion to avoid infinite loops
        if meaningful_exchanges >= 6:
            logger.debug(
                f"[check_requirements] Extended conversation ({meaningful_exchanges} exchanges) - forcing completion"
            )
            return True

        return False

    def generate_comprehensive_response(self, session_id: str) -> str:
        """Generate a comprehensive final response when requirements are complete"""
        session_data = self.get_session_data(session_id)
        prompt_context = session_data.get("prompt_context", "")
        conversation_history = session_data.get("conversation_history", [])

        # Build a comprehensive prompt for final response
        conversation_summary = self.format_conversation_history(conversation_history)

        comprehensive_prompt = f"""Based on our conversation, provide a precise, well-structured response (~200 words) that directly addresses the user’s needs, incorporates all relevant context, and references available knowledge base data.

Original Context: {prompt_context}  
Full Conversation: {conversation_summary}  

FORMATTING INSTRUCTIONS:  
1. Limit the response to around **300 words**, concise, informative, and focused.  
2. Use **bold text** to highlight important terms, technologies, and key concepts.  
3. Organize content with ### headings (e.g., “### Summary”, “### Key Points”, “### Recommendations”, “### Next Steps”).  
4. Use short paragraphs (2–3 lines) for readability.  
5. Include **bullet points or numbered lists** where helpful to highlight key information.  
6. Maintain a natural, conversational flow similar to ChatGPT responses.  
7. Conclude with a **friendly closing/thank you message**, such as:  
   “Hope you are satisfied with the provided inputs and solutions. Feel free to ask further questions if needed.”  
8. Use Markdown formatting naturally (bold, lists, headings) for clarity.  

CONTENT REQUIREMENTS:  
- Provide a clear summary addressing the original question.  
- Include relevant insights from the chat context and knowledge base.  
- Offer actionable recommendations or next steps.  
- Keep the response practical, concise, and easy to follow.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that provides concise (200 words max), focused responses with minimal formatting. Use only bullet points and bold text where absolutely necessary. Avoid unnecessary headers or complex formatting.",
            },
            {"role": "user", "content": comprehensive_prompt},
        ]

        try:
            response = generate_llm_response(messages)
            logger.debug(
                f"[generate_comprehensive_response] Generated comprehensive response of {len(response)} characters"
            )
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
        conversation_summary = self.format_conversation_history(
            conversation_history[-4:]
        )  # Last 4 messages for context

        suggestion_prompt = f"""Based on this conversation, generate a single concise and actionable suggestion or recommendation.

Original Context: {prompt_context}  
Additional Context: {context}  

Recent Conversation:  
{conversation_summary}  

FORMATTING INSTRUCTIONS:  
1. Provide exactly **one suggestion** in 1–2 sentences.  
2. Use **bold text** for key terms and technologies.  
3. Keep it practical, relevant, and easy to apply.  
4. Avoid numbering or excessive formatting.  

Examples:  
- **Improve performance** by implementing lazy loading for heavy assets.  
- **Strengthen security** with role-based access control and regular audits.  
"""

        messages = [
            {
                "role": "system",
                "content": "You generate practical, actionable suggestions based on conversation context.",
            },
            {"role": "user", "content": suggestion_prompt},
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

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a session.
        Returns a list of message dictionaries with role and content.
        """
        session_data = self.get_session_data(session_id)
        return session_data.get("conversation_history", [])

    def generate_follow_ups(
        self,
        session_id: str,
        latest_query: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[str]:
        """Generate a single, focused follow-up based on session data, latest query, and optional context."""
        session_data = self.get_session_data(session_id)
        conversation_history = session_data.get("conversation_history", [])
        prompt_context = session_data.get("prompt_context", "")

        # Build follow-up prompt
        conversation_summary = self.format_conversation_history(
            conversation_history[-4:]
        )  # Last 4 messages for context

        follow_up_prompt = f"""Based on this conversation, generate a single dynamic follow-up to guide the user and gather more information.

Original Context: {prompt_context}  
Latest Query: {latest_query if latest_query else 'N/A'}  
Additional Context: {context if context else 'N/A'}  

Recent Conversation:  
{conversation_summary}  

FORMATTING INSTRUCTIONS:  
1. Generate exactly **one follow-up** that feels natural and conversational.  
2. Keep it precise, context-aware, and helpful — like ChatGPT’s follow-up style.  
3. The follow-up can be:  
   - A clarifying question to better understand user needs, OR  
   - A helpful prompt suggesting the most logical next step.  
4. Ensure it moves the conversation forward and avoids redundancy.  

Examples:  
- Could you share more details about your integration setup?  
- Would you like me to walk you through optimizing the deployment process?  
- Should I suggest best practices for your current approach?  
"""

        messages = [
            {
                "role": "system",
                "content": "You generate dynamic, open-ended follow-up questions based on conversation context.",
            },
            {"role": "user", "content": follow_up_prompt},
        ]

        try:
            response = generate_llm_response(messages)
            logger.debug(f"[generate_follow_ups] Generated follow-ups: {response}")
            return response.split("\n")  # Split follow-ups into list
        except Exception as e:
            logger.error(f"[generate_follow_ups] Failed: {e}")
            return ["Could you provide more details?"]  # Single fallback follow-up
        session_data = self.get_session_data(session_id)
        conversation_history = session_data.get("conversation_history", [])
        prompt_context = session_data.get("prompt_context", "")

        # Build follow-up prompt
        conversation_summary = self.format_conversation_history(
            conversation_history[-4:]
        )  # Last 4 messages for context

        follow_up_prompt = f"""Based on this conversation, generate dynamic follow-up questions to gather more information.

Original Context: {prompt_context}
Additional Context: {context if context else 'N/A'}

Recent Conversation:
{conversation_summary}

FORMATTING INSTRUCTIONS:
1. Provide concise, open-ended follow-up questions.
2. Ensure questions are relevant to the context and conversation.
3. Avoid yes/no questions; focus on gathering detailed responses.

Example:
- Can you elaborate on your requirements for scalability?
- What specific features are you looking for in the solution?"""

        messages = [
            {
                "role": "system",
                "content": "You generate dynamic, open-ended follow-up questions based on conversation context.",
            },
            {"role": "user", "content": follow_up_prompt},
        ]

        try:
            response = generate_llm_response(messages)
            logger.debug(f"[generate_follow_ups] Generated follow-ups: {response}")
            return response.split("\n")  # Split follow-ups into list
        except Exception as e:
            logger.error(f"[generate_follow_ups] Failed: {e}")
            return ["Could you provide more details?"]  # Single fallback follow-up
