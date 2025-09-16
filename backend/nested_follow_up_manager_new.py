from typing import Dict, Any, Optional
import json
from backend.api_v1.models import FollowUp, SessionState
import logging

logger = logging.getLogger(__name__)

async def generate_contextual_follow_up(
    self,
    current_context: str,
    gathered_requirements: Dict[str, Any],
    user_response: str
) -> Optional[FollowUp]:
    """
    Generate a contextual follow-up question based on current context and gathered requirements.
    """
    try:
        # Create prompt for generating contextual follow-up
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant generating follow-up questions.
                            Analyze the context and previous responses to determine the next most relevant question.
                            Your response should be a JSON object containing:
                            {
                                "question": "The follow-up question",
                                "type": "yes_no|nested|expansion|clarification",
                                "context": "Why this question is being asked",
                                "context_key": "requirement_key",
                                "options": ["option1", "option2"] # Only for nested type
                                "required": true/false
                            }"""
            },
            {
                "role": "user",
                "content": f"""Generate a focused follow-up question based on:

                Current Context: {current_context}
                
                Gathered Requirements: {json.dumps(gathered_requirements, indent=2)}
                
                Latest Response: {user_response}

                Consider:
                1. What critical information is still missing?
                2. What needs clarification?
                3. Are there any inconsistencies to resolve?
                4. What details would help provide a better response?

                Return a single well-formed JSON object for the next question:"""
            }
        ]

        # Get LLM response
        response = await self.llm.async_generate(messages, max_tokens=500)
        
        # Parse the response into a FollowUp object
        follow_up_data = json.loads(response)
        return FollowUp(**follow_up_data)

    except Exception as e:
        logger.error(f"Error generating follow-up: {str(e)}")
        return None

def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data including conversation history and state"""
    return self.sessions.get(session_id)

def update_session_state(self, session_id: str, session_state: SessionState):
    """Update the session state with new information"""
    if session_id in self.sessions:
        self.sessions[session_id]["session_state"] = session_state

def validate_response(self, follow_up: FollowUp, response: str) -> bool:
    """Validate user response against follow-up validation rules"""
    if not follow_up.validation_rules:
        return True
        
    try:
        rules = follow_up.validation_rules
        
        # Basic type validation
        if rules.get("type") == "number":
            try:
                float(response)
                return True
            except ValueError:
                return False
                
        # Options validation
        if rules.get("options"):
            return response.lower() in [opt.lower() for opt in rules["options"]]
            
        # Pattern validation
        if rules.get("pattern"):
            import re
            return bool(re.match(rules["pattern"], response))
            
        # Length validation
        min_length = rules.get("minLength", 0)
        max_length = rules.get("maxLength", float("inf"))
        response_length = len(response)
        return min_length <= response_length <= max_length
        
    except Exception as e:
        logger.error(f"Error in response validation: {str(e)}")
        return False
