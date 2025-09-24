from backend.llm_client import llm
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def generate_llm_response(prompt):
    """
    Handles both string prompts and list of messages.
    Converts inputs into proper LangChain message objects.
    """
    try:
        # Convert input into LangChain message objects
        if isinstance(prompt, list):
            messages = []
            has_system_message = False
            for message in prompt:
                if not isinstance(message, dict):
                    logger.error(f"Invalid message format (not a dict): {message}")
                    return "Invalid message format."
                if "role" not in message or "content" not in message:
                    logger.error(f"Invalid message keys: {message}")
                    return "Invalid message keys."
                
                role = message["role"]
                content = message["content"]
                if role == "system":
                    has_system_message = True
                
                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    logger.error(f"Invalid role type: {role}")
                    return "Invalid role type."
            
            # Only add default system message if none was provided
            if not has_system_message:
                messages.insert(0, SystemMessage(content="You are an AI assistant that provides direct, concise responses."))
        else:
            # Handle single string prompt with default system message
            messages = [
                SystemMessage(content="You are an AI assistant that provides direct, concise responses."),
                HumanMessage(content=prompt.strip())
            ]

        logger.debug(f"Messages sent to LLM: {messages}")
        
        # Use invoke instead of generate for ChatOpenAI
        response = llm.invoke(messages)
        
        # Extract content from response
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
            
        logger.debug(f"Response from LLM: {result}")
        return result.strip()
        
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}")
        logger.error(f"Messages that caused the error: {messages if 'messages' in locals() else 'No messages created'}")
        return "Failed to generate response. Ensure the prompt is valid."
