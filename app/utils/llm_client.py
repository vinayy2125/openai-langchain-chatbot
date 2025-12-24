import os
from app.logger import get_logger
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import SecretStr

# Configure logger
logger = get_logger(__name__)


load_dotenv()

# Define LLM instance here so it always exists
# ChatGroq will read the API key from the environment (GROQ_API_KEY) loaded via load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""),
)


# Helper to summarize context chunks using the LLM
def call_llm_summarize_chunks(prompt: str) -> str:
    """
    Calls the LLM with the provided prompt and returns the summary text.
    """
    try:
        logger.info("[LLMClient] Summarizing context chunks with LLM.")
        response = llm.invoke(prompt)
        # Handle different response types with fallbacks
        # First check if response is already a string
        if isinstance(response, str):
            return response
        # Check for .content attribute (most common case)
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Join string elements, convert dicts to str
                return " ".join(str(item) for item in content)
            return str(content)
        # Fallback to .text attribute if it exists
        if hasattr(response, "text"):
            return response.text
        # Final fallback: convert to string
        return str(response)
    except Exception as e:
        logger.error(f"[LLMClient] Error summarizing chunks: {e}")
        return ""


# Conversation-specific summarization prompt template
CONVERSATION_SUMMARY_PROMPT = """Analyze this conversation history and extract a structured summary. 
Your task is to preserve ALL user-provided information so the AI assistant knows what has already been discussed.

CONVERSATION HISTORY:
{conversation}

---

Extract and return a structured summary in this EXACT format:

## USER DETAILS COLLECTED
- Name: {name or "Not provided"}
- Email: {email or "Not provided"}
- Phone: {phone or "Not provided"}
- Location: {location/city or "Not provided"}
- Industry: {industry or "Not provided"}
- Role/Profession: {role or "Not provided"}
- Meeting Availability: {availability or "Not provided"}

## USER REQUIREMENTS
- What they're looking for: {brief description}
- Their goals/objectives: {goals if mentioned}
- Budget mentioned: {any budget info or "Not discussed"}
- Timeline mentioned: {any timeline or "Not discussed"}

## QUESTIONS ALREADY ASKED BY ASSISTANT
{list of questions the assistant already asked - DO NOT ask these again}

## KEY CONVERSATION POINTS
{brief bullet points of important topics discussed}

CRITICAL RULES:
1. NEVER omit any user-provided information - every detail matters
2. If the user said "IT" for industry, write "IT" not "Not provided"
3. If user said "Delhi" or any location, capture it exactly
4. If user mentioned any time for meeting (e.g., "Saturday 12 noon"), capture it exactly
5. List ALL questions the assistant already asked to prevent repetition
"""


def call_llm_conversation_summary(messages: list) -> str:
    """
    Summarize conversation history with explicit extraction of user-provided details.
    
    This function uses a specialized prompt that instructs the LLM to:
    1. Extract all user personal details (name, email, location, industry, etc.)
    2. Capture meeting availability and preferences
    3. List questions already asked to prevent repetition
    4. Preserve key conversation context
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        Structured summary string with all extracted information
    """
    if not messages:
        return ""
    
    try:
        # Format conversation for the prompt
        conversation_text = []
        for msg in messages:
            role = msg.get("role") or msg.get("sender", "unknown")
            content = msg.get("content", "")
            if content:
                role_label = "User" if role.lower() == "user" else "Assistant"
                conversation_text.append(f"{role_label}: {content}")
        
        if not conversation_text:
            return ""
        
        conversation_str = "\n".join(conversation_text)
        prompt = CONVERSATION_SUMMARY_PROMPT.format(conversation=conversation_str)
        
        logger.info("[LLMClient] Generating conversation-aware summary...")
        response = llm.invoke(prompt)
        
        # Extract content from response
        if isinstance(response, str):
            summary = response
        elif hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                summary = content
            elif isinstance(content, list):
                summary = " ".join(str(item) for item in content)
            else:
                summary = str(content)
        elif hasattr(response, "text"):
            summary = response.text
        else:
            summary = str(response)
        
        logger.info(f"[LLMClient] Conversation summary generated ({len(summary)} chars)")
        return summary.strip() if summary else ""
        
    except Exception as e:
        logger.error(f"[LLMClient] Error generating conversation summary: {e}")
        return ""
