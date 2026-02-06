import os
from app.logger import get_logger
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr

# Configure logger
logger = get_logger(__name__)


load_dotenv()

# Token limits for OpenAI API (keep synchronized with llm_utils.py)
MAX_SUMMARY_INPUT_TOKENS: int = 5000  # Conservative limit for summarization
CHARS_PER_TOKEN: float = 4.0  # Approximate: 1 token ≈ 4 characters


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


def _truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximately fit within max_tokens."""
    if not text:
        return text
    
    estimated = _estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    
    target_chars = int(max_tokens * CHARS_PER_TOKEN)
    truncated = text[:target_chars]
    
    # Try to find a good break point
    last_newline = truncated.rfind('\n')
    if last_newline > target_chars * 0.5:
        return truncated[:last_newline] + "\n[...conversation history truncated...]"
    
    return truncated + "\n[...conversation history truncated...]"

# Define LLM instance here so it always exists
# ChatOpenAI will read the API key from the environment (OPENAI_API_KEY) loaded via load_dotenv()
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
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


# NOTE: CONVERSATION_SUMMARY_PROMPT and call_llm_conversation_summary removed in v3.1.0
# LangChain ConversationSummaryBufferMemory in conversation_memory.py replaces this functionality
# with incremental summarization (no separate LLM call per message)

