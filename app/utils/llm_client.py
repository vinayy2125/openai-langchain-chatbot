import os
from app.logger import get_logger
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr

# Configure logger
logger = get_logger(__name__)


load_dotenv()

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
