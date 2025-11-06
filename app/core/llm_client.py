import os
from app.logger import get_logger
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Configure logger
logger = get_logger(__name__)


load_dotenv()

# Define LLM instance here so it always exists
# ChatOpenAI will read the API key from the environment (OPENAI_API_KEY) loaded via load_dotenv()
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=None,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)


# Helper to summarize context chunks using the LLM
def call_llm_summarize_chunks(prompt: str) -> str:
    """
    Calls the LLM with the provided prompt and returns the summary text.
    """
    try:
        logger.info("[LLMClient] Summarizing context chunks with LLM.")
        response = llm.invoke(prompt)
        # If response is a string, return directly; if object, extract text
        if isinstance(response, str):
            return response
        if hasattr(response, 'content'):
            return response.content
        if hasattr(response, 'text'):
            return response.text
        return str(response)
    except Exception as e:
        logger.error(f"[LLMClient] Error summarizing chunks: {e}")
        return ""
 