import os
import logging
# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"[DEBUG] Using API key: {os.getenv('OPENAI_API_KEY')}")
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv

load_dotenv()

# Define LLM instance here so it always exists
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    max_tokens=None,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Use the correct parameter name
)
