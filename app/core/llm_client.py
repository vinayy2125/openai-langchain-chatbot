import os
import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger(__name__)

load_dotenv()

# Define LLM instance here so it always exists
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=None,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
