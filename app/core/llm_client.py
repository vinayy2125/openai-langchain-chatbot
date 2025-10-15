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
