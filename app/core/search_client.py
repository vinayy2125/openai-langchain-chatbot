import os
from tavily import TavilyClient 
from dotenv import load_dotenv

load_dotenv()

# Initialize once
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

client = TavilyClient(api_key=TAVILY_API_KEY)

