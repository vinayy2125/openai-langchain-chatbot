import os
from functools import lru_cache
from tavily import TavilyClient 
from dotenv import load_dotenv

load_dotenv()

# Initialize once
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

client = TavilyClient(api_key=TAVILY_API_KEY)


@lru_cache(maxsize=500)
def search_site(query: str, site_url: str, max_results: int = 5):
    """
    Perform a site-specific internet search using Tavily.
    Results are cached to reduce API usage.
    
    Args:
        query (str): User's search query
        site_url (str): Website domain (e.g., "ditstek.com")
        max_results (int): Number of results to fetch
    
    Returns:
        list of dicts with {title, url, snippet}
    """
    site_query = f"{query} site:{site_url}"

    try:
        results = client.search(query=site_query, max_results=max_results)

        structured = []
        for r in results.get("results", []):
            structured.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content"),
            })

        return structured

    except Exception as e:
        return [{"error": f"Search failed: {e}"}]
