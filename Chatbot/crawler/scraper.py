import requests
from bs4 import BeautifulSoup


def scrape_website(url: str) -> str:
    """
    Scrape the content of a website and return the raw text.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from paragraphs and headings
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        return soup.get_text(separator=' ', strip=True)

    except Exception as e:
        return f"Error scraping {url}: {e}"