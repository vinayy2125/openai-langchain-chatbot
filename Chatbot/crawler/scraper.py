import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def scrape_website_recursive(start_url: str, max_pages: int = 50, max_depth: int = 3) -> dict:
    """
    Crawl a website starting from `start_url` and return a dict of {url: text_content}.
    
    Args:
        start_url (str): The starting URL for crawling.
        max_pages (int): Maximum number of pages to visit.
        max_depth (int): Maximum crawl depth.
    
    Returns:
        dict: {url: extracted_text}
    """
    visited = set()
    to_visit = [(start_url, 0)]
    scraped_data = {}

    while to_visit and len(visited) < max_pages:
        url, depth = to_visit.pop(0)

        if url in visited or depth > max_depth:
            continue
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts, styles, etc.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        scraped_data[url] = text
        visited.add(url)
        print(f"✅ Crawled: {url}")

        # Find internal links
        base_domain = urlparse(start_url).netloc
        for link_tag in soup.find_all("a", href=True):
            link = urljoin(url, link_tag["href"])
            if urlparse(link).netloc == base_domain and link not in visited:
                to_visit.append((link, depth + 1))

    return scraped_data

# Example usage:
if __name__ == "__main__":
    site_data = scrape_website_recursive("https://www.ditstek.com/", max_pages=20, max_depth=2)
    print(f"Scraped {len(site_data)} pages.")
