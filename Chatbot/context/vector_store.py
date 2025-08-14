import importlib.util
import os
import requests

import sys, os
sys.path.append(os.path.abspath(".."))

# #Dynamically load scraper.py
# scraper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'crawler', 'scraper.py'))
# spec = importlib.util.spec_from_file_location("scraper", scraper_path)
# scraper = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(scraper)

# #Get the function
# scrape_website_recursive = scraper.scrape_website_recursive

# chunker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chunking", "chunk_generator.py"))
# spec = importlib.util.spec_from_file_location("chunker", chunker_path)
# chunker = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(chunker)

# chunk_text = chunker.chunk_text

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from crawler.scraper import scrape_website_recursive
from chunking.chunk_generator import chunk_text



load_dotenv()


# Config
INDEX_DIR = "vectorstore/faiss_index"
embedding_model = OpenAIEmbeddings()

# Example URLs to crawl
URLS = [
    "https://www.ditstek.com/",
    # Add more URLs as needed
]

# --- Full Site Crawler ---
def scrape_website_recursive(start_url: str, max_pages: int = 150, max_depth: int = 20) -> dict:
    """
    Crawl a website starting from `start_url` and return a dict of {url: text_content}.
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

        # Remove non-content elements
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

# --- Build and Save FAISS Index ---
def build_vectorstore(urls):
    documents = []

    for url in urls:
        pages = scrape_website_recursive(url, max_pages=50, max_depth=3)
        for page_url, page_text in pages.items():
            if page_text:
                chunks = chunk_text(page_text)
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": page_url, "chunk_index": i}  # Add chunk index for better traceability
                    ))

    if not documents:
        print("No documents found to index.")
        return

    # Build and save FAISS index
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ Vectorstore built with {len(documents)} chunks.")

if __name__ == "__main__":
    build_vectorstore(URLS)