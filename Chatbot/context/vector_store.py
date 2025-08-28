import os
import asyncio
import importlib.util
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# -------------------------------
# Dynamic imports for project layout
# -------------------------------
# scraper.py (should expose: async def scrape_website_recursive(start_url, ...))
scraper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'crawler', 'scraper.py'))
spec = importlib.util.spec_from_file_location("scraper", scraper_path)
scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scraper)
scrape_website_recursive = getattr(scraper, "scrape_website_recursive")

# chunker (optional if you have a custom one; we’ll use LangChain splitter here)
# If you prefer your custom chunker, import and use it instead of RecursiveCharacterTextSplitter.


# -------------------------------
# Config
# -------------------------------
load_dotenv()
INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index"))
os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
embedding_model = OpenAIEmbeddings()

# default start URLs
URLS = [
    "https://www.ditstek.com/",
]


# -------------------------------
# Helper: load extra URLs from project root
# -------------------------------
def load_extra_urls(file_name: str = "urls.txt") -> List[str]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(project_root, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"📄 Loaded {len(urls)} extra URLs from {file_path}")
        return urls
    print(f"⚠️ urls.txt not found at {file_path} — continuing without extra URLs")
    return []


# -------------------------------
# Crawl one start URL (handles async scraper)
# -------------------------------
def crawl_start_url(start_url: str, max_pages: int = 300, max_depth: int = 5) -> Dict[str, str]:
    """Runs async scraper even if we’re in a sync context."""
    if asyncio.iscoroutinefunction(scrape_website_recursive):
        return asyncio.run(scrape_website_recursive(start_url, max_pages=max_pages, max_depth=max_depth))
    # If your scraper is synchronous (older version), call directly:
    return scrape_website_recursive(start_url, max_pages=max_pages, max_depth=max_depth)


# -------------------------------
# Build & Save FAISS Index
# -------------------------------
def build_vectorstore(auto_urls: List[str]) -> None:
    # Merge auto-crawl + urls.txt
    extra_urls = load_extra_urls()
    all_seeds = list(dict.fromkeys(auto_urls + extra_urls))  # de-dupe, keep order

    print("\n🔎 Seeds to crawl (auto + urls.txt):")
    for u in all_seeds:
        print(f"   - {u}")

    # Crawl
    pages_collected: Dict[str, str] = {}
    for seed in all_seeds:
        print(f"\n🚀 Crawling seed: {seed}")
        result = crawl_start_url(seed, max_pages=300, max_depth=5)
        print(f"✅ Collected {len(result)} pages from {seed}")
        pages_collected.update(result)  # later seeds overwrite duplicates

    print(f"\n🧾 Total unique pages collected: {len(pages_collected)}")

    if not pages_collected:
        print("⚠️ No pages collected — aborting index build.")
        return

    # Chunking (use LC’s Recursive splitter for consistent granularity)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents: List[Document] = []
    for page_url, page_text in pages_collected.items():
        if not page_text:
            continue
        chunks = splitter.split_text(page_text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": page_url, "chunk_index": i}
                )
            )

    if not documents:
        print("⚠️ No chunks produced — aborting index build.")
        return

    print(f"📦 Prepared {len(documents)} chunks for embedding.")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved to: {INDEX_DIR}")


if __name__ == "__main__":
    build_vectorstore(URLS)
