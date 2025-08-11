import importlib.util
import os

# Dynamically load scraper.py
scraper_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'crawler', 'scraper.py'))
spec = importlib.util.spec_from_file_location("scraper", scraper_path)
scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scraper)

# Get the function
scrape_website = scraper.scrape_website

chunker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chunking", "chunk_generator.py"))
spec = importlib.util.spec_from_file_location("chunker", chunker_path)
chunker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunker)

chunk_text = chunker.chunk_text

# from crawler.scraper import scrape_website
# from chunking.chunk_generator import chunk_text

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document



load_dotenv()


# Config
INDEX_DIR = "vectorstore/faiss_index"

# Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Sample input URL list (can be CLI or UI driven)
URLS = [
    "https://www.ditstek.com/",
    # Add more URLs as needed
]

def build_vectorstore(urls):
    documents = []
    
    for url in urls:
        content = scrape_website(url) # This should return raw text
        if content:
            chunks = chunk_text(content) # Chunk into smaller parts
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": url}
                    ))
                
    if not documents:
        print("No documents found to index.")
        return

# BUild and save FAISS index
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(INDEX_DIR)
    print(f"✅ Vectorstore built with {len(documents)} chunks.")

    vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
    
if __name__ == "__main__":
    build_vectorstore(URLS)
                
                