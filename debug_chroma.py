import chromadb
from chromadb.config import Settings
import logging
import sys
import time

def debug_write_test():
    url = "http://localhost:8001"
    print(f"Connecting to ChromaDB Server at {url} ...")
    try:
        client = chromadb.HttpClient(host='localhost', port=8001)
        print(f"Server heartbeat: {client.heartbeat()}")
        
        target = "website_embeddings"
        col = client.get_or_create_collection(target)
        print(f"✅ Collection accessed: {col.name}")
        
        print("Attempting to insert 1 test document...")
        try:
            col.add(
                documents=["This is a test document to verify write capability."],
                metadatas=[{"source": "debug_script"}],
                ids=["debug_test_001"]
            )
            print("✅ Write successful!")
        except Exception as e:
            print(f"❌ Write failed: {e}")
            
        print(f"New Count: {col.count()}")
        
    except Exception as e:
        print(f"Global Error: {e}")

if __name__ == "__main__":
    debug_write_test()
