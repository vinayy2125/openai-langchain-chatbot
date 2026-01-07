"""Quick test to check ChromaDB sources and context retrieval."""
from app.db.chroma_manager import get_chroma_manager
from app.utils.redis_context import get_redis_context_chunks

def main():
    # Check ChromaDB stats and sample data
    m = get_chroma_manager()
    print("ChromaDB Stats:", m.get_stats())
    
    # Sample search
    results = m.similarity_search("services", n_results=2)
    print("\n--- Sample ChromaDB Results ---")
    for r in results:
        meta = r.get("metadata", {})
        text = r.get("text", "")[:100]
        url = meta.get("url", "NO URL")
        print(f"URL: {url}")
        print(f"Text: {text}...")
        print()
    
    # Check context chunks (what gets sent to LLM)
    print("\n--- Context Chunks (with source annotations) ---")
    chunks = get_redis_context_chunks("test-session", "what services do you offer?", top_n=2)
    for i, chunk in enumerate(chunks):
        print(f"\n=== Chunk {i+1} ===")
        print(chunk[:300])
        if "[Source:" in chunk:
            print("\n✅ SOURCE ANNOTATION FOUND!")
        else:
            print("\n❌ NO SOURCE ANNOTATION")

if __name__ == "__main__":
    main()
