#!/usr/bin/env python
"""Quick script to check ChromaDB services search."""
from app.db.chroma_manager import get_chroma_manager

c = get_chroma_manager()
results = c.similarity_search('services offered', n_results=15)

print("=== ChromaDB Search Results for 'services offered' ===\n")
for i, r in enumerate(results):
    text = r.get("text", "")[:300]
    meta = r.get("metadata", {})
    url = meta.get("url", "N/A")
    sim = r.get("similarity", 0)
    print(f"{i+1}. [Similarity: {sim:.3f}] URL: {url}")
    print(f"   Text: {text}...")
    print()

print(f"\nTotal results: {len(results)}")
