#!/usr/bin/env python
"""Test script for hybrid search implementation.

Compares hybrid search results with semantic-only search to validate
the implementation and show accuracy improvements.
"""
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_services.hybrid_search import get_hybrid_search_manager
from app.db.chroma_manager import get_chroma_manager


def test_hybrid_search():
    """Run comparison tests between hybrid and semantic-only search."""
    
    print("=" * 70)
    print("HYBRID SEARCH TEST")
    print("=" * 70)
    
    # Initialize managers
    chroma = get_chroma_manager()
    hybrid = get_hybrid_search_manager()
    
    # Get ChromaDB stats
    stats = chroma.get_stats()
    print(f"\nChromaDB Documents: {stats['document_count']}")
    
    # Build BM25 index
    print("\nBuilding BM25 index...")
    if hybrid.build_bm25_index():
        print("BM25 index built successfully!")
    else:
        print("ERROR: Failed to build BM25 index")
        return
    
    # Test queries
    test_queries = [
        "What services do you offer?",
        "AI chatbot development",
        "healthcare software solutions",
        "contact email phone",
        "custom software development cost",
    ]
    
    for query in test_queries:
        print("\n" + "-" * 70)
        print(f"QUERY: {query}")
        print("-" * 70)
        
        # Semantic-only results
        semantic_results = hybrid.semantic_search(query, top_n=5)
        print(f"\n[SEMANTIC ONLY] Top 5 results:")
        for i, r in enumerate(semantic_results[:5], 1):
            text_preview = r.get("text", "")[:80].replace("\n", " ")
            sim = r.get("similarity", 0)
            print(f"  {i}. [{sim:.3f}] {text_preview}...")
        
        # BM25-only results
        bm25_results = hybrid.bm25_search(query, top_n=5)
        print(f"\n[BM25 ONLY] Top 5 results:")
        for i, r in enumerate(bm25_results[:5], 1):
            text_preview = r.get("text", "")[:80].replace("\n", " ")
            score = r.get("bm25_score", 0)
            print(f"  {i}. [{score:.2f}] {text_preview}...")
        
        # Hybrid results
        hybrid_results = hybrid.hybrid_search(query, top_n=5, alpha=0.5)
        print(f"\n[HYBRID (alpha=0.5)] Top 5 results:")
        for i, r in enumerate(hybrid_results[:5], 1):
            text_preview = r.get("text", "")[:80].replace("\n", " ")
            rrf_score = r.get("rrf_score", 0)
            sources = r.get("sources", [])
            print(f"  {i}. [{rrf_score:.4f}] [{'+'.join(sources)}] {text_preview}...")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_hybrid_search()
