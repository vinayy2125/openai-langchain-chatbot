"""
Startup Warm-up Module

Eagerly initializes heavy singletons and pre-builds indexes to eliminate
cold-start latency on first request.

DEPLOYMENT NOTES:
- Each worker process will independently warm up. This is expected.
- BM25 index is O(N) on corpus size. Acceptable if corpus is moderate.
- Background thread allows health checks to pass while warming up.
"""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Warm-up state with proper lock protection
_warmup_complete = False
_warmup_lock = threading.Lock()


def is_warmup_complete() -> bool:
    """
    Check if warm-up has completed.
    
    Thread-safe read with lock protection.
    """
    with _warmup_lock:
        return _warmup_complete


def warmup_rag_components(background: bool = True) -> Optional[threading.Thread]:
    """
    Warm up RAG components at startup.
    
    Initializes:
    1. ChromaDB singleton
    2. HybridSearchManager singleton + BM25 index
    3. Embedding model (verification)
    
    Args:
        background: If True, run in background thread. If False, block until complete.
        
    Returns:
        Thread object if background=True, else None
    """
    # Check with lock to avoid duplicate warm-ups
    with _warmup_lock:
        if _warmup_complete:
            logger.info("✅ RAG warm-up already complete, skipping")
            return None
    
    def _do_warmup():
        global _warmup_complete
        start_time = time.perf_counter()
        
        # Acquire lock for the entire warm-up process
        with _warmup_lock:
            # Double-check after acquiring lock (another thread may have completed)
            if _warmup_complete:
                return
            
            logger.info("🔥 Starting RAG component warm-up...")
            
            # 1. Initialize ChromaDB singleton
            chroma_ready = False
            try:
                # Try to import chroma_manager (it handles chromadb import internally)
                from app.db.chroma_manager import get_chroma_manager, CHROMADB_AVAILABLE
                
                if not CHROMADB_AVAILABLE:
                    logger.warning("  ✗ ChromaDB module not installed. Skipping ChromaDB warm-up.")
                    logger.info("  Install with: pip install chromadb")
                    chroma_ready = False
                else:
                    chroma = get_chroma_manager()
                    doc_count = chroma.collection.count()
                    logger.info(f"  ✓ ChromaDB initialized ({doc_count} documents)")
                    chroma_ready = True
            except ImportError as ie:
                logger.warning(f"  ✗ ChromaDB module not available: {ie}")
                logger.info("  Install with: pip install chromadb")
                chroma_ready = False
            except Exception as e:
                logger.warning(f"  ✗ ChromaDB warm-up failed: {e}")
            
            # 2. Initialize HybridSearchManager and build BM25 index
            # Only attempt if ChromaDB is ready (BM25 reads from Chroma)
            if chroma_ready:
                try:
                    from core_services.hybrid_search import get_hybrid_search_manager
                    hybrid = get_hybrid_search_manager()
                    if hybrid.build_bm25_index():
                        logger.info(f"  ✓ BM25 index built ({len(hybrid._documents)} documents)")
                    else:
                        logger.warning("  ✗ BM25 index build returned False (empty corpus?)")
                except Exception as e:
                    logger.warning(f"  ✗ HybridSearch warm-up failed: {e}")
            else:
                logger.warning("  ⊘ Skipping BM25 warm-up (ChromaDB not available)")
            
            # 3. Verify embedding model is ready with a test query
            try:
                from core_services.embedding_utils import get_embedding
                # Warm the embedding model with a test query
                test_emb = get_embedding("test warm-up query for model initialization")
                if test_emb and len(test_emb) > 0:
                    logger.info(f"  ✓ Embedding model ready (dim={len(test_emb)})")
                else:
                    logger.warning("  ✗ Embedding model returned empty vector")
            except Exception as e:
                logger.warning(f"  ✗ Embedding model warm-up failed: {e}")
            
            # Mark warm-up complete (still inside lock)
            _warmup_complete = True
            elapsed = time.perf_counter() - start_time
            logger.info(f"🔥 RAG warm-up complete in {elapsed:.2f}s")
    
    if background:
        thread = threading.Thread(
            target=_do_warmup, 
            daemon=True, 
            name="rag-warmup"
        )
        thread.start()
        logger.info("🔥 RAG warm-up started in background thread")
        return thread
    else:
        _do_warmup()
        return None


def reset_warmup_state():
    """
    Reset warm-up state. For testing only.
    
    WARNING: Do not call in production.
    """
    global _warmup_complete
    with _warmup_lock:
        _warmup_complete = False
        logger.warning("⚠️ Warm-up state reset (testing only)")
