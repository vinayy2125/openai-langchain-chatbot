"""
Hybrid Search Module

Combines semantic (dense embedding) search with BM25 (sparse keyword) search
using Reciprocal Rank Fusion (RRF) for improved retrieval accuracy.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Converts text to lowercase, removes punctuation, and splits on whitespace.
    """
    if not text:
        return []
    # Convert to lowercase and remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter empty tokens
    tokens = [t.strip() for t in text.split() if t.strip() and len(t.strip()) > 1]
    return tokens


class HybridSearchManager:
    """
    Manages hybrid search combining semantic similarity with BM25 keyword matching.
    
    Features:
    - BM25 index built from document corpus (lazy-loaded)
    - Semantic search via ChromaDB
    - Reciprocal Rank Fusion (RRF) for result merging
    - Configurable alpha weight for semantic vs keyword balance
    """
    
    _instance: Optional["HybridSearchManager"] = None
    
    def __init__(self, chroma_manager=None):
        """
        Initialize HybridSearchManager.
        
        Args:
            chroma_manager: Optional ChromaDB manager instance.
                           If not provided, will be lazy-loaded.
        """
        self._chroma_manager = chroma_manager
        self._bm25_index: Optional[BM25Okapi] = None
        self._documents: List[Dict] = []
        self._doc_texts: List[str] = []
        self._index_built = False
        
        logger.info("✅ HybridSearchManager initialized")
    
    @classmethod
    def get_instance(cls, chroma_manager=None) -> "HybridSearchManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(chroma_manager)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing or re-indexing)."""
        cls._instance = None
    
    def _get_chroma(self):
        """Get ChromaDB manager (lazy-load if needed)."""
        if self._chroma_manager is None:
            try:
                from app.db.chroma_manager import get_chroma_manager
                self._chroma_manager = get_chroma_manager()
            except Exception as e:
                logger.warning(f"Failed to get ChromaDB manager: {e}")
        return self._chroma_manager
    
    def build_bm25_index(self, force_rebuild: bool = False) -> bool:
        """
        Build BM25 index from ChromaDB documents.
        
        Args:
            force_rebuild: If True, rebuild index even if already exists
            
        Returns:
            True if index was built successfully
        """
        if self._index_built and not force_rebuild:
            logger.debug("BM25 index already built, skipping")
            return True
        
        chroma = self._get_chroma()
        if not chroma:
            logger.warning("ChromaDB not available, cannot build BM25 index")
            return False
        
        try:
            # Get total document count
            total_count = chroma.collection.count()
            if total_count == 0:
                logger.warning("No documents found in ChromaDB")
                return False
            
            logger.info(f"Building BM25 index from {total_count} documents...")
            
            # Paginated retrieval to avoid server disconnection on large datasets
            BATCH_SIZE = 5000
            all_documents = []
            all_metadatas = []
            all_ids = []
            
            offset = 0
            while offset < total_count:
                try:
                    batch_data = chroma.collection.get(
                        include=["documents", "metadatas"],
                        limit=BATCH_SIZE,
                        offset=offset
                    )
                    
                    batch_docs = batch_data.get("documents", [])
                    batch_metas = batch_data.get("metadatas", [])
                    batch_ids = batch_data.get("ids", [])
                    
                    if not batch_docs:
                        break
                    
                    all_documents.extend(batch_docs)
                    all_metadatas.extend(batch_metas)
                    all_ids.extend(batch_ids)
                    
                    offset += len(batch_docs)
                    logger.debug(f"  BM25 build progress: {offset}/{total_count} documents")
                    
                except Exception as batch_error:
                    logger.warning(f"Error fetching batch at offset {offset}: {batch_error}")
                    # Try to continue with what we have
                    if all_documents:
                        break
                    raise
            
            if not all_documents:
                logger.warning("No documents retrieved from ChromaDB")
                return False
            
            # Store documents with metadata for later retrieval
            self._documents = []
            self._doc_texts = []
            tokenized_corpus = []
            
            for i, (doc, meta, doc_id) in enumerate(zip(all_documents, all_metadatas, all_ids)):
                if not doc:
                    continue
                    
                self._documents.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {},
                })
                self._doc_texts.append(doc)
                
                # Tokenize for BM25
                tokens = _tokenize(doc)
                tokenized_corpus.append(tokens)
            
            # Build BM25 index
            self._bm25_index = BM25Okapi(tokenized_corpus)
            self._index_built = True
            
            logger.info(f"✅ BM25 index built with {len(self._documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build BM25 index: {e}")
            return False
    
    def bm25_search(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_n: Number of results to return
            
        Returns:
            List of results with text, metadata, and BM25 score
        """
        if not self._index_built:
            if not self.build_bm25_index():
                return []
        
        if not query or not self._bm25_index:
            return []
        
        # Tokenize query
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top-N document indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_n]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include documents with positive scores
                doc = self._documents[idx]
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "bm25_score": float(scores[idx]),
                    "bm25_rank": rank + 1,
                })
        
        logger.debug(f"🔍 BM25 search found {len(results)} results for query")
        return results
    
    def semantic_search(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search via ChromaDB.
        
        Args:
            query: Search query
            top_n: Number of results to return
            
        Returns:
            List of results with text, metadata, and similarity score
        """
        chroma = self._get_chroma()
        if not chroma:
            return []
        
        try:
            results = chroma.similarity_search(query, n_results=top_n)
            
            # Add rank for RRF calculation
            for rank, result in enumerate(results):
                result["semantic_rank"] = rank + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_n: int = 10,
        alpha: float = 0.5,
        semantic_top_n: int = 20,
        bm25_top_n: int = 20,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and BM25 results.
        
        Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
        
        Args:
            query: Search query
            top_n: Final number of results to return
            alpha: Weight for semantic vs BM25 (0.0 = BM25 only, 1.0 = semantic only)
            semantic_top_n: Number of semantic candidates to retrieve
            bm25_top_n: Number of BM25 candidates to retrieve
            rrf_k: RRF constant (default 60, standard value)
            
        Returns:
            List of results sorted by combined RRF score
        """
        if not query:
            return []
        
        # Get results from both methods
        semantic_results = self.semantic_search(query, top_n=semantic_top_n)
        bm25_results = self.bm25_search(query, top_n=bm25_top_n)
        
        # If one method fails, fall back to the other
        if not semantic_results and not bm25_results:
            logger.warning("Both search methods returned no results")
            return []
        
        if not semantic_results:
            logger.debug("Semantic search empty, using BM25 only")
            return bm25_results[:top_n]
        
        if not bm25_results:
            logger.debug("BM25 search empty, using semantic only")
            return semantic_results[:top_n]
        
        # Merge using Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            alpha=alpha,
            rrf_k=rrf_k,
        )
        
        logger.info(
            f"🔀 Hybrid search: {len(semantic_results)} semantic + "
            f"{len(bm25_results)} BM25 → {len(merged)} merged results"
        )
        
        return merged[:top_n]
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum of 1 / (k + rank) for each result list
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            alpha: Weight for semantic (1-alpha for BM25)
            rrf_k: RRF constant (typically 60)
            
        Returns:
            Merged and re-ranked results
        """
        # Create a map of document text to combined scores
        doc_scores: Dict[str, Dict] = {}
        
        # Process semantic results
        for result in semantic_results:
            text = result.get("text", "")
            if not text:
                continue
                
            # RRF score for semantic: alpha * 1/(k + rank)
            rank = result.get("semantic_rank", result.get("rank", 1))
            rrf_score = alpha * (1.0 / (rrf_k + rank))
            
            # Use text hash as key for deduplication
            key = text[:200].lower()  # Use first 200 chars as key
            
            if key not in doc_scores:
                doc_scores[key] = {
                    "text": text,
                    "metadata": result.get("metadata", {}),
                    "id": result.get("id", ""),
                    "rrf_score": 0.0,
                    "semantic_score": result.get("similarity", 0),
                    "bm25_score": 0.0,
                    "sources": ["semantic"],
                }
            
            doc_scores[key]["rrf_score"] += rrf_score
        
        # Process BM25 results
        for result in bm25_results:
            text = result.get("text", "")
            if not text:
                continue
                
            # RRF score for BM25: (1-alpha) * 1/(k + rank)
            rank = result.get("bm25_rank", 1)
            rrf_score = (1.0 - alpha) * (1.0 / (rrf_k + rank))
            
            key = text[:200].lower()
            
            if key not in doc_scores:
                doc_scores[key] = {
                    "text": text,
                    "metadata": result.get("metadata", {}),
                    "id": result.get("id", ""),
                    "rrf_score": 0.0,
                    "semantic_score": 0.0,
                    "bm25_score": result.get("bm25_score", 0),
                    "sources": ["bm25"],
                }
            else:
                doc_scores[key]["sources"].append("bm25")
                doc_scores[key]["bm25_score"] = result.get("bm25_score", 0)
            
            doc_scores[key]["rrf_score"] += rrf_score
        
        # Sort by combined RRF score (descending)
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        
        # Add final ranking
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
            # Convert similarity to match expected format
            result["similarity"] = result["rrf_score"]
        
        return sorted_results


def get_hybrid_search_manager(chroma_manager=None) -> HybridSearchManager:
    """Get the singleton HybridSearchManager instance."""
    return HybridSearchManager.get_instance(chroma_manager)
