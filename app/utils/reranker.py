"""
Deep Reranker Service.

Implements high-precision semantic reranking using Cross-Encoders.
Prioritizes correctness over speed (though MiniLM is fast).
Includes granular observability to explain WHY documents were ranked/dropped.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import logging
import math
from functools import lru_cache

# Lazy import to avoid startup overhead
_cross_encoder = None

logger = logging.getLogger("reranker")

@dataclass
class RerankResult:
    """Detailed result of the reranking process for observability."""
    sorted_docs: List[Dict[str, Any]]
    scores: List[float]
    decision_reason: str
    supporting_chunk_count: int
    drop_stats: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

class DeepReranker:
    """
    Reranks candidate documents using a Cross-Encoder model.
    This provides true semantic understanding of Query <-> Doc relationship.
    """
    
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self):
        pass

    def _get_model(self):
        """Lazy load the model."""
        global _cross_encoder
        if _cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.MODEL_NAME}")
                _cross_encoder = CrossEncoder(self.MODEL_NAME)
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder: {e}")
                return None
        return _cross_encoder

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
        threshold: float = 0.0, # Will be set by Dual Gate logic usually
    ) -> RerankResult:
        """
        Rerank documents and return detailed trace.
        
        Args:
            query: User query
            documents: List of dicts (must contain 'text')
            top_n: Max docs to return
            threshold: Minimum score to keep (hard gate)
            
        Returns:
            RerankResult with trace data
        """
        start_time = time.perf_counter()
        
        if not documents:
            return RerankResult([], [], "No documents provided", 0, {}, 0.0)
            
        model = self._get_model()
        if not model:
            # Fallback: Return original list if model fails
            logger.warning("Reranker fallback: Model unavailable")
            return RerankResult(
                documents[:top_n], 
                [0.5] * len(documents), 
                "Fallback: Model unavailable", 
                len(documents), 
                {}, 
                (time.perf_counter() - start_time) * 1000
            )
            
        # Prepare pairs for cross-encoder
        pairs = []
        valid_indices = []
        
        for idx, doc in enumerate(documents):
            text = doc.get("text", "")
            if text:
                pairs.append((query, text))
                valid_indices.append(idx)
        
        if not pairs:
             return RerankResult(
                [], 
                [], 
                "No valid text in documents", 
                0, 
                {"empty_text": len(documents)}, 
                (time.perf_counter() - start_time) * 1000
            )
            
        try:
            # Predict scores
            scores = model.predict(pairs)
            # Ensure scores is a list even if single result
            if not isinstance(scores, (list, tuple, type(documents))): # numpy array check
                 scores = scores.tolist() if hasattr(scores, "tolist") else [scores]
            
            # Combine docs with scores (Apply Sigmoid if needed)
            # MS MARCO CrossEncoders usually output logits. 
            # We apply 1 / (1 + exp(-x)) to get 0-1 probability.
            scored_docs = []
            drop_stats = {"total_candidates": len(documents), "below_threshold": 0}
            
            for i, raw_score in enumerate(scores):
                # Sigmoid normalization
                score = 1 / (1 + math.exp(-raw_score)) if isinstance(raw_score, (int, float)) else 0.0
                
                original_idx = valid_indices[i]
                original_doc = documents[original_idx]
                
                if score >= threshold:
                    scored_docs.append((score, original_doc))
                else:
                    drop_stats["below_threshold"] += 1
            
            # Sort by score descending
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Slice top_n
            final_docs = [doc for _, doc in scored_docs[:top_n]]
            final_scores = [score for score, _ in scored_docs[:top_n]]
            
            latency = (time.perf_counter() - start_time) * 1000
            supporting_count = len(scored_docs) # Count of ALL docs passing threshold
            
            decision = (
                f"Selected {len(final_docs)} from {len(documents)} candidates. "
                f"Threshold {threshold:.2f}. "
                f"Passing: {supporting_count}."
            )
            
            return RerankResult(
                final_docs,
                final_scores,
                decision,
                supporting_count,
                drop_stats,
                latency
            )
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback
            return RerankResult(
                documents[:top_n], 
                [0.0] * len(documents), 
                f"Fallback: Error {str(e)}", 
                0, 
                {},
                (time.perf_counter() - start_time) * 1000
            )

# Singleton
_reranker_instance = DeepReranker()

def get_reranker() -> DeepReranker:
    return _reranker_instance
