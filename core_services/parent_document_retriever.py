"""
Parent Document Retriever Module

Implements the ParentDocumentRetriever pattern for improved RAG context:
- Search using small, precise child chunks for accurate retrieval
- Return larger parent chunks to give the LLM full context

This solves the chunk size dilemma:
- Small chunks = better search precision but less context
- Large chunks = more context but lower search precision
- ParentDocumentRetriever = best of both worlds
"""
import logging
from typing import List, Dict, Any, Optional
import hashlib

from app.db.chroma_manager import ChromaManager, get_chroma_manager
from core_services.parent_document_store import ParentDocumentStore, get_parent_store

logger = logging.getLogger(__name__)

# Default chunk sizes
CHILD_CHUNK_SIZE = 400    # Small chunks for precise embedding search
PARENT_CHUNK_SIZE = 2000  # Large chunks for LLM context
CHUNK_OVERLAP = 80        # Overlap in characters


class ParentDocumentRetriever:
    """
    Retriever that searches child chunks but returns parent documents.
    
    Workflow:
    1. User query is embedded and searched against child chunks in ChromaDB
    2. Matching child chunks have parent_id in their metadata
    3. Parent documents are fetched from ParentDocumentStore
    4. Full parent context is returned to the LLM
    """
    
    def __init__(
        self, 
        vectorstore: Optional[ChromaManager] = None,
        parent_store: Optional[ParentDocumentStore] = None,
        child_chunk_size: int = CHILD_CHUNK_SIZE,
        parent_chunk_size: int = PARENT_CHUNK_SIZE,
    ):
        """
        Initialize the ParentDocumentRetriever.
        
        Args:
            vectorstore: ChromaManager for child chunk search (lazy-loaded if None)
            parent_store: ParentDocumentStore for parent docs (lazy-loaded if None)
            child_chunk_size: Size of child chunks for embedding
            parent_chunk_size: Size of parent chunks for context
        """
        self._vectorstore = vectorstore
        self._parent_store = parent_store
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
    
    @property
    def vectorstore(self) -> ChromaManager:
        """Lazy-load the vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = get_chroma_manager()
        return self._vectorstore
    
    @property
    def parent_store(self) -> ParentDocumentStore:
        """Lazy-load the parent store."""
        if self._parent_store is None:
            self._parent_store = get_parent_store()
        return self._parent_store
    
    def retrieve(
        self, 
        query: str, 
        k: int = 4,
        include_child_context: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve parent documents for a query.
        
        Args:
            query: The search query
            k: Number of results to return
            include_child_context: If True, include the matched child chunk info
            
        Returns:
            List of dicts with parent content, metadata, and optional child context
        """
        # Step 1: Search child chunks in vectorstore
        child_results = self.vectorstore.similarity_search(query, n_results=k * 2)
        
        if not child_results:
            logger.debug(f"No child chunks found for query: {query[:50]}...")
            return []
        
        # Step 2: Extract unique parent IDs from results
        seen_parents = set()
        results = []
        
        for child in child_results:
            metadata = child.get("metadata", {})
            parent_id = metadata.get("parent_id")
            
            if not parent_id:
                # No parent linking, use child chunk directly as fallback
                if child.get("text"):
                    results.append({
                        "content": child["text"],
                        "metadata": metadata,
                        "source": "child_fallback",
                        "similarity": child.get("similarity", 0),
                    })
                continue
            
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            
            # Step 3: Fetch parent document
            parent_doc = self.parent_store.get_parent(parent_id)
            
            if parent_doc:
                result = {
                    "content": parent_doc["content"],
                    "metadata": parent_doc["metadata"],
                    "source": "parent",
                    "parent_id": parent_id,
                    "similarity": child.get("similarity", 0),
                }
                
                if include_child_context:
                    result["matched_child"] = {
                        "text": child.get("text", ""),
                        "similarity": child.get("similarity", 0),
                    }
                
                results.append(result)
            else:
                # Parent not found, use child as fallback
                logger.warning(f"Parent {parent_id} not found, using child chunk")
                results.append({
                    "content": child.get("text", ""),
                    "metadata": metadata,
                    "source": "child_fallback",
                    "similarity": child.get("similarity", 0),
                })
        
        # Return top k results sorted by similarity
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results[:k]
    
    def retrieve_text(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieve just the text content (convenience method).
        
        Args:
            query: The search query
            k: Number of results
            
        Returns:
            List of parent document text contents
        """
        results = self.retrieve(query, k=k)
        
        texts = []
        for result in results:
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            # Format with source URL if available
            source_url = metadata.get("url", "")
            if source_url:
                texts.append(f"{content}\n[Source: {source_url}]")
            else:
                texts.append(content)
        
        return texts
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        clear_existing: bool = False,
    ) -> int:
        """
        Add documents with automatic parent-child splitting.
        
        Args:
            documents: List of dicts with 'content', 'url', 'title', 'metadata'
            clear_existing: Whether to clear existing data first
            
        Returns:
            Number of child chunks added
        """
        if clear_existing:
            self.vectorstore.clear_collection()
            self.parent_store.clear()
        
        child_texts = []
        child_metadatas = []
        child_ids = []
        
        for doc in documents:
            content = doc.get("content", "")
            url = doc.get("url", "")
            title = doc.get("title", "")
            base_metadata = doc.get("metadata", {})
            
            if not content:
                continue
            
            # Create parent chunks
            parent_chunks = self._split_into_chunks(content, self.parent_chunk_size)
            
            for p_idx, parent_content in enumerate(parent_chunks):
                # Generate parent ID
                parent_id = self._generate_id(url, p_idx, "parent")
                
                # Store parent document
                parent_metadata = {
                    **base_metadata,
                    "url": url,
                    "title": title,
                    "chunk_type": "parent",
                }
                self.parent_store.add_parent(
                    parent_id=parent_id,
                    content=parent_content,
                    metadata=parent_metadata,
                    child_ids=[],  # Will be populated as we add children
                )
                
                # Create child chunks from this parent
                child_chunks = self._split_into_chunks(parent_content, self.child_chunk_size)
                
                for c_idx, child_content in enumerate(child_chunks):
                    child_id = self._generate_id(f"{parent_id}_{c_idx}", c_idx, "child")
                    
                    child_texts.append(child_content)
                    child_metadatas.append({
                        **base_metadata,
                        "url": url,
                        "title": title,
                        "parent_id": parent_id,  # Link to parent
                        "chunk_type": "child",
                    })
                    child_ids.append(child_id)
                    
                    # Register child with parent
                    self.parent_store.add_child_to_parent(parent_id, child_id)
        
        # Add all child chunks to vectorstore
        if child_texts:
            self.vectorstore.add_documents(
                texts=child_texts,
                metadatas=child_metadatas,
                ids=child_ids,
            )
        
        # Persist parent store
        self.parent_store.save_to_disk()
        
        logger.info(f"Added {len(child_texts)} child chunks linked to {self.parent_store.count()} parents")
        return len(child_texts)
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within the last 20% of chunk
                search_start = int(end - chunk_size * 0.2)
                last_period = text.rfind('.', search_start, end)
                if last_period > search_start:
                    end = last_period + 1
            
            chunks.append(text[start:end].strip())
            start = end - CHUNK_OVERLAP  # Overlap for context
            
            if start >= len(text):
                break
        
        return [c for c in chunks if c]
    
    def _generate_id(self, base: str, index: int, prefix: str) -> str:
        """Generate a unique ID for a chunk."""
        content = f"{prefix}_{base}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "vectorstore_stats": self.vectorstore.get_stats(),
            "parent_store_stats": self.parent_store.get_stats(),
            "child_chunk_size": self.child_chunk_size,
            "parent_chunk_size": self.parent_chunk_size,
        }


# Singleton instance
_retriever_instance: Optional[ParentDocumentRetriever] = None


def get_parent_document_retriever() -> ParentDocumentRetriever:
    """Get the singleton ParentDocumentRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ParentDocumentRetriever()
    return _retriever_instance
