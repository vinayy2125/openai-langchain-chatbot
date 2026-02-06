"""
Parent Document Store Module

In-memory store for parent documents used by ParentDocumentRetriever.
Stores larger context chunks that are retrieved when child chunks match a search query.
"""
import json
import logging
from typing import Dict, Optional, List, Any
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class ParentDocumentStore:
    """
    Key-value store for parent documents with optional persistence.
    
    Used with ParentDocumentRetriever to enable:
    - Small chunks for accurate embedding search
    - Large parent chunks for rich LLM context
    """
    
    _instance: Optional["ParentDocumentStore"] = None
    _lock = threading.Lock()
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the parent document store.
        
        Args:
            persist_path: Optional path to persist documents to JSON file
        """
        self._store: Dict[str, Dict[str, Any]] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Load existing data if persistence file exists
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()
    
    @classmethod
    def get_instance(cls, persist_path: Optional[str] = None) -> "ParentDocumentStore":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(persist_path)
        return cls._instance
    
    def add_parent(
        self, 
        parent_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        child_ids: Optional[List[str]] = None
    ) -> None:
        """
        Add a parent document to the store.
        
        Args:
            parent_id: Unique identifier for the parent document
            content: The full content of the parent document
            metadata: Optional metadata (source URL, title, etc.)
            child_ids: Optional list of child chunk IDs that link to this parent
        """
        with self._lock:
            self._store[parent_id] = {
                "content": content,
                "metadata": metadata or {},
                "child_ids": child_ids or [],
            }
        
        logger.debug(f"Added parent document: {parent_id[:20]}...")
    
    def get_parent(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a parent document by ID.
        
        Args:
            parent_id: The parent document ID
            
        Returns:
            Dict with content, metadata, and child_ids, or None if not found
        """
        return self._store.get(parent_id)
    
    def get_parent_content(self, parent_id: str) -> Optional[str]:
        """
        Retrieve just the content of a parent document.
        
        Args:
            parent_id: The parent document ID
            
        Returns:
            The parent document content, or None if not found
        """
        doc = self._store.get(parent_id)
        return doc["content"] if doc else None
    
    def get_parents_by_child_ids(self, child_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve parent documents that contain any of the given child IDs.
        
        Args:
            child_ids: List of child chunk IDs to search for
            
        Returns:
            List of parent documents (with content and metadata)
        """
        child_set = set(child_ids)
        results = []
        seen_parents = set()
        
        for parent_id, doc in self._store.items():
            if parent_id in seen_parents:
                continue
            # Check if any of the parent's children match
            if set(doc.get("child_ids", [])) & child_set:
                results.append({
                    "parent_id": parent_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                })
                seen_parents.add(parent_id)
        
        return results
    
    def add_child_to_parent(self, parent_id: str, child_id: str) -> bool:
        """
        Add a child ID to an existing parent's child_ids list.
        
        Args:
            parent_id: The parent document ID
            child_id: The child chunk ID to add
            
        Returns:
            True if successful, False if parent not found
        """
        with self._lock:
            if parent_id in self._store:
                if child_id not in self._store[parent_id]["child_ids"]:
                    self._store[parent_id]["child_ids"].append(child_id)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        with self._lock:
            self._store.clear()
        logger.info("Parent document store cleared")
    
    def count(self) -> int:
        """Return the number of parent documents in the store."""
        return len(self._store)
    
    def save_to_disk(self) -> bool:
        """
        Persist the store to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.persist_path:
            logger.warning("No persist_path configured, skipping save")
            return False
        
        try:
            # Ensure parent directory exists
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._lock:
                with open(self.persist_path, 'w', encoding='utf-8') as f:
                    json.dump(self._store, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(self._store)} parent documents to {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save parent store: {e}")
            return False
    
    def _load_from_disk(self) -> bool:
        """
        Load the store from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                self._store = json.load(f)
            
            logger.info(f"Loaded {len(self._store)} parent documents from {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load parent store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        total_content_size = sum(len(doc["content"]) for doc in self._store.values())
        total_children = sum(len(doc["child_ids"]) for doc in self._store.values())
        
        return {
            "parent_count": len(self._store),
            "total_content_size_chars": total_content_size,
            "total_child_mappings": total_children,
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }


def get_parent_store(persist_path: Optional[str] = None) -> ParentDocumentStore:
    """Get the singleton ParentDocumentStore instance."""
    # Default persist path
    if persist_path is None:
        persist_path = "./chroma_db/parent_documents.json"
    return ParentDocumentStore.get_instance(persist_path)
