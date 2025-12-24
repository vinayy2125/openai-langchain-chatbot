"""
ChromaDB Manager Module

Central manager for ChromaDB vector database operations.
Replaces Redis vector storage for RAG functionality.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings

from core_services.embedding_utils import get_embedding, get_embeddings_batch
from app.scraper.config import get_scraper_config, ScraperConfig

logger = logging.getLogger(__name__)

# ChromaDB max batch size (default is 5461)
CHROMA_BATCH_SIZE = 5000  # Use slightly less than max for safety


class ChromaManager:
    """
    Manages ChromaDB collections for storing and querying embeddings.
    
    Features:
    - Persistent storage on disk
    - Uses existing embedding_utils for vector generation
    - Semantic similarity search
    - Collection management (create, update, delete)
    """
    
    _instance: Optional["ChromaManager"] = None
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Path for persistent storage. 
                             Uses config default if not provided.
        """
        config = get_scraper_config()
        
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = config.chroma_db_path
        
        self.collection_name = config.chroma_collection_name
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        self._collection = None
        logger.info(f"✅ ChromaDB initialized at {self.persist_directory}")
    
    @classmethod
    def get_instance(cls, persist_directory: Optional[str] = None) -> "ChromaManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(persist_directory)
        return cls._instance
    
    def get_collection(self, name: Optional[str] = None) -> chromadb.Collection:
        """
        Get or create a collection.
        
        Args:
            name: Collection name. Uses default if not provided.
            
        Returns:
            ChromaDB Collection object
        """
        collection_name = name or self.collection_name
        
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.debug(f"📚 Using collection: {collection_name} ({collection.count()} documents)")
        return collection
    
    @property
    def collection(self) -> chromadb.Collection:
        """Get the default collection (lazy-loaded)."""
        if self._collection is None:
            self._collection = self.get_collection()
        return self._collection
    
    def _generate_id(self, text: str, url: str = "", index: int = 0) -> str:
        """Generate a unique ID for a document."""
        content = f"{url}_{index}_{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the collection with embeddings.
        
        Args:
            texts: List of text content to embed and store
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs. Auto-generated if not provided.
            
        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("⚠️ No texts provided to add_documents")
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(text, "", i) for i, text in enumerate(texts)]
        
        # Ensure metadatas is same length as texts
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Generate embeddings in batch (5-10x faster than sequential)
        logger.info(f"🧠 Generating embeddings for {len(texts)} documents (batch mode)...")
        try:
            embeddings = get_embeddings_batch(texts, batch_size=32)
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            # Fallback to sequential on batch failure
            embeddings = []
            for text in texts:
                try:
                    emb = get_embedding(text)
                    embeddings.append(emb)
                except Exception:
                    embeddings.append([0.0] * 768)
        
        # Add to ChromaDB in batches (max batch size is 5461)
        total_docs = len(texts)
        added_count = 0
        
        for i in range(0, total_docs, CHROMA_BATCH_SIZE):
            batch_end = min(i + CHROMA_BATCH_SIZE, total_docs)
            batch_texts = texts[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            try:
                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
                added_count += len(batch_texts)
                logger.info(f"📦 Added batch {i // CHROMA_BATCH_SIZE + 1}: {len(batch_texts)} documents ({added_count}/{total_docs})")
            except Exception as e:
                logger.error(f"❌ Failed to add batch starting at {i}: {e}")
                raise
        
        logger.info(f"✅ Added {total_docs} documents to ChromaDB")
        return ids
    
    def upsert_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add or update documents in the collection.
        
        Args:
            texts: List of text content to embed and store
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs. Auto-generated if not provided.
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(text, "", i) for i, text in enumerate(texts)]
        
        # Ensure metadatas is same length as texts
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Generate embeddings in batch (5-10x faster than sequential)
        embeddings = get_embeddings_batch(texts, batch_size=32)
        
        # Upsert to ChromaDB in batches (max batch size is 5461)
        total_docs = len(texts)
        upserted_count = 0
        
        for i in range(0, total_docs, CHROMA_BATCH_SIZE):
            batch_end = min(i + CHROMA_BATCH_SIZE, total_docs)
            batch_texts = texts[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            self.collection.upsert(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            upserted_count += len(batch_texts)
            logger.info(f"📦 Upserted batch {i // CHROMA_BATCH_SIZE + 1}: {len(batch_texts)} documents ({upserted_count}/{total_docs})")
        
        logger.info(f"✅ Upserted {total_docs} documents to ChromaDB")
        return ids
    
    def similarity_search(
        self,
        query: str,
        n_results: int = 4,
        where: Optional[Dict] = None,
        include_distances: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text to find similar documents for
            n_results: Number of results to return
            where: Optional metadata filter
            include_distances: Whether to include similarity scores
            
        Returns:
            List of result dicts with text, metadata, and optional distance
        """
        try:
            # Generate query embedding
            query_embedding = get_embedding(query)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"] if include_distances else ["documents", "metadatas"],
            )
            
            # Format results
            formatted = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if include_distances else [None] * len(documents)
            ids = results.get("ids", [[]])[0]
            
            for i, (doc, meta, dist, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                result = {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "rank": i + 1,
                }
                if dist is not None:
                    # Convert distance to similarity score (1 - distance for cosine)
                    result["similarity"] = 1 - dist
                    result["distance"] = dist
                formatted.append(result)
            
            logger.debug(f"🔍 Found {len(formatted)} results for query")
            return formatted
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"🗑️ Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Delete all documents in the collection.
        
        Returns:
            True if successful
        """
        try:
            # Get all IDs
            all_data = self.collection.get()
            ids = all_data.get("ids", [])
            
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"🗑️ Cleared collection: deleted {len(ids)} documents")
            else:
                logger.info("📭 Collection was already empty")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dict with count and other stats
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory),
        }
    
    def ingest_from_scrape(self, scrape_data: Dict[str, Any], clear_existing: bool = False) -> int:
        """
        Ingest scraped website data into ChromaDB.
        
        Args:
            scrape_data: Scraped data dict (from ScraperStorage.load())
            clear_existing: Whether to clear existing documents first
            
        Returns:
            Number of documents added
        """
        if clear_existing:
            self.clear_collection()
        
        scrape_id = scrape_data.get("scrape_id", "unknown")
        pages = scrape_data.get("pages", [])
        
        texts = []
        metadatas = []
        ids = []
        
        for page in pages:
            url = page.get("url", "")
            title = page.get("title", "")
            
            for i, chunk in enumerate(page.get("chunks", [])):
                if not chunk or not chunk.strip():
                    continue
                
                doc_id = f"{scrape_id}_{hashlib.md5(f'{url}_{i}'.encode()).hexdigest()[:12]}"
                
                texts.append(chunk)
                metadatas.append({
                    "url": url,
                    "title": title,
                    "scrape_id": scrape_id,
                    "chunk_index": i,
                })
                ids.append(doc_id)
        
        if texts:
            self.add_documents(texts=texts, metadatas=metadatas, ids=ids)
        
        logger.info(f"📥 Ingested {len(texts)} chunks from scrape {scrape_id}")
        return len(texts)


# Convenience function
def get_chroma_manager() -> ChromaManager:
    """Get the singleton ChromaManager instance."""
    return ChromaManager.get_instance()
