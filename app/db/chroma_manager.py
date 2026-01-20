"""
ChromaDB Manager Module

Central manager for ChromaDB vector database operations.
Replaces Redis vector storage for RAG functionality.

Supports two modes:
- HttpClient: Connects to external ChromaDB server (production)
- PersistentClient: Embedded local storage (development)
"""
import logging
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

import chromadb
from chromadb.config import Settings

from core_services.embedding_utils import get_embedding, get_embeddings_batch
from app.scraper.config import get_scraper_config, ScraperConfig

logger = logging.getLogger(__name__)

# ChromaDB max batch size (default is 5461)
CHROMA_BATCH_SIZE = 500  # Reduced to 500 for better stability with Docker

# Configuration for HttpClient
CHROMA_SERVER_URL = os.getenv("CHROMA_SERVER_URL", "")
CHROMA_USE_HTTP_CLIENT = os.getenv("CHROMA_USE_HTTP_CLIENT", "false").lower() == "true"


class ChromaManager:
    """
    Manages ChromaDB collections for storing and querying embeddings.
    
    Features:
    - Dual client support: HttpClient (production) or PersistentClient (dev)
    - Persistent storage on disk or remote server
    - Uses existing embedding_utils for vector generation
    - Semantic similarity search
    - Collection management (create, update, delete)
    - Connection retry logic for HttpClient
    """
    
    _instance: Optional["ChromaManager"] = None
    
    def __init__(self, persist_directory: Optional[str] = None, server_url: Optional[str] = None):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Path for persistent storage (PersistentClient mode).
                             Uses config default if not provided.
            server_url: Optional URL for ChromaDB server (HttpClient mode).
                       If provided, uses HttpClient instead of PersistentClient.
        """
        config = get_scraper_config()
        
        self.collection_name = config.chroma_collection_name
        self._collection = None
        
        # Determine which client to use
        effective_server_url = server_url or CHROMA_SERVER_URL
        use_http = CHROMA_USE_HTTP_CLIENT or bool(effective_server_url)
        
        if use_http and effective_server_url:
            # Use HttpClient for production (separate server process)
            try:
                self.client = self._create_http_client(effective_server_url)
                self.persist_directory = None
                self.client_mode = "http"
                logger.info(f"✅ ChromaDB initialized with HttpClient at {effective_server_url}")
                return  # Successfully initialized HTTP client
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect to ChromaDB server at {effective_server_url}: {e}")
                logger.warning("🔄 Falling back to local PersistentClient...")
                # Fall through to PersistentClient initialization
        
        # Use PersistentClient for local development (embedded) or fallback
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = config.chroma_db_path
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        self.client_mode = "persistent"
        logger.info(f"✅ ChromaDB initialized with PersistentClient at {self.persist_directory}")
    
    def _create_http_client(self, server_url: str, max_retries: int = 3) -> chromadb.HttpClient:
        """
        Create HttpClient with retry logic for connection resilience.
        
        Args:
            server_url: ChromaDB server URL (e.g., http://localhost:8000)
            max_retries: Number of connection retries
            
        Returns:
            chromadb.HttpClient instance
        """
        # Parse URL to extract host and port
        from urllib.parse import urlparse
        parsed = urlparse(server_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000
        
        for attempt in range(max_retries):
            try:
                client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(
                        anonymized_telemetry=False,
                    )
                )
                # Test connection with heartbeat
                client.heartbeat()
                logger.info(f"ChromaDB HttpClient connected to {host}:{port}")
                return client
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"ChromaDB connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to ChromaDB server at {server_url}")
                    raise
    
    def is_healthy(self) -> bool:
        """Check if ChromaDB connection is healthy."""
        try:
            if self.client_mode == "http":
                self.client.heartbeat()
            else:
                # For PersistentClient, try to access collection
                self.collection.count()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False
    
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
            embeddings = get_embeddings_batch(texts, batch_size=32)  # Optimized for 8GB GPU
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
        stats = {
            "collection_name": self.collection_name,
            "document_count": count,
            "client_mode": self.client_mode,
            "is_healthy": self.is_healthy(),
        }
        if self.persist_directory:
            stats["persist_directory"] = str(self.persist_directory)
        return stats
    
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
            page_metadata = page.get("metadata", {})
            
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
                    # Enhanced metadata from page extraction
                    "page_type": page_metadata.get("page_type", "general"),
                    "service_category": page_metadata.get("service_category", ""),
                    "scraped_at": scrape_data.get("created_at", ""),
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
