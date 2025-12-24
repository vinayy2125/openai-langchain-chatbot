"""
Scraper Storage Module

Handles saving and loading scraped data as JSON files with
proper timestamping and metadata.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

from .config import ScraperConfig, get_scraper_config
from .crawler import CrawlResult

logger = logging.getLogger(__name__)


class ScraperStorage:
    """
    JSON file storage for scraped website data.
    
    Features:
    - Automatic timestamped filenames
    - Structured JSON format with metadata
    - Load/list historical scrapes
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize storage.
        
        Args:
            config: Scraper configuration. Uses global config if not provided.
        """
        self.config = config or get_scraper_config()
        self.config.ensure_directories()
    
    def save(self, crawl_result: CrawlResult, filename: Optional[str] = None) -> Path:
        """
        Save crawl result to JSON file.
        
        Args:
            crawl_result: Result from WebCrawler.crawl()
            filename: Optional custom filename. Auto-generated if not provided.
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"scrape_{timestamp}.json"
        
        filepath = self.config.output_dir / filename
        
        # Build complete document
        document = {
            "scrape_id": str(uuid.uuid4()),
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "base_urls": self.config.base_urls,
                "max_depth": self.config.max_depth,
                "chunk_size": self.config.chunk_size,
            },
            **crawl_result.to_dict()
        }
        
        # Write to file with pretty formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved scrape data to {filepath}")
        return filepath
    
    def load(self, filepath: Path) -> Dict[str, Any]:
        """
        Load scraped data from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"📂 Loaded scrape data from {filepath}")
        return data
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent scrape data.
        
        Returns:
            Most recent scrape data or None if no scrapes exist
        """
        scrapes = self.list_scrapes()
        if not scrapes:
            return None
        
        # Sort by modification time (most recent first)
        scrapes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return self.load(scrapes[0])
    
    def list_scrapes(self) -> List[Path]:
        """
        List all scrape files in the output directory.
        
        Returns:
            List of paths to scrape JSON files
        """
        if not self.config.output_dir.exists():
            return []
        
        return list(self.config.output_dir.glob("scrape_*.json"))
    
    def get_all_chunks(self, data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract all chunks from scrape data with metadata.
        
        Args:
            data: Scrape data dict. Uses latest scrape if not provided.
            
        Returns:
            List of chunk dicts with text and metadata
        """
        if data is None:
            data = self.get_latest()
            if data is None:
                return []
        
        chunks = []
        scrape_id = data.get("scrape_id", "unknown")
        
        for page in data.get("pages", []):
            url = page.get("url", "")
            title = page.get("title", "")
            
            for i, chunk_text in enumerate(page.get("chunks", [])):
                chunks.append({
                    "id": f"{scrape_id}_{hash(url)}_{i}",
                    "text": chunk_text,
                    "metadata": {
                        "url": url,
                        "title": title,
                        "scrape_id": scrape_id,
                        "chunk_index": i,
                    }
                })
        
        return chunks
    
    def cleanup_old_scrapes(self, keep_last: int = 5) -> int:
        """
        Remove old scrape files, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent scrapes to keep
            
        Returns:
            Number of files deleted
        """
        scrapes = self.list_scrapes()
        if len(scrapes) <= keep_last:
            return 0
        
        # Sort by modification time (oldest first)
        scrapes.sort(key=lambda p: p.stat().st_mtime)
        
        to_delete = scrapes[:-keep_last]
        deleted = 0
        
        for filepath in to_delete:
            try:
                filepath.unlink()
                deleted += 1
                logger.info(f"🗑️ Deleted old scrape: {filepath.name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {filepath}: {e}")
        
        return deleted
