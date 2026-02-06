"""
Scraper Configuration Module

Environment-based configuration for the web scraper with sensible defaults.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ScraperConfig:
    """Configuration for the web scraper."""
    
    # Target URLs to scrape (comma-separated in env)
    base_urls: List[str] = field(default_factory=lambda: _parse_urls(
        os.getenv("SCRAPER_BASE_URLS", "")
    ))
    
    # Maximum crawl depth (0 = only base URL, 1 = base + direct links, etc.)
    max_depth: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_MAX_DEPTH", "300")
    ))
    
    # Maximum number of pages to scrape (0 = unlimited)
    max_pages: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_MAX_PAGES", "1500")
    ))
    
    # Delay between requests in seconds (respect rate limits)
    rate_limit_delay: float = field(default_factory=lambda: float(
        os.getenv("SCRAPER_RATE_LIMIT", "1.0")
    ))
    
    # Batch processing: number of pages to scrape concurrently
    batch_size: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_BATCH_SIZE", "10")  # Increased from 5 for faster crawling
    ))
    
    # Delay between batches in seconds (prevents overwhelming servers)
    batch_delay: float = field(default_factory=lambda: float(
        os.getenv("SCRAPER_BATCH_DELAY", "1.0")  # Reduced from 2.0
    ))
    
    # Domains to allow crawling (empty = allow all from base URLs)
    allowed_domains: List[str] = field(default_factory=lambda: _parse_urls(
        os.getenv("SCRAPER_ALLOWED_DOMAINS", "")
    ))
    
    # Output directory for scraped JSON files
    output_dir: Path = field(default_factory=lambda: Path(
        os.getenv("SCRAPER_OUTPUT_DIR", "./scraped_data")
    ))
    
    # Scraper log file for detailed operation logging
    log_file: Path = field(default_factory=lambda: Path(
        os.getenv("SCRAPER_LOG_FILE", "./scraped_data/scraper.log")
    ))
    
    # Enable detailed logging (logs every page, chunk, and operation)
    detailed_logging: bool = field(default_factory=lambda: 
        os.getenv("SCRAPER_DETAILED_LOGGING", "true").lower() == "true"
    )
    
    # Speed & Optimization settings
    use_sitemap: bool = field(default_factory=lambda:
        os.getenv("SCRAPER_USE_SITEMAP", "true").lower() == "true"
    )
    exclude_resources: List[str] = field(default_factory=lambda:
        os.getenv("SCRAPER_EXCLUDE_RESOURCES", "image,media,font,stylesheet").split(",")
    )
    
    # Interaction settings for dynamic content
    interaction_wait_ms: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_INTERACTION_WAIT_MS", "500")
    ))
    scroll_to_bottom: bool = field(default_factory=lambda:
        os.getenv("SCRAPER_SCROLL_TO_BOTTOM", "true").lower() == "true"
    )
    
    # ChromaDB settings
    chroma_db_path: Path = field(default_factory=lambda: Path(
        os.getenv("CHROMA_DB_PATH", "./chroma_db")
    ))
    chroma_collection_name: str = field(default_factory=lambda: 
        os.getenv("CHROMA_COLLECTION_NAME", "website_embeddings")
    )
    
    # Playwright settings
    headless: bool = field(default_factory=lambda: 
        os.getenv("SCRAPER_HEADLESS", "true").lower() == "true"
    )
    page_timeout_ms: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_PAGE_TIMEOUT_MS", "30000")
    ))
    wait_for_network_idle: bool = field(default_factory=lambda: 
        os.getenv("SCRAPER_WAIT_NETWORK_IDLE", "true").lower() == "true"
    )
    
    # Retry settings for transient network errors
    max_retries: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_MAX_RETRIES", "3")
    ))
    retry_base_delay: float = field(default_factory=lambda: float(
        os.getenv("SCRAPER_RETRY_BASE_DELAY", "1.0")  # Base delay in seconds (exponential: 1s, 2s, 4s)
    ))
    
    # Content extraction settings
    # Token-based chunking: 400 tokens chunk size, 80 tokens overlap
    # Approximate conversion: 1 token ≈ 4 characters for English text
    chunk_size: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_CHUNK_SIZE", "1600")  # 400 tokens * 4 chars
    ))
    chunk_overlap: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_CHUNK_OVERLAP", "320")  # 80 tokens * 4 chars
    ))
    
    # Validation & Deduplication settings
    # Minimum content length to accept a page (0 = accept all pages)
    min_content_length: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_MIN_CONTENT_LENGTH", "0")
    ))
    # Minimum chunk length to keep (lowered to keep service lists)
    min_chunk_length: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_MIN_CHUNK_LENGTH", "30")
    ))
    # Enable content hash deduplication (skip duplicate content across URLs)
    enable_content_dedup: bool = field(default_factory=lambda:
        os.getenv("SCRAPER_ENABLE_CONTENT_DEDUP", "true").lower() == "true"
    )
    
    # Scheduler settings
    schedule_day: str = field(default_factory=lambda: 
        os.getenv("SCRAPER_SCHEDULE_DAY", "sun")
    )
    schedule_hour: int = field(default_factory=lambda: int(
        os.getenv("SCRAPER_SCHEDULE_HOUR", "2")
    ))
    
    def __post_init__(self):
        """Validate and create directories."""
        self.output_dir = Path(self.output_dir)
        self.chroma_db_path = Path(self.chroma_db_path)
        
        # Auto-extract allowed domains from base URLs if not specified
        if not self.allowed_domains and self.base_urls:
            from urllib.parse import urlparse
            self.allowed_domains = list(set(
                urlparse(url).netloc for url in self.base_urls
            ))
    
    def ensure_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


def _parse_urls(url_string: str) -> List[str]:
    """Parse comma-separated URL string into list."""
    if not url_string:
        return []
    return [url.strip() for url in url_string.split(",") if url.strip()]


# Global config instance (lazy-loaded)
_config: Optional[ScraperConfig] = None


def get_scraper_config() -> ScraperConfig:
    """Get or create the global scraper config instance."""
    global _config
    if _config is None:
        _config = ScraperConfig()
    return _config
