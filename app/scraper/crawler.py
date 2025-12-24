"""
Web Crawler Module

Async web crawler using Playwright for JavaScript-rendered content.
Supports depth-based crawling with batch processing, rate limiting, and domain restrictions.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import ScraperConfig, get_scraper_config
from .content_extractor import ContentExtractor, PageData

logger = logging.getLogger(__name__)


class ScrapeLogger:
    """
    Detailed logging for scraping operations.
    
    Logs to both console and file for full transparency of scraping activities.
    """
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self._file_handler = None
        self._logger = logging.getLogger("scraper.detailed")
        
        if config.detailed_logging:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file handler for detailed logging."""
        try:
            log_file = Path(self.config.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            self._file_handler = logging.FileHandler(
                log_file, mode='a', encoding='utf-8'
            )
            self._file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._file_handler.setFormatter(formatter)
            self._logger.addHandler(self._file_handler)
            self._logger.setLevel(logging.DEBUG)
            
            logger.info(f"📝 Detailed logging enabled: {log_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup file logging: {e}")
    
    def log_start(self, base_urls: List[str], max_depth: int, max_pages: int, batch_size: int):
        """Log the start of a scraping session."""
        self._log("=" * 70)
        self._log("SCRAPE SESSION STARTED")
        self._log(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
        self._log(f"  Base URLs: {', '.join(base_urls)}")
        self._log(f"  Max Depth: {max_depth}")
        self._log(f"  Max Pages: {max_pages if max_pages > 0 else 'Unlimited'}")
        self._log(f"  Batch Size: {batch_size} (concurrent pages)")
        self._log(f"  Allowed Domains: {', '.join(self.config.allowed_domains)}")
        self._log("=" * 70)
    
    def log_batch_start(self, batch_num: int, batch_size: int, total_pages: int):
        """Log when starting a new batch."""
        self._log(f"[BATCH {batch_num}] Processing {batch_size} pages | Total scraped: {total_pages}")
    
    def log_page_start(self, url: str, depth: int, queue_size: int, page_num: int):
        """Log when starting to scrape a page."""
        self._log(f"[PAGE {page_num}] Depth={depth} | Queue={queue_size} | URL={url}")
    
    def log_page_success(self, url: str, title: str, chunks: int, links: int, time_ms: float):
        """Log successful page scrape."""
        self._log(
            f"[SUCCESS] {url}\n"
            f"  Title: {title[:80]}{'...' if len(title) > 80 else ''}\n"
            f"  Chunks: {chunks} | Links Found: {links} | Time: {time_ms:.0f}ms"
        )
    
    def log_page_error(self, url: str, error: str):
        """Log page scrape error."""
        self._log(f"[ERROR] {url}\n  Error: {error}", level="ERROR")
    
    def log_page_skipped(self, url: str, reason: str):
        """Log skipped page."""
        self._log(f"[SKIPPED] {url} - {reason}", level="DEBUG")
    
    def log_max_pages_reached(self, current: int, max_pages: int):
        """Log when max pages limit is reached."""
        self._log(f"[LIMIT] Scraped {current}/{max_pages} pages - stopping crawl", level="WARNING")
    
    def log_complete(self, pages: int, errors: int, urls_found: int, duration_sec: float):
        """Log completion of scraping session."""
        self._log("=" * 70)
        self._log("SCRAPE SESSION COMPLETED")
        self._log(f"  Pages Scraped: {pages}")
        self._log(f"  Errors: {errors}")
        self._log(f"  URLs Discovered: {urls_found}")
        self._log(f"  Duration: {duration_sec:.1f} seconds")
        self._log(f"  Avg Time/Page: {(duration_sec / max(pages, 1) * 1000):.0f}ms")
        self._log("=" * 70)
    
    def log_chunk_details(self, url: str, chunks: List[str]):
        """Log chunk details for a page (debug level)."""
        for i, chunk in enumerate(chunks):
            preview = chunk[:100].replace('\n', ' ')
            self._log(f"  [CHUNK {i+1}] {preview}...", level="DEBUG")
    
    def _log(self, message: str, level: str = "INFO"):
        """Write log message to both console and file."""
        log_func = getattr(self._logger, level.lower(), self._logger.info)
        log_func(message)
        
        # Also log to console via main logger for important items
        if level in ("WARNING", "ERROR"):
            console_func = getattr(logger, level.lower(), logger.info)
            console_func(message)
    
    def close(self):
        """Close file handler."""
        if self._file_handler:
            self._file_handler.close()
            self._logger.removeHandler(self._file_handler)


@dataclass
class CrawlResult:
    """Result of a complete crawl operation."""
    pages: Dict[str, PageData] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    total_urls_found: int = 0
    max_pages_reached: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_pages": len(self.pages),
            "total_errors": len(self.errors),
            "total_urls_found": self.total_urls_found,
            "max_pages_reached": self.max_pages_reached,
            "pages": [page.to_dict() for page in self.pages.values()],
            "errors": self.errors,
        }


@dataclass
class PageResult:
    """Result of processing a single page."""
    url: str
    depth: int
    success: bool
    page_data: Optional[PageData] = None
    error: Optional[str] = None
    time_ms: float = 0


class WebCrawler:
    """
    Async web crawler with Playwright for dynamic content rendering.
    
    Features:
    - Batch processing with configurable concurrency
    - Depth-based BFS crawling
    - JavaScript rendering via Playwright
    - Rate limiting to respect server resources
    - Domain whitelisting
    - URL deduplication and normalization
    - Maximum pages limit
    - Detailed operation logging
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the crawler.
        
        Args:
            config: Scraper configuration. Uses global config if not provided.
        """
        self.config = config or get_scraper_config()
        self.extractor = ContentExtractor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.scrape_logger = ScrapeLogger(self.config)
        
        # Semaphore for controlling concurrent page fetches
        self._semaphore = asyncio.Semaphore(self.config.batch_size)
        
        # Content hash tracking for deduplication
        self._content_hashes: Set[str] = set()
        
        # Browser will be initialized on first use
        self._playwright = None
        self._browser = None
        self._context = None
    
    def _get_content_hash(self, content: str) -> str:
        """Generate MD5 hash of content for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate_content(self, content: str) -> bool:
        """Check if content has been seen before (hash-based deduplication)."""
        if not self.config.enable_content_dedup:
            return False
        content_hash = self._get_content_hash(content)
        if content_hash in self._content_hashes:
            return True
        self._content_hashes.add(content_hash)
        return False
    
    def _is_content_valid(self, page_data) -> tuple:
        """Validate page content meets minimum requirements.
        
        Returns:
            (is_valid, reason) tuple
        """
        if len(page_data.content) < self.config.min_content_length:
            return False, f"Content too short ({len(page_data.content)} < {self.config.min_content_length} chars)"
        return True, ""
    
    async def _init_browser(self):
        """Initialize Playwright browser."""
        if self._browser is not None:
            return
        
        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless
            )
            self._context = await self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            logger.info("✅ Browser initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    async def _close_browser(self):
        """Close browser and cleanup resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("🔒 Browser closed")
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a page and return its HTML content.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if fetch failed
        """
        try:
            page = await self._context.new_page()
            page.set_default_timeout(self.config.page_timeout_ms)
            
            # Navigate to URL
            await page.goto(url, wait_until="domcontentloaded")
            
            # Wait for dynamic content if configured
            if self.config.wait_for_network_idle:
                try:
                    await page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    # Network idle timeout is acceptable, page may still have content
                    pass
            
            # Get final HTML after JS execution
            html = await page.content()
            await page.close()
            
            return html
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {url}: {e}")
            return None
    
    async def _process_single_page(
        self, 
        url: str, 
        depth: int,
        page_num: int,
        queue_size: int
    ) -> PageResult:
        """
        Process a single page with semaphore for concurrency control.
        
        Args:
            url: URL to process
            depth: Current crawl depth
            page_num: Page number for logging
            queue_size: Current queue size for logging
            
        Returns:
            PageResult with success/failure info
        """
        async with self._semaphore:
            page_start_time = time.time()
            
            # Log page start
            self.scrape_logger.log_page_start(url, depth, queue_size, page_num)
            
            # Fetch page
            html = await self._fetch_page(url)
            
            if html is None:
                return PageResult(
                    url=url,
                    depth=depth,
                    success=False,
                    error="Failed to fetch page",
                    time_ms=(time.time() - page_start_time) * 1000
                )
            
            # Extract content
            try:
                page_data = self.extractor.extract(html, url)
                page_time_ms = (time.time() - page_start_time) * 1000
                
                # Log success
                self.scrape_logger.log_page_success(
                    url, 
                    page_data.title, 
                    len(page_data.chunks), 
                    len(page_data.links),
                    page_time_ms
                )
                
                return PageResult(
                    url=url,
                    depth=depth,
                    success=True,
                    page_data=page_data,
                    time_ms=page_time_ms
                )
                
            except Exception as e:
                error_msg = str(e)
                self.scrape_logger.log_page_error(url, error_msg)
                
                return PageResult(
                    url=url,
                    depth=depth,
                    success=False,
                    error=error_msg,
                    time_ms=(time.time() - page_start_time) * 1000
                )
    
    def _should_visit(self, url: str, visited: Set[str]) -> bool:
        """
        Check if a URL should be visited.
        
        Args:
            url: URL to check
            visited: Set of already visited URLs
            
        Returns:
            True if URL should be visited
        """
        if url in visited:
            return False
        
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Check domain whitelist
        if self.config.allowed_domains:
            if parsed.netloc not in self.config.allowed_domains:
                return False
        
        # Skip common non-page URLs
        skip_extensions = {
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg',
            '.mp4', '.mp3', '.wav', '.avi', '.mov',
            '.zip', '.rar', '.tar', '.gz',
            '.css', '.js', '.json', '.xml',
        }
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False
        
        return True
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize a URL for comparison.
        
        Removes trailing slashes and fragments.
        """
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'
        
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        
        return normalized
    
    async def crawl(
        self, 
        base_urls: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        max_pages: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> CrawlResult:
        """
        Crawl websites starting from base URLs with batch processing.
        
        Uses asyncio.gather() to process multiple pages concurrently,
        controlled by a semaphore to limit concurrency.
        
        Args:
            base_urls: Starting URLs. Uses config if not provided.
            max_depth: Maximum crawl depth. Uses config if not provided.
            max_pages: Maximum pages to scrape. Uses config if not provided.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            CrawlResult with all extracted page data
        """
        urls = base_urls or self.config.base_urls
        depth = max_depth if max_depth is not None else self.config.max_depth
        page_limit = max_pages if max_pages is not None else self.config.max_pages
        batch_size = self.config.batch_size
        
        if not urls:
            logger.error("❌ No base URLs provided")
            return CrawlResult()
        
        start_time = time.time()
        result = CrawlResult(
            started_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Log start with batch info
        self.scrape_logger.log_start(urls, depth, page_limit, batch_size)
        
        # BFS queue: (url, current_depth)
        queue: List[Tuple[str, int]] = [(url, 0) for url in urls]
        visited: Set[str] = set()
        pages_scraped = 0
        batch_num = 0
        
        try:
            await self._init_browser()
            
            logger.info(f"🚀 Starting crawl of {len(urls)} base URL(s) with max depth {depth}")
            logger.info(f"⚡ Batch mode: {batch_size} concurrent pages")
            if page_limit > 0:
                logger.info(f"📊 Max pages limit: {page_limit}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("Crawling...", total=page_limit if page_limit > 0 else None)
                
                while queue:
                    # Check max pages limit
                    if page_limit > 0 and pages_scraped >= page_limit:
                        self.scrape_logger.log_max_pages_reached(pages_scraped, page_limit)
                        result.max_pages_reached = True
                        break
                    
                    # Build batch of pages to process
                    batch: List[Tuple[str, int]] = []
                    remaining = page_limit - pages_scraped if page_limit > 0 else batch_size
                    target_batch_size = min(batch_size, remaining)
                    
                    while queue and len(batch) < target_batch_size:
                        url, url_depth = queue.pop(0)
                        normalized_url = self._normalize_url(url)
                        
                        if not self._should_visit(normalized_url, visited):
                            self.scrape_logger.log_page_skipped(url, "Already visited or filtered")
                            continue
                        
                        visited.add(normalized_url)
                        batch.append((url, url_depth))
                    
                    if not batch:
                        continue
                    
                    batch_num += 1
                    self.scrape_logger.log_batch_start(batch_num, len(batch), pages_scraped)
                    progress.update(task, description=f"Batch {batch_num}: {len(batch)} pages...")
                    
                    # Process batch concurrently
                    tasks = [
                        self._process_single_page(
                            url, 
                            url_depth, 
                            pages_scraped + i + 1,
                            len(queue)
                        )
                        for i, (url, url_depth) in enumerate(batch)
                    ]
                    
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process batch results
                    for page_result in batch_results:
                        if isinstance(page_result, Exception):
                            logger.error(f"❌ Batch task exception: {page_result}")
                            continue
                        
                        pages_scraped += 1
                        
                        if page_result.success and page_result.page_data:
                            page_data = page_result.page_data
                            
                            # Validate content meets minimum requirements
                            is_valid, reason = self._is_content_valid(page_data)
                            if not is_valid:
                                self.scrape_logger.log_page_skipped(page_result.url, reason)
                                result.errors[page_result.url] = f"Skipped: {reason}"
                                continue
                            
                            # Check for duplicate content (hash-based)
                            if self._is_duplicate_content(page_data.content):
                                self.scrape_logger.log_page_skipped(page_result.url, "Duplicate content")
                                result.errors[page_result.url] = "Skipped: Duplicate content"
                                continue
                            
                            # Filter out short chunks
                            original_chunk_count = len(page_data.chunks)
                            page_data.chunks = [
                                chunk for chunk in page_data.chunks
                                if len(chunk.strip()) >= self.config.min_chunk_length
                            ]
                            filtered_count = original_chunk_count - len(page_data.chunks)
                            if filtered_count > 0:
                                logger.debug(f"Filtered {filtered_count} short chunks from {page_result.url}")
                            
                            result.pages[page_result.url] = page_data
                            
                            logger.info(
                                f"📄 [{pages_scraped}] [{page_result.depth}/{depth}] "
                                f"{page_result.url} - {len(page_data.chunks)} chunks"
                            )
                            
                            # Add discovered links to queue
                            if page_result.depth < depth:
                                for link in page_data.links:
                                    normalized_link = self._normalize_url(link)
                                    if self._should_visit(normalized_link, visited):
                                        queue.append((link, page_result.depth + 1))
                                        result.total_urls_found += 1
                        else:
                            result.errors[page_result.url] = page_result.error or "Unknown error"
                    
                    # Update progress
                    if page_limit > 0:
                        progress.update(task, completed=pages_scraped)
                    
                    # Batch delay (prevents overwhelming servers)
                    if self.config.batch_delay > 0 and queue:
                        await asyncio.sleep(self.config.batch_delay)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(pages_scraped, len(queue))
            
        finally:
            await self._close_browser()
            self.scrape_logger.close()
        
        result.completed_at = datetime.now(timezone.utc).isoformat()
        duration = time.time() - start_time
        
        # Log completion
        self.scrape_logger.log_complete(
            len(result.pages), 
            len(result.errors), 
            result.total_urls_found,
            duration
        )
        
        logger.info(
            f"✅ Crawl complete: {len(result.pages)} pages in {duration:.1f}s "
            f"({duration/max(len(result.pages),1):.1f}s/page avg)"
        )
        
        return result
    
    async def crawl_single(self, url: str) -> Optional[PageData]:
        """
        Crawl a single URL without following links.
        
        Args:
            url: URL to crawl
            
        Returns:
            PageData or None if fetch failed
        """
        try:
            await self._init_browser()
            html = await self._fetch_page(url)
            
            if html:
                return self.extractor.extract(html, url)
            return None
            
        finally:
            await self._close_browser()
