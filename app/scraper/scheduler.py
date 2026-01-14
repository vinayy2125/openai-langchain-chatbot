"""
Scraper Scheduler Module

APScheduler-based cron job scheduler for running weekly scrape jobs.
"""
import logging
from typing import Optional, Callable
from datetime import datetime, timezone
import asyncio

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, JobExecutionEvent

from .config import get_scraper_config, ScraperConfig
from .crawler import WebCrawler
from .storage import ScraperStorage
from app.db.chroma_manager import get_chroma_manager

logger = logging.getLogger(__name__)


class ScraperScheduler:
    """
    Scheduler for running scrape jobs on a weekly basis.
    
    Features:
    - Weekly cron scheduling (configurable day/hour)
    - Background thread execution
    - Job logging and error handling
    - Manual trigger support
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the scheduler.
        
        Args:
            config: Scraper configuration. Uses global config if not provided.
        """
        self.config = config or get_scraper_config()
        self.scheduler = BackgroundScheduler()
        self._job_id = "weekly_scrape"
        self._is_running = False
        
        # Add event listeners
        self.scheduler.add_listener(
            self._on_job_event,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
    
    def _on_job_event(self, event: JobExecutionEvent):
        """Handle job execution events."""
        if event.exception:
            logger.error(f"❌ Scrape job failed: {event.exception}")
        else:
            logger.info(f"✅ Scrape job completed successfully at {datetime.now()}")
    
    def _run_scrape_job(self):
        """Execute the scrape job (runs in background thread)."""
        logger.info("🚀 Starting scheduled scrape job...")
        
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the scraper
                crawler = WebCrawler(self.config)
                result = loop.run_until_complete(crawler.crawl())
                
                # Save results
                storage = ScraperStorage(self.config)
                filepath = storage.save(result)
                
                # Load and ingest into ChromaDB
                scrape_data = storage.load(filepath)
                chroma = get_chroma_manager()
                count = chroma.ingest_from_scrape(scrape_data, clear_existing=True)
                
                # Rebuild BM25 index after ingestion for hybrid search
                try:
                    from core_services.hybrid_search import get_hybrid_search_manager
                    hybrid = get_hybrid_search_manager()
                    hybrid.build_bm25_index(force_rebuild=True)
                    logger.info("✅ BM25 index rebuilt after scrape ingestion")
                except Exception as bm25_e:
                    logger.warning(f"BM25 index rebuild failed after scrape: {bm25_e}")
                
                # Cleanup old scrapes
                storage.cleanup_old_scrapes(keep_last=5)
                
                logger.info(
                    f"✅ Scrape job complete: {len(result.pages)} pages, "
                    f"{count} chunks indexed"
                )
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.exception(f"❌ Scrape job failed: {e}")
            raise
    
    def start(
        self,
        day_of_week: Optional[str] = None,
        hour: Optional[int] = None,
        run_immediately: bool = False,
    ):
        """
        Start the scheduler with weekly cron job.
        
        Args:
            day_of_week: Day to run (mon, tue, wed, thu, fri, sat, sun).
                        Uses config default if not provided.
            hour: Hour to run (0-23). Uses config default if not provided.
            run_immediately: If True, run the job once immediately.
        """
        if self._is_running:
            logger.warning("⚠️ Scheduler is already running")
            return
        
        day = day_of_week or self.config.schedule_day
        run_hour = hour if hour is not None else self.config.schedule_hour
        
        # Add weekly job
        trigger = CronTrigger(
            day_of_week=day,
            hour=run_hour,
            minute=0,
        )
        
        self.scheduler.add_job(
            self._run_scrape_job,
            trigger=trigger,
            id=self._job_id,
            name="Weekly Website Scrape",
            replace_existing=True,
        )
        
        self.scheduler.start()
        self._is_running = True
        
        logger.info(
            f"📅 Scheduler started: running every {day.upper()} at {run_hour:02d}:00"
        )
        
        if run_immediately:
            self.run_now()
    
    def stop(self):
        """Stop the scheduler."""
        if not self._is_running:
            return
        
        self.scheduler.shutdown(wait=False)
        self._is_running = False
        logger.info("🛑 Scheduler stopped")
    
    def run_now(self):
        """Trigger a scrape job immediately."""
        logger.info("🏃 Triggering immediate scrape job...")
        
        # Run in a new thread to not block
        import threading
        thread = threading.Thread(target=self._run_scrape_job, daemon=True)
        thread.start()
    
    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled run time."""
        job = self.scheduler.get_job(self._job_id)
        if job:
            return job.next_run_time
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running


async def run_scrape_once(
    urls: Optional[list] = None,
    depth: Optional[int] = None,
    update_chroma: bool = True,
) -> dict:
    """
    Run a one-time scrape operation.
    
    Args:
        urls: URLs to scrape. Uses config if not provided.
        depth: Crawl depth. Uses config if not provided.
        update_chroma: Whether to update ChromaDB with results.
        
    Returns:
        Dict with scrape results summary
    """
    config = get_scraper_config()
    
    # Override config if parameters provided
    if urls:
        from urllib.parse import urlparse
        config.base_urls = urls
        config.allowed_domains = [urlparse(u).netloc for u in urls]
    if depth is not None:
        config.max_depth = depth
    
    config.ensure_directories()
    
    # Run crawler
    crawler = WebCrawler(config)
    result = await crawler.crawl()
    
    # Save to JSON
    storage = ScraperStorage(config)
    filepath = storage.save(result)
    
    summary = {
        "pages_scraped": len(result.pages),
        "errors": len(result.errors),
        "total_urls_found": result.total_urls_found,
        "json_file": str(filepath),
        "chunks_indexed": 0,
    }
    
    # Update ChromaDB if requested
    if update_chroma:
        scrape_data = storage.load(filepath)
        chroma = get_chroma_manager()
        summary["chunks_indexed"] = chroma.ingest_from_scrape(scrape_data, clear_existing=True)
        
        # Rebuild BM25 index after ingestion for hybrid search
        try:
            from core_services.hybrid_search import get_hybrid_search_manager
            hybrid = get_hybrid_search_manager()
            hybrid.build_bm25_index(force_rebuild=True)
            logger.info("✅ BM25 index rebuilt after scrape ingestion")
        except Exception as bm25_e:
            logger.warning(f"BM25 index rebuild failed after scrape: {bm25_e}")
    
    return summary
