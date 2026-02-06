#!/usr/bin/env python
"""
Web Scraper CLI

Entry point for running the web scraper with various modes:
- One-time scrape
- Weekly scheduled scraping
- Migration from Redis to ChromaDB
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)
console = Console()


def setup_config(args):
    """Configure scraper based on CLI arguments."""
    from app.scraper.config import get_scraper_config
    
    config = get_scraper_config()
    
    if args.url:
        from urllib.parse import urlparse
        config.base_urls = args.url
        config.allowed_domains = [urlparse(u).netloc for u in args.url]
    
    if args.depth is not None:
        config.max_depth = args.depth
    
    if hasattr(args, 'max_pages') and args.max_pages is not None:
        config.max_pages = args.max_pages
    
    if args.output:
        config.output_dir = Path(args.output)
    
    config.ensure_directories()
    return config


async def run_scrape(args):
    """Run a one-time scrape."""
    config = setup_config(args)
    
    if not config.base_urls:
        console.print("[red]❌ No URLs specified. Use --url or set SCRAPER_BASE_URLS env var[/red]")
        return 1
    
    console.print(f"\n[bold cyan]🕷️ Web Scraper[/bold cyan]")
    console.print(f"URLs: {', '.join(config.base_urls)}")
    console.print(f"Max Depth: {config.max_depth}")
    console.print(f"Max Pages: {config.max_pages if config.max_pages > 0 else 'Unlimited'}")
    console.print(f"Log File: {config.log_file}")
    console.print(f"Output: {config.output_dir}\n")
    
    from app.scraper.crawler import WebCrawler
    from app.scraper.storage import ScraperStorage
    from app.db.chroma_manager import get_chroma_manager
    
    # Run crawler
    crawler = WebCrawler(config)
    result = await crawler.crawl()
    
    # Save to JSON
    storage = ScraperStorage(config)
    filepath = storage.save(result)
    
    # Display results table
    table = Table(title="Scrape Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Pages Scraped", str(len(result.pages)))
    table.add_row("Errors", str(len(result.errors)))
    table.add_row("URLs Found", str(result.total_urls_found))
    table.add_row("Max Pages Reached", "Yes" if result.max_pages_reached else "No")
    table.add_row("JSON File", str(filepath))
    table.add_row("Log File", str(config.log_file))
    
    # Update ChromaDB if not skipped
    if not args.no_chroma:
        scrape_data = storage.load(filepath)
        chroma = get_chroma_manager()
        count = chroma.ingest_from_scrape(scrape_data, clear_existing=not args.append)
        table.add_row("Chunks Indexed", str(count))
    
    console.print(table)
    
    # Cleanup old scrapes
    if not args.keep_all:
        deleted = storage.cleanup_old_scrapes(keep_last=5)
        if deleted:
            console.print(f"[dim]Cleaned up {deleted} old scrape files[/dim]")
    
    return 0


def run_scheduler(args):
    """Start the weekly scheduler."""
    config = setup_config(args)
    
    if not config.base_urls:
        console.print("[red]❌ No URLs specified. Use --url or set SCRAPER_BASE_URLS env var[/red]")
        return 1
    
    console.print(f"\n[bold cyan]📅 Scraper Scheduler[/bold cyan]")
    console.print(f"URLs: {', '.join(config.base_urls)}")
    console.print(f"Schedule: Every {config.schedule_day.upper()} at {config.schedule_hour:02d}:00\n")
    
    from app.scraper.scheduler import ScraperScheduler
    
    scheduler = ScraperScheduler(config)
    scheduler.start(run_immediately=args.run_now)
    
    console.print("[green]✅ Scheduler started. Press Ctrl+C to stop.[/green]")
    
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(60)
            next_run = scheduler.get_next_run_time()
            if next_run:
                console.print(f"[dim]Next run: {next_run}[/dim]")
    except KeyboardInterrupt:
        scheduler.stop()
        console.print("\n[yellow]Scheduler stopped.[/yellow]")
    
    return 0


async def run_migration(args):
    """Migrate existing Redis vectors to ChromaDB."""
    console.print("\n[bold cyan]🔄 Redis to ChromaDB Migration[/bold cyan]\n")
    
    from app.config import get_redis
    from app.db.chroma_manager import get_chroma_manager
    from core_services.embedding_utils import get_embedding
    import json
    
    redis = get_redis
    chroma = get_chroma_manager()
    
    # Find all chunk:* keys in Redis
    console.print("Scanning Redis for chunk documents...")
    
    keys = []
    cursor = 0
    while True:
        cursor, batch = redis.scan(cursor, match="chunk:*", count=100)
        keys.extend(batch)
        if cursor == 0:
            break
    
    console.print(f"Found {len(keys)} chunk documents in Redis")
    
    if args.dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]")
        return 0
    
    if not keys:
        console.print("[yellow]No chunks to migrate[/yellow]")
        return 0
    
    # Migrate chunks
    texts = []
    metadatas = []
    ids = []
    errors = 0
    
    from rich.progress import Progress
    
    with Progress() as progress:
        task = progress.add_task("Migrating...", total=len(keys))
        
        for key in keys:
            try:
                # Get chunk data from Redis
                data = redis.json().get(key.decode() if isinstance(key, bytes) else key)
                
                if not data:
                    errors += 1
                    progress.advance(task)
                    continue
                
                text = data.get("text", "")
                chunk_id = data.get("chunk_id", key.decode() if isinstance(key, bytes) else key)
                
                if text:
                    texts.append(text)
                    metadatas.append({
                        "source": "redis_migration",
                        "original_chunk_id": chunk_id,
                        "session_id": data.get("session_id", "unknown"),
                    })
                    ids.append(f"migrated_{chunk_id}")
                
            except Exception as e:
                logger.warning(f"Failed to process {key}: {e}")
                errors += 1
            
            progress.advance(task)
    
    # Add to ChromaDB in batches
    if texts:
        console.print(f"Adding {len(texts)} documents to ChromaDB...")
        chroma.add_documents(texts=texts, metadatas=metadatas, ids=ids)
    
    # Show results
    table = Table(title="Migration Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Redis Keys Found", str(len(keys)))
    table.add_row("Documents Migrated", str(len(texts)))
    table.add_row("Errors", str(errors))
    table.add_row("ChromaDB Collection", chroma.collection_name)
    
    console.print(table)
    
    # Optionally delete Redis data
    if args.delete_redis:
        console.print("\n[yellow]Deleting migrated data from Redis...[/yellow]")
        for key in keys:
            try:
                redis.delete(key)
            except Exception:
                pass
        console.print(f"[green]Deleted {len(keys)} keys from Redis[/green]")
    
    return 0


def show_stats(args):
    """Show ChromaDB collection statistics."""
    from app.db.chroma_manager import get_chroma_manager
    
    chroma = get_chroma_manager()
    stats = chroma.get_stats()
    
    table = Table(title="ChromaDB Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        table.add_row(key, str(value))
    
    console.print(table)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Web Scraper with ChromaDB Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Run a one-time scrape")
    scrape_parser.add_argument("--url", "-u", nargs="+", help="URLs to scrape")
    scrape_parser.add_argument("--depth", "-d", type=int, help="Maximum crawl depth")
    scrape_parser.add_argument("--max-pages", "-m", type=int, help="Maximum pages to scrape (0=unlimited)")
    scrape_parser.add_argument("--output", "-o", help="Output directory")
    scrape_parser.add_argument("--no-chroma", action="store_true", help="Skip ChromaDB indexing")
    scrape_parser.add_argument("--append", action="store_true", help="Append to existing ChromaDB data")
    scrape_parser.add_argument("--keep-all", action="store_true", help="Keep all scrape files")
    
    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Start weekly scheduler")
    schedule_parser.add_argument("--url", "-u", nargs="+", help="URLs to scrape")
    schedule_parser.add_argument("--depth", "-d", type=int, help="Maximum crawl depth")
    schedule_parser.add_argument("--output", "-o", help="Output directory")
    schedule_parser.add_argument("--run-now", action="store_true", help="Run immediately then schedule")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate Redis vectors to ChromaDB")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    migrate_parser.add_argument("--delete-redis", action="store_true", help="Delete Redis data after migration")
    
    # Stats command
    subparsers.add_parser("stats", help="Show ChromaDB statistics")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "scrape":
        return asyncio.run(run_scrape(args))
    elif args.command == "schedule":
        return run_scheduler(args)
    elif args.command == "migrate":
        return asyncio.run(run_migration(args))
    elif args.command == "stats":
        return show_stats(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
