#!/usr/bin/env python
"""
ChromaDB Ingestion Script

Loads an existing scraped JSON file and ingests its chunks into ChromaDB.
Usage: python scripts/ingest_to_chroma.py path/to/scrape_file.json [--append]
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from app.scraper.config import get_scraper_config
from app.scraper.storage import ScraperStorage
from app.db.chroma_manager import get_chroma_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)]
)
logger = logging.getLogger(__name__)
console = Console()

def main():
    parser = argparse.ArgumentParser(description="Ingest scraped JSON data into ChromaDB")
    parser.add_argument("file", help="Path to the scraped JSON file")
    parser.add_argument("--append", action="store_true", help="Append to existing data instead of clearing")
    parser.add_argument("--collection", help="Override default collection name")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]❌ Error: File not found at {file_path}[/red]")
        sys.exit(1)
        
    console.print(f"\n[bold cyan]📥 ChromaDB Ingestion[/bold cyan]")
    console.print(f"File: {file_path}")
    console.print(f"Mode: {'Append' if args.append else 'Replace'}")
    
    try:
        # 1. Load data
        console.print("Reading JSON file...")
        config = get_scraper_config()
        storage = ScraperStorage(config)
        scrape_data = storage.load(file_path)
        
        # 2. Initialize Chroma
        console.print("Initializing ChromaDB...")
        chroma = get_chroma_manager()
        
        if args.collection:
            # If a custom collection is specified, we might need to handle it manually
            # But for now we use the manager's default logic
            pass
            
        # 3. Ingest
        console.print(f"Ingesting chunks (this may take a while if embeddings need generation)...")
        count = chroma.ingest_from_scrape(
            scrape_data, 
            clear_existing=not args.append
        )
        
        console.print(f"\n[green]✅ Success![/green]")
        console.print(f"Total Chunks Indexed: [bold]{count}[/bold]")
        
        # 4. Show stats
        stats = chroma.get_stats()
        console.print(f"Current Collection Size: [bold]{stats['document_count']}[/bold] chunks")
        
    except Exception as e:
        console.print(f"[red]❌ Ingestion failed: {e}[/red]")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
