#!/usr/bin/env python
"""
ChromaDB Ingestion Script with ParentDocumentRetriever Support

Loads an existing scraped JSON file and ingests its chunks into ChromaDB
with parent-child linking for the ParentDocumentRetriever pattern.

Usage: 
    python scripts/ingest_to_chroma.py path/to/scrape_file.json [--append]
    python scripts/ingest_to_chroma.py path/to/scrape_file.json --with-parents  # Use parent retriever
"""
import sys
import argparse
import logging
import hashlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
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


def ingest_with_parent_retriever(scrape_data: dict, clear_existing: bool = True) -> int:
    """
    Ingest data using ParentDocumentRetriever for enhanced context retrieval.
    
    Creates dual-level chunks:
    - Child chunks (400 chars): Stored in ChromaDB for search
    - Parent chunks (2000 chars): Stored in ParentDocumentStore for context
    """
    from core_services.parent_document_retriever import get_parent_document_retriever
    
    retriever = get_parent_document_retriever()
    
    # Prepare documents from scraped pages
    documents = []
    pages = scrape_data.get("pages", [])
    
    console.print(f"[cyan]Processing {len(pages)} pages for parent-child chunking...[/cyan]")
    
    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        page_metadata = page.get("metadata", {})
        
        # Combine all chunks from the page into full content
        chunks = page.get("chunks", [])
        if not chunks:
            continue
            
        # Join chunks to form full page content
        full_content = "\n\n".join([c for c in chunks if c and c.strip()])
        
        if not full_content.strip():
            continue
        
        documents.append({
            "content": full_content,
            "url": url,
            "title": title,
            "metadata": {
                **page_metadata,
                "source_url": url,
                "scrape_id": scrape_data.get("scrape_id", ""),
            }
        })
    
    console.print(f"[cyan]Adding {len(documents)} documents with parent-child linking...[/cyan]")
    
    # Use the retriever's add_documents which handles parent-child splitting
    count = retriever.add_documents(documents, clear_existing=clear_existing)
    
    return count


def ingest_standard(scrape_data: dict, clear_existing: bool = True) -> int:
    """
    Standard ingestion using ChromaDB directly (backward compatible).
    """
    chroma = get_chroma_manager()
    count = chroma.ingest_from_scrape(scrape_data, clear_existing=clear_existing)
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest scraped JSON data into ChromaDB")
    parser.add_argument("file", nargs="?", help="Path to the scraped JSON file (optional, uses latest if not provided)")
    parser.add_argument("--append", action="store_true", help="Append to existing data instead of clearing")
    parser.add_argument("--with-parents", action="store_true", 
                       help="Use ParentDocumentRetriever for enhanced context (recommended)")
    parser.add_argument("--collection", help="Override default collection name")
    
    args = parser.parse_args()
    
    # Find file
    if args.file:
        file_path = Path(args.file)
    else:
        # Find latest scraped file
        config = get_scraper_config()
        scraped_files = list(config.output_dir.glob("scrape_*.json"))
        if not scraped_files:
            console.print("[red]❌ No scraped files found in scraped_data/[/red]")
            sys.exit(1)
        file_path = max(scraped_files, key=lambda f: f.stat().st_mtime)
        console.print(f"[yellow]Using latest scraped file: {file_path.name}[/yellow]")
    
    if not file_path.exists():
        console.print(f"[red]❌ Error: File not found at {file_path}[/red]")
        sys.exit(1)
        
    console.print(f"\n[bold cyan]📥 ChromaDB Ingestion[/bold cyan]")
    console.print(f"File: {file_path}")
    console.print(f"Mode: {'Append' if args.append else 'Replace'}")
    console.print(f"Method: {'ParentDocumentRetriever (enhanced context)' if args.with_parents else 'Standard'}")
    
    try:
        # 1. Load data
        console.print("\n[dim]Reading JSON file...[/dim]")
        config = get_scraper_config()
        storage = ScraperStorage(config)
        scrape_data = storage.load(file_path)
        
        pages = scrape_data.get("pages", [])
        console.print(f"Loaded [bold]{len(pages)}[/bold] pages from scrape")
        
        # 2. Ingest based on method
        if args.with_parents:
            console.print("\n[dim]Ingesting with parent-child chunking...[/dim]")
            count = ingest_with_parent_retriever(scrape_data, clear_existing=not args.append)
            
            # Show parent store stats
            from core_services.parent_document_store import get_parent_store
            parent_stats = get_parent_store().get_stats()
            console.print(f"\n[green]✅ Success![/green]")
            console.print(f"Child Chunks Indexed: [bold]{count}[/bold]")
            console.print(f"Parent Documents Created: [bold]{parent_stats['parent_count']}[/bold]")
        else:
            console.print("\n[dim]Ingesting with standard method...[/dim]")
            count = ingest_standard(scrape_data, clear_existing=not args.append)
            console.print(f"\n[green]✅ Success![/green]")
            console.print(f"Total Chunks Indexed: [bold]{count}[/bold]")
        
        # 3. Show ChromaDB stats
        chroma = get_chroma_manager()
        stats = chroma.get_stats()
        console.print(f"\n[bold]ChromaDB Stats:[/bold]")
        console.print(f"  Collection: {stats['collection_name']}")
        console.print(f"  Documents: {stats['document_count']}")
        console.print(f"  Client Mode: {stats.get('client_mode', 'unknown')}")
        console.print(f"  Healthy: {'✅' if stats.get('is_healthy', False) else '❌'}")
        
    except Exception as e:
        console.print(f"[red]❌ Ingestion failed: {e}[/red]")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
