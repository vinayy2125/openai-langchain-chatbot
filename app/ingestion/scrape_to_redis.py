import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from app.config import get_redis
from core_services.generate_embeddings import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNIVERSAL_SESSION_ID = "universal_session_id"
CHUNK_EMBEDDING_THREADS = 8  # Adjust based on CPU cores
BATCH_SIZE = 100  # Process and upload in batches to avoid huge Redis ops


def load_scraped_data(file_path: str | Path) -> Dict[str, Any]:
    logger.info("📥 Loading scraped data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_chunks(scraped_data: Dict[str, Any]) -> List[str]:
    chunks = []
    for page in scraped_data.get("pages", []):
        chunks.extend(page.get("chunks", []))
    logger.info(f"✂️ Extracted {len(chunks)} chunks from scraped data.")
    return chunks


def generate_chunk_id(index: int) -> str:
    return f"chunk_{index:06d}"  # Support up to 1M chunks


def embed_chunk_with_id(index: int, chunk_text: str) -> Dict[str, Any]:
    """Embed a single chunk (used in thread pool)."""
    if not chunk_text or not chunk_text.strip():
        return None
    try:
        embedding = get_embedding(chunk_text)
        return {
            "query": generate_chunk_id(index),
            "query_embedding": embedding,
            "response": chunk_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to embed chunk {index}: {e}")
        return None


def store_session_batch(session_id: str, batch: List[Dict], is_first: bool, total_chunks: int):
    """Store or append a batch of chunks to Redis session."""
    r = get_redis
    key = f"session:{session_id}"

    try:
        if is_first:
            # Initialize session
            session_obj = {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "queries": batch
            }
            r.json().set(key, '$', session_obj)
        else:
            # Append batch
            r.json().arrappend(key, '$.queries', *batch)
    except Exception as e:
        logger.error(f"❌ Redis batch store failed (batch size={len(batch)}): {e}")
        raise


def embed_and_store_chunks_in_session(
    chunks: List[str],
    session_id: str = UNIVERSAL_SESSION_ID,
    overwrite: bool = True
) -> bool:
    if overwrite:
        r = get_redis
        r.delete(f"session:{session_id}")  # Ensure clean start

    total = len(chunks)
    if total == 0:
        logger.warning("⚠️ No chunks to process.")
        return True

    logger.info(f"🧠 Generating embeddings for {total} chunks using {CHUNK_EMBEDDING_THREADS} threads...")

    all_batches = []
    # Split into batches for memory & Redis efficiency
    for i in range(0, total, BATCH_SIZE):
        all_batches.append(chunks[i:i + BATCH_SIZE])

    batch_index = 0
    global_chunk_offset = 0

    # Use Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        refresh_per_second=2
    ) as progress:
        task = progress.add_task("Embedding & storing chunks...", total=total)

        for batch in all_batches:
            # Prepare chunk indices for this batch
            indices = range(global_chunk_offset, global_chunk_offset + len(batch))
            embed_func = partial(embed_chunk_with_id)

            batch_results = []
            with ThreadPoolExecutor(max_workers=CHUNK_EMBEDDING_THREADS) as executor:
                futures = {
                    executor.submit(embed_func, idx, text): idx
                    for idx, text in zip(indices, batch)
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        batch_results.append(result)
                    progress.advance(task)

            # Store batch in Redis
            if batch_results:
                is_first = (batch_index == 0)
                store_session_batch(session_id, batch_results, is_first, total)

            global_chunk_offset += len(batch)
            batch_index += 1

    logger.info(f"✅ Successfully stored {total} chunks in session '{session_id}'")
    return True


def ingest_website_data(json_file_path: str | Path) -> bool:
    try:
        data = load_scraped_data(json_file_path)
        chunks = flatten_chunks(data)
        return embed_and_store_chunks_in_session(chunks)
    except Exception as e:
        logger.exception(f"💥 Ingestion failed: {e}")
        return False