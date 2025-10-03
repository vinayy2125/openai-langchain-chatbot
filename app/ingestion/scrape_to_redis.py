import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import yaml
import numpy as np
from redis.commands.search.field import TextField, VectorField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from app.config import get_redis
from core_services.generate_embeddings import get_embedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNIVERSAL_SESSION_ID = "universal_session_id"
CHUNK_EMBEDDING_THREADS = 20
BATCH_SIZE = 100


def create_index_from_yaml(yaml_path: str):
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ Could not load index config from {yaml_path}: {e}")
        return

    index_config = config.get("index", {})
    index_name = index_config.get("name", "chunk_index")
    prefix = index_config.get("prefix", "chunk:")
    fields_config = config.get("fields", [])

    if not fields_config:
        logger.error("❌ No 'fields' found in YAML.")
        return

    r = get_redis

    # Skip if index exists
    try:
        r.ft(index_name).info()
        logger.info(f"🔍 Index '{index_name}' already exists.")
        return
    except:
        pass

    redis_schema = []
    for field in fields_config:
        name = field.get("name")
        ftype = field.get("type", "").lower()
        if not name or not ftype:
            continue

        if ftype == "text":
            weight = field.get("weight", 1.0)
            redis_schema.append(TextField(f"$.{name}", as_name=name, weight=weight))
        elif ftype == "tag":
            redis_schema.append(TagField(f"$.{name}", as_name=name))
        elif ftype == "vector":
            attrs = field.get("attrs", {})
            vector_params = {
                "TYPE": attrs.get("dtype", "FLOAT32").upper(),
                "DIM": attrs["dims"],
                "DISTANCE_METRIC": attrs.get("distance_metric", "COSINE").upper(),
            }
            algorithm = attrs.get("algorithm", "FLAT").upper()
            if algorithm == "HNSW":
                if "initial_cap" in attrs:
                    vector_params["INITIAL_CAP"] = attrs["initial_cap"]
                if "M" in attrs:
                    vector_params["M"] = attrs["M"]
                if "ef_construction" in attrs:
                    vector_params["EF_CONSTRUCTION"] = attrs["ef_construction"]
            redis_schema.append(VectorField(f"$.{name}", algorithm, vector_params, as_name=name))
        else:
            logger.warning(f"⚠️ Unsupported field type: {ftype}")

    if not redis_schema:
        logger.error("❌ No valid fields to index.")
        return

    try:
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
        r.ft(index_name).create_index(redis_schema, definition=definition)
        logger.info(f"✅ Created RediSearch index '{index_name}' on prefix '{prefix}'")
    except Exception as e:
        logger.exception(f"❌ Failed to create index: {e}")
        raise


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
    return f"chunk_{index:06d}"


def embed_chunk_with_id(index: int, chunk_text: str, session_id: str) -> Dict[str, Any]:
    if not chunk_text or not chunk_text.strip():
        return None
    try:
        embedding = get_embedding(chunk_text)
        return {
            "chunk_id": generate_chunk_id(index),
            "text": chunk_text,
            "embedding": embedding,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to embed chunk {index}: {e}")
        return None


def store_chunk_document(chunk_id: str, data: Dict[str, Any]) -> None:
    """Store a single chunk as a Redis JSON document."""
    r = get_redis  # ← Not callable
    r.json().set(f"chunk:{chunk_id}", "$", data)


def embed_and_store_chunks_in_session(
    chunks: List[str],
    session_id: str = UNIVERSAL_SESSION_ID,
    overwrite: bool = True
) -> bool:
    total = len(chunks)
    if total == 0:
        logger.warning("⚠️ No chunks to process.")
        return True

    logger.info(f"🧠 Generating embeddings for {total} chunks using {CHUNK_EMBEDDING_THREADS} threads...")

    all_batches = [chunks[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    global_chunk_offset = 0

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
            indices = range(global_chunk_offset, global_chunk_offset + len(batch))
            embed_func = partial(embed_chunk_with_id, session_id=session_id)

            with ThreadPoolExecutor(max_workers=CHUNK_EMBEDDING_THREADS) as executor:
                futures = {
                    executor.submit(embed_func, idx, text): idx
                    for idx, text in zip(indices, batch)
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        store_chunk_document(result["chunk_id"], result)
                    progress.advance(task)

            global_chunk_offset += len(batch)

    logger.info(f"✅ Successfully stored {total} chunk documents for session '{session_id}'")
    return True


def ingest_website_data(json_file_path: str | Path) -> bool:
    try:
        # ✅ Correct path to app/db/user_message.yaml
        yaml_path = Path(__file__).parent.parent / "db" / "user_message.yaml"
        logger.info(f"📁 Using index config: {yaml_path.resolve()}")
        create_index_from_yaml(str(yaml_path))

        data = load_scraped_data(json_file_path)
        chunks = flatten_chunks(data)
        return embed_and_store_chunks_in_session(chunks)
    except Exception as e:
        logger.exception(f"💥 Ingestion failed: {e}")
        return False
