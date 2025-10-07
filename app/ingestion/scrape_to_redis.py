import json
import logging
from redis.exceptions import ResponseError
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
CHUNK_EMBEDDING_THREADS = 12
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

    # get_redis in config returns a connected redis client instance
    r = get_redis

    # Check if required Redis modules are loaded
    try:
        # Use separate args; some redis-py versions expect the command and args separately
        modules = r.execute_command('MODULE', 'LIST')
        # modules is typically a list of module-info arrays; make a safe string representation
        modules_normalized = []
        for mod in modules:
            try:
                if isinstance(mod, (bytes, bytearray)):
                    modules_normalized.append(mod.decode('utf-8', errors='ignore').lower())
                elif isinstance(mod, (list, tuple)):
                    # join parts into a single string for easier matching
                    parts = []
                    for part in mod:
                        if isinstance(part, (bytes, bytearray)):
                            parts.append(part.decode('utf-8', errors='ignore'))
                        else:
                            parts.append(str(part))
                    modules_normalized.append(" ".join(parts).lower())
                else:
                    modules_normalized.append(str(mod).lower())
            except Exception:
                modules_normalized.append(str(mod).lower())

        redisearch_loaded = any('search' in m for m in modules_normalized)
        redisjson_loaded = any('rejson' in m for m in modules_normalized)

        if not redisearch_loaded or not redisjson_loaded:
            missing = []
            if not redisearch_loaded:
                missing.append("RediSearch")
            if not redisjson_loaded:
                missing.append("RedisJSON")
            logger.error(f"❌ Required Redis modules not loaded: {', '.join(missing)}. Please ensure redis-stack is properly configured. MODULE LIST returned: {modules_normalized}")
            return False

        logger.info("✅ Required Redis modules (RediSearch, RedisJSON) are loaded")
    except Exception as e:
        # Log full exception for easier debugging (some Redis responses can be numeric or unexpected)
        logger.error(f"❌ Failed to check Redis modules: {e}")
        return False

    # Check if index exists
    index_exists = False
    try:
        r.ft(index_name).info()
        logger.info(f"🔍 Index '{index_name}' already exists")
        index_exists = True
    except ResponseError as e:
        if "unknown command" in str(e).lower():
            logger.error("❌ RediSearch module not properly initialized")
            return False
        logger.info(f"🏗️ Index '{index_name}' does not exist, will create it")
    except Exception as e:
        logger.error(f"❌ Error checking index: {e}")
        return False

    # If index doesn't exist, create it and build schema
    if not index_exists:
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
                # Validate required vector attributes
                if "dims" not in attrs:
                    logger.error(f"❌ Vector field '{name}' missing required 'dims' attribute")
                    continue
                vector_params = {
                    "TYPE": attrs.get("dtype", "FLOAT32").upper(),
                    "DIM": int(attrs["dims"]),
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

    # If we didn't build a schema because index already exists, nothing to do
    if index_exists:
        logger.info(f"ℹ️ Index '{index_name}' already present; skipping creation.")
        return True

    if not redis_schema:
        logger.error("❌ No valid fields to index.")
        return False

    try:
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
        r.ft(index_name).create_index(redis_schema, definition=definition)
        logger.info(f"✅ Created RediSearch index '{index_name}' on prefix '{prefix}'")
        return True
    except Exception as e:
        if "Index already exists" in str(e):
            logger.info(f"🔍 Index '{index_name}' already exists.")
            return True
        else:
            logger.exception(f"❌ Failed to create index: {e}")
            return False


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


def embed_chunk_with_id(index: int, chunk_text: str, session_id: str) -> Dict[str, Any] | None:
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


def store_chunk_document(chunk_id: str, data: Dict[str, Any]) -> bool:
    """Store a single chunk as a Redis JSON document."""
    try:
        r = get_redis
        # Verify RedisJSON functionality
        key = f"chunk:{chunk_id}"
        r.json().set(key, "$", data)
        return True
    except ResponseError as e:
        if "unknown command" in str(e).lower():
            logger.error("❌ RedisJSON module not available. Please ensure redis-stack is properly configured.")
        else:
            logger.error(f"❌ Failed to store chunk {chunk_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error storing chunk {chunk_id}: {e}")
        return False


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
                        if not store_chunk_document(result["chunk_id"], result):
                            logger.error(f"❌ Failed to store chunk {result['chunk_id']}")
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
