"""
Redis prompts helper

Provides a single public function `load_prompts(redis_conn, limit=10, retries=3, backoff=0.5)` which:
- Validates the provided `redis_conn` by pinging it.
- Ensures a RediSearch index named `chat_prompts` exists (creates if missing).
- Attempts to fetch latest prompts using three strategies (in order):
  a) Sorted set `chat_prompts:z` (ZREVRANGE -> lookup hashes)
  b) RediSearch FT.SEARCH ordering by `created_at` (if index available)
  c) Hash keys `chat_prompt:{id}` scanned via SCAN
- If no prompts are found in Redis, falls back to `app.utils.prompts` and looks
  for a module-level variable like `DEFAULT_PROMPTS` or `SAMPLE_PROMPTS`.

Expected prompt dict shape for returned items:
  {"id": "<id>", "prompt": "<text>", "created_at": <unix_ts_or_numeric>, "source": "optional", "lang": "optional"}

The module uses retries with exponential backoff for transient Redis errors
and logs structured messages for observability.
"""
from __future__ import annotations

import logging
import json
import time
from typing import Any, Dict, List, Optional, Sequence
import importlib

# Import the new helper for assistant instructions
from app.db.assistant_instructions_helper import get_assistant_instruction_by_key

INDEX_NAME = "chat_prompts"
SORTED_SET_KEY = "chat_prompts:z"
HASH_PREFIX = "chat_prompt:"

logger = logging.getLogger("redis_prompts")


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _retry(func):
    def wrapper(*args, retries: int = 3, backoff: float = 0.5, **kwargs):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = backoff * (2 ** (attempt - 1))
                logger.warning("Transient error (attempt %s/%s): %s - retrying in %.2fs",
                               attempt, retries, exc, wait)
                time.sleep(wait)
        logger.error("Operation failed after %s attempts: %s", retries, last_exc)
        # mypy / pylance may warn that last_exc could be None; ensure we raise a proper Exception
        if last_exc is None:
            raise RuntimeError(f"Operation failed after {retries} attempts")
        raise last_exc

    return wrapper


def _validate_redis_conn(redis_conn) -> None:
    if redis_conn is None:
        raise ValueError("redis_conn is required")
    try:
        # decode_responses may be False in this project; ping returns True on success
        redis_conn.ping()
    except Exception as exc:
        logger.error("Redis ping failed: %s", exc)
        raise


INDEX_JSON_NAME = "chat_prompts_json"


def ensure_index(redis_conn, recreate: bool = False, json_index: bool = False) -> bool:
    """Ensure the RediSearch index exists. Returns True if created, False if skipped.

    If RediSearch client helpers are available on the `redis_conn` (i.e., `redis_conn.ft`),
    attempt to use them; otherwise fall back to raw `FT.INFO` / `FT.CREATE` commands.
    """
    try:
        # attempt to use redis-py helpers for same-index checks when available
        if json_index:
            target_name = INDEX_JSON_NAME
            prefix = "chat_prompt_json:"
            # Try high-level FT helper if available
            try:
                if hasattr(redis_conn, "ft"):
                    # redis-py may not have a high-level JSON index builder; use raw commands below
                    redis_conn.execute_command("FT.INFO", target_name)
                    logger.info("JSON index `%s` already exists - skipping creation", target_name)
                    return False
            except Exception:
                pass

            # Create JSON index via raw FT.CREATE ON JSON
            try:
                redis_conn.execute_command("FT.INFO", target_name)
                logger.info("JSON index `%s` already exists (raw) - skipping creation", target_name)
                return False
            except Exception:
                try:
                    cmd = [
                        "FT.CREATE",
                        target_name,
                        "ON",
                        "JSON",
                        "PREFIX",
                        "1",
                        prefix,
                        "SCHEMA",
                        "$.prompt",
                        "AS",
                        "prompt",
                        "TEXT",
                        "$.created_at",
                        "AS",
                        "created_at",
                        "NUMERIC",
                        "SORTABLE",
                        "$.source",
                        "AS",
                        "source",
                        "TAG",
                        "$.lang",
                        "AS",
                        "lang",
                        "TAG",
                    ]
                    redis_conn.execute_command(*cmd)
                    logger.info("Created RediSearch JSON index `%s` via raw command", target_name)
                    return True
                except Exception as exc2:
                    logger.error("Failed to create RediSearch JSON index: %s", exc2)
                    raise
        else:
            # Hash index (existing behavior)
            try:
                if hasattr(redis_conn, "ft"):
                    redis_conn.ft(INDEX_NAME).info()
                    logger.info("Index `%s` already exists - skipping creation", INDEX_NAME)
                    return False
            except Exception:
                pass

            # Try using redis-py schema helpers
            try:
                # Import redis-py search helpers dynamically to avoid static import complaints
                try:
                    field_mod = importlib.import_module("redis.commands.search.field")
                    idxdef_mod = importlib.import_module("redis.commands.search.indexDefinition")
                    TextField = getattr(field_mod, "TextField")
                    NumericField = getattr(field_mod, "NumericField")
                    TagField = getattr(field_mod, "TagField")
                    IndexDefinition = getattr(idxdef_mod, "IndexDefinition")
                    IndexType = getattr(idxdef_mod, "IndexType")

                    schema = [
                        TextField("id", sortable=True),
                        TextField("prompt"),
                        NumericField("created_at", sortable=True),
                        TagField("source"),
                        TagField("lang"),
                    ]
                    definition = IndexDefinition(prefix=[HASH_PREFIX], index_type=IndexType.HASH)
                    redis_conn.ft(INDEX_NAME).create_index(schema, definition=definition)
                    logger.info("Created RediSearch index `%s`", INDEX_NAME)
                    return True
                except Exception:
                    # Fall through to raw-command based creation below
                    pass
            except Exception:
                # Fallback to raw commands
                try:
                    redis_conn.execute_command("FT.INFO", INDEX_NAME)
                    logger.info("Index `%s` already exists (raw) - skipping creation", INDEX_NAME)
                    return False
                except Exception:
                    try:
                        cmd = [
                            "FT.CREATE",
                            INDEX_NAME,
                            "ON",
                            "HASH",
                            "PREFIX",
                            "1",
                            HASH_PREFIX,
                            "SCHEMA",
                            "id",
                            "TEXT",
                            "SORTABLE",
                            "prompt",
                            "TEXT",
                            "created_at",
                            "NUMERIC",
                            "SORTABLE",
                            "source",
                            "TAG",
                            "lang",
                            "TAG",
                        ]
                        redis_conn.execute_command(*cmd)
                        logger.info("Created RediSearch index `%s` via raw command", INDEX_NAME)
                        return True
                    except Exception as exc2:
                        logger.error("Failed to create RediSearch index: %s", exc2)
                        raise
    except Exception as exc:
        logger.error("ensure_index unexpected error: %s", exc)
        raise


def _parse_hash(redis_conn, key: str) -> Optional[Dict[str, Any]]:
    try:
        data = redis_conn.hgetall(key)
        if not data:
            return None
        # Ensure bytes -> decode if necessary
        parsed = {}
        for k, v in data.items():
            if isinstance(k, bytes):
                k = k.decode()
            if isinstance(v, bytes):
                try:
                    v = v.decode()
                except Exception:
                    pass
            parsed[k] = v

        # normalize fields
        pid = parsed.get("id") or key.replace(HASH_PREFIX, "")
        prompt = parsed.get("prompt") or parsed.get("text")
        created_at = _to_float(parsed.get("created_at"), None)

        result = {"id": pid, "prompt": prompt, "created_at": created_at}
        # include optional metadata
        if parsed.get("source"):
            result["source"] = parsed.get("source")
        if parsed.get("lang"):
            result["lang"] = parsed.get("lang")
        return result
    except Exception as exc:
        logger.warning("Failed to parse hash %s: %s", key, exc)
        return None


@_retry
def _fetch_from_sorted_set(redis_conn, limit: int) -> List[Dict[str, Any]]:
    if not redis_conn.exists(SORTED_SET_KEY):
        return []
    members = redis_conn.zrevrange(SORTED_SET_KEY, 0, limit - 1)
    results: List[Dict[str, Any]] = []
    for member in members:
        # member may be bytes or string; assume it's an id or a key
        if isinstance(member, bytes):
            member = member.decode()
        # If member looks like full key, use it, otherwise prepend prefix
        if member.startswith(HASH_PREFIX):
            key = member
        else:
            key = f"{HASH_PREFIX}{member}"
        parsed = _parse_hash(redis_conn, key)
        if parsed:
            results.append(parsed)
        if len(results) >= limit:
            break
    return results


@_retry
def _fetch_from_redisearch(redis_conn, limit: int) -> List[Dict[str, Any]]:
    try:
        # Try high-level client first
        try:
            res = redis_conn.ft(INDEX_NAME).search("*", sort_by=[("created_at", False)], limit=limit)
            docs = []
            for doc in getattr(res, "docs", []):
                fields = getattr(doc, "__dict__", {})
                docid = getattr(doc, "id", None) or fields.get("id")
                doc_fields = {k: v for k, v in fields.items() if not k.startswith("_")}
                created_at = _to_float(doc_fields.get("created_at"), None)
                parsed = {"id": docid, "prompt": doc_fields.get("prompt"), "created_at": created_at}
                if doc_fields.get("source"):
                    parsed["source"] = doc_fields.get("source")
                if doc_fields.get("lang"):
                    parsed["lang"] = doc_fields.get("lang")
                docs.append(parsed)
            return docs[:limit]
        except Exception:
            # Fallback to FT.SEARCH raw
            args: List[Any] = ["FT.SEARCH", INDEX_NAME, "*", "SORTBY", "created_at", "DESC", "LIMIT", "0", str(limit)]
            raw = redis_conn.execute_command(*args)
            # raw format: [total, docid1, [field, val, ...], docid2, [field, val, ...], ...]
            if not raw or len(raw) < 2:
                return []
            total = raw[0]
            docs: List[Dict[str, Any]] = []
            i = 1
            while i < len(raw):
                docid = raw[i]
                fields = raw[i + 1]
                parsed = {"id": docid}
                # fields is list [k1, v1, k2, v2]
                for j in range(0, len(fields), 2):
                    k = fields[j]
                    v = fields[j + 1]
                    if isinstance(k, bytes):
                        k = k.decode()
                    if isinstance(v, bytes):
                        v = v.decode()
                    parsed[k] = v
                # Normalize created_at safely
                parsed["created_at"] = _to_float(parsed.get("created_at"), None)
                docs.append({"id": str(parsed.get("id")), "prompt": parsed.get("prompt"), "created_at": parsed.get("created_at"), "source": parsed.get("source"), "lang": parsed.get("lang")})
                i += 2
            return docs[:limit]
    except Exception as exc:
        logger.warning("Redisearch fetch failed: %s", exc)
        return []


@_retry
def _fetch_from_hashes(redis_conn, limit: int) -> List[Dict[str, Any]]:
    cursor = 0
    results: List[Dict[str, Any]] = []
    temp: List[Dict[str, Any]] = []
    while True:
        cursor, keys = redis_conn.scan(cursor=cursor, match=f"{HASH_PREFIX}*", count=100)
        for key in keys:
            if isinstance(key, bytes):
                key = key.decode()
            parsed = _parse_hash(redis_conn, key)
            if parsed:
                temp.append(parsed)
        if cursor == 0:
            break
    # sort by created_at desc, fallback to 0
    temp.sort(key=lambda x: _to_float(x.get("created_at"), 0), reverse=True)
    for p in temp[:limit]:
        results.append(p)
    return results


@_retry
def _fetch_from_redisearch_json(redis_conn, limit: int) -> List[Dict[str, Any]]:
    """Fetch prompt docs using the JSON RediSearch index `chat_prompts_json`.

    Uses FT.SEARCH with NOCONTENT to get doc ids, then retrieves JSON via RedisJSON or GET fallback.
    """
    try:
        # ensure index exists
        try:
            redis_conn.execute_command("FT.INFO", INDEX_JSON_NAME)
        except Exception:
            return []

        # Get doc ids only
        args = ["FT.SEARCH", INDEX_JSON_NAME, "*", "NOCONTENT", "SORTBY", "created_at", "DESC", "LIMIT", "0", str(limit)]
        raw = redis_conn.execute_command(*args)
        if not raw or len(raw) < 2:
            return []
        # raw format with NOCONTENT: [total, docid1, docid2, ...]
        total = raw[0]
        docids = raw[1:]
        results: List[Dict[str, Any]] = []
        for did in docids:
            if isinstance(did, bytes):
                did = did.decode()
            # fetch JSON document
            doc_obj = None
            try:
                json_client = getattr(redis_conn, "json", None)
                if json_client:
                    doc_obj = json_client().get(did)
                else:
                    rawj = redis_conn.get(did)
                    if rawj:
                        if isinstance(rawj, bytes):
                            rawj = rawj.decode()
                        doc_obj = json.loads(rawj)
            except Exception as exc:
                logger.warning("Failed to retrieve JSON for %s: %s", did, exc)
                doc_obj = None

            if not doc_obj:
                continue

            try:
                created_at = float(doc_obj.get("created_at")) if doc_obj.get("created_at") else None
            except Exception:
                created_at = None

            parsed = {"id": str(doc_obj.get("id") or did), "prompt": doc_obj.get("prompt"), "created_at": created_at}
            if doc_obj.get("source"):
                parsed["source"] = doc_obj.get("source")
            if doc_obj.get("lang"):
                parsed["lang"] = doc_obj.get("lang")
            results.append(parsed)
        return results[:limit]
    except Exception as exc:
        logger.warning("Redisearch JSON fetch failed: %s", exc)
        return []


def _fallback_to_prompts_module(limit: int) -> List[Dict[str, Any]]:
    """Import `app.utils.prompts` and try to extract a list of prompts.

    Supports module-level variables: `DEFAULT_PROMPTS`, `SAMPLE_PROMPTS`, `PROMPTS`.
    Each should be a Sequence of dict-like objects with required keys.
    """
    try:
        from app.utils import prompts as prompts_module
    except Exception as exc:
        logger.error("Failed to import app.utils.prompts: %s", exc)
        return []

    candidates = ["DEFAULT_PROMPTS", "SAMPLE_PROMPTS", "PROMPTS", "get_default_prompts"]
    for name in candidates:
        obj = getattr(prompts_module, name, None)
        if obj:
            try:
                if callable(obj):
                    data = obj()
                else:
                    data = obj
                if isinstance(data, Sequence):
                    out: List[Dict[str, Any]] = []
                    for item in list(data)[:limit]:
                        if not isinstance(item, dict):
                            continue
                        if "prompt" not in item:
                            continue
                        parsed = {"id": str(item.get("id") or item.get("name") or ""), "prompt": item.get("prompt"), "created_at": item.get("created_at")}
                        out.append(parsed)
                    if out:
                        logger.info("Loaded %s prompts from prompts.py fallback (%s)", len(out), name)
                        return out
            except Exception as exc:
                logger.warning("Failed to read prompts from %s: %s", name, exc)
                continue
    logger.warning("No usable prompts found in app.utils.prompts")
    # As a last resort, try to extract named section variables or build prompts
    # by invoking final_response_prompt (if available) and splitting into sections.
    section_names = ["core", "behavior", "funnel_logic", "output_schema", "context_block", "reminders"]
    sections_found = []
    for name in section_names:
        val = getattr(prompts_module, name, None)
        if val and isinstance(val, str) and val.strip():
            sections_found.append({"id": name, "prompt": val, "created_at": None})

    if sections_found:
        logger.info("Loaded %s section prompts from prompts.py variables", len(sections_found))
        return sections_found[:limit]

    # Try calling final_response_prompt to produce the combined prompt and split
    final_fn = getattr(prompts_module, "final_response_prompt", None)
    if callable(final_fn):
        try:
            combined = final_fn("", "", "", 1)
            if combined and isinstance(combined, str):
                # Split by top-level headings '## ' to create section prompts
                parts = [p.strip() for p in combined.split('\n## ') if p.strip()]
                out = []
                # try to map parts to known section names
                section_names = ["core", "behavior", "funnel_logic", "output_schema", "context_block", "reminders"]
                for i, part in enumerate(parts[:limit]):
                    # Restore heading prefix for non-first parts
                    if not part.startswith("## "):
                        part = ("## " + part) if i > 0 else part
                    sid = section_names[i] if i < len(section_names) else f"section_{i+1}"
                    out.append({"id": sid, "prompt": part, "created_at": None})
                if out:
                    logger.info("Loaded %s prompts by splitting final_response_prompt output", len(out))
                    return out
        except Exception as exc:
            logger.warning("Calling final_response_prompt failed: %s", exc)

    return []


def load_prompts(redis_conn, limit: int = 10, retries: int = 3, backoff: float = 0.5, json_index: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Main entrypoint.

    Returns a list of prompt dicts (may be empty). Raises on fatal Redis errors.
    """
    # basic validation
    _validate_redis_conn(redis_conn)

    # ensure index exists unless it already does
    try:
        # If json_index is not specified, detect whether a JSON RediSearch index exists
        if json_index is None:
            try:
                redis_conn.execute_command("FT.INFO", INDEX_JSON_NAME)
                json_index = True
            except Exception:
                json_index = False

        if json_index:
            # Prefer querying the JSON index directly
            items = _fetch_from_redisearch_json(redis_conn, limit)
            if items:
                return items[:limit]

        ensure_index(redis_conn)
    except Exception:
        logger.exception("Index ensure failed - continuing to fetch with best-effort")

    # Try strategies in order
    strategies = [
        ("sorted_set", _fetch_from_sorted_set),
        ("redisearch", _fetch_from_redisearch),
        ("hashes", _fetch_from_hashes),
    ]

    for name, func in strategies:
        try:
            items = func(redis_conn, limit=limit)
            if items:
                logger.info("Fetched %s prompts using strategy: %s", len(items), name)
                # validate items
                valid = []
                for it in items:
                    if not it.get("prompt"):
                        logger.warning("Dropping prompt without text: %s", it)
                        continue
                    # ensure id str
                    it["id"] = str(it.get("id") or "")
                    valid.append(it)
                if valid:
                    return valid[:limit]
        except Exception as exc:
            logger.warning("Strategy %s failed: %s", name, exc)

    # fallback to prompts module
    fallback = _fallback_to_prompts_module(limit)
    if fallback:
        return fallback

    logger.info("No prompts found in Redis or local prompts module; returning empty list")
    return []


def refresh_prompts(redis_conn, prompts: Optional[Sequence[Dict[str, Any]]] = None, limit: int = 100, retries: int = 3, backoff: float = 0.5, ensure_idx: bool = True, as_json: bool = False) -> int:
    """Write prompts into Redis using hashes and a sorted set.

    - `prompts` can be a sequence of dicts with keys: id, prompt, created_at (numeric), source, lang
    - If `prompts` is None, attempts to load from `app.utils.prompts` module
    - Returns number of prompts written
    """
    _validate_redis_conn(redis_conn)

    if ensure_idx:
        try:
            ensure_index(redis_conn, json_index=as_json)
        except Exception:
            logger.exception("Failed to ensure index while refreshing prompts; continuing")

    if prompts is None:
        prompts = _fallback_to_prompts_module(limit)

    if not prompts:
        logger.warning("No prompts available to refresh into Redis")
        return 0

    written = 0
    now = time.time()
    for p in list(prompts)[:limit]:
        try:
            pid = str(p.get("id") or p.get("name") or "")
            prompt_text = p.get("prompt") or p.get("prompt_text") or p.get("text")
            if not prompt_text:
                logger.warning("Skipping prompt without text: %s", p)
                continue
            created_at = p.get("created_at")
            try:
                score = float(created_at) if created_at is not None else now
            except Exception:
                score = now
            if as_json:
                key = f"chat_prompt_json:{pid}" if pid else f"chat_prompt_json:{int(score)}_{written}"
                # Build JSON document
                doc: Dict[str, Any] = {
                    "id": pid,
                    "prompt": prompt_text,
                    "created_at": score,
                }
                if p.get("source"):
                    doc["source"] = str(p.get("source"))
                if p.get("lang"):
                    doc["lang"] = str(p.get("lang"))

                # Enrich with assistant_name and assistant_instruction if available
                if pid:
                    ai_info = get_assistant_instruction_by_key(pid)
                    if ai_info:
                        doc["assistant_name"] = ai_info["assistant_name"]
                        doc["assistant_instruction"] = ai_info["assistant_instruction"]

                # Try to use RedisJSON if available, otherwise fall back to storing JSON string
                json_client = getattr(redis_conn, "json", None)
                if json_client:
                    try:
                        json_client().set(key, "$", doc)
                        logger.info("Stored JSON document using RedisJSON at key=%s", key)
                    except Exception as exc:
                        logger.warning("RedisJSON set failed, falling back to string SET: %s", exc)
                        try:
                            redis_conn.set(key, json.dumps(doc))
                            logger.info("Stored JSON string at key=%s", key)
                        except Exception as exc2:
                            logger.error("Failed to write JSON string to Redis: %s", exc2)
                            raise
                else:
                    # Fallback: write raw JSON string
                    try:
                        redis_conn.set(key, json.dumps(doc))
                        logger.info("Stored JSON string at key=%s (RedisJSON not available)", key)
                    except Exception as exc:
                        logger.error("Failed to write JSON string to Redis: %s", exc)
                        raise

                # Also add to sorted set for ordering
                try:
                    redis_conn.zadd(SORTED_SET_KEY, {key: score})
                except Exception as exc:
                    logger.warning("Failed to add JSON key to sorted set %s: %s", SORTED_SET_KEY, exc)
            else:
                key = f"{HASH_PREFIX}{pid}" if pid else f"{HASH_PREFIX}{int(score)}_{written}"
                # Prepare hash fields
                fields = {"id": pid, "prompt": prompt_text, "created_at": str(score)}
                if p.get("source"):
                    fields["source"] = str(p.get("source"))
                if p.get("lang"):
                    fields["lang"] = str(p.get("lang"))

                # HMSET
                redis_conn.hset(key, mapping=fields)
                # Add to sorted set for ordering
                redis_conn.zadd(SORTED_SET_KEY, {key: score})
            written += 1
        except Exception as exc:
            logger.warning("Failed to write prompt to Redis: %s - %s", p, exc)
            continue

    logger.info("Refreshed %s prompts into Redis (sorted set: %s)", written, SORTED_SET_KEY)
    return written
