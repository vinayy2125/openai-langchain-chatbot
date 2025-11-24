"""Script to refresh prompts into Redis.

Usage examples:
  # Use built-in prompts from app.utils.prompts
  python scripts/refresh_prompts_to_redis.py

  # Load prompts from a JSON file (array of prompt objects)
  python scripts/refresh_prompts_to_redis.py --file prompts.json --limit 200

  # Override Redis connection
  python scripts/refresh_prompts_to_redis.py --host 127.0.0.1 --port 6379 --password mypass

The script writes prompts as hashes `chat_prompt:{id}` and adds them to sorted set `chat_prompts:z`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("refresh_prompts_to_redis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh prompts into Redis")
    p.add_argument("--file", "-f", help="JSON file containing an array of prompt objects (optional)")
    p.add_argument("--limit", "-n", type=int, default=100, help="Maximum number of prompts to write")
    p.add_argument("--ensure-index", dest="ensure_index", action="store_true", help="Ensure RediSearch index exists (default)")
    p.add_argument("--no-ensure-index", dest="ensure_index", action="store_false", help="Do not attempt to create RediSearch index")
    p.set_defaults(ensure_index=True)
    p.add_argument("--host", help="Redis host (overrides .env)")
    p.add_argument("--port", type=int, help="Redis port (overrides .env)")
    p.add_argument("--password", help="Redis password (overrides .env)")
    p.add_argument("--as-json", dest="as_json", action="store_true", help="Store prompts as JSON using RedisJSON (writes keys chat_prompt_json:{id})")
    p.set_defaults(as_json=True)
    return p.parse_args()


def load_prompts_from_file(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of prompt objects")
    return data[:limit]


def build_redis_client(host: Optional[str], port: Optional[int], password: Optional[str]):
    # Lazy import to avoid requiring redis if not used
    import redis

    if host or port or password:
        cfg = {}
        if host:
            cfg["host"] = host
        if port:
            cfg["port"] = port
        if password:
            cfg["password"] = password
        cfg["decode_responses"] = False
        return redis.Redis(**cfg)
    else:
        # Use app config's client (reads .env)
        try:
            from app.config import get_redis

            if get_redis:
                return get_redis
        except Exception:
            pass
        # Fallback to default local Redis
        return redis.Redis(decode_responses=False)


def main() -> int:
    args = parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        # Ensure project root is on sys.path so `app` package imports work when script run directly
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        redis_client = build_redis_client(args.host, args.port, args.password)
        # validate connection
        redis_client.ping()
    except Exception as exc:
        logger.error("Failed to connect to Redis: %s", exc)
        return 2

    prompts: Optional[List[Dict[str, Any]]] = None
    if args.file:
        try:
            prompts = load_prompts_from_file(args.file, args.limit)
            logger.info("Loaded %d prompts from file %s", len(prompts), args.file)
        except Exception as exc:
            logger.error("Failed to load prompts from file: %s", exc)
            return 3

    # Use helper to refresh prompts
    try:
        from app.db.redis_prompts import refresh_prompts

        written = refresh_prompts(redis_client, prompts=prompts, limit=args.limit, ensure_idx=args.ensure_index, as_json=args.as_json)
        logger.info("Wrote %d prompts into Redis", written)
        print(json.dumps({"status": "success", "written": written}))
        return 0
    except Exception as exc:
        logger.exception("Failed to refresh prompts: %s", exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
