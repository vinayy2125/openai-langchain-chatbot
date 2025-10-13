from __future__ import annotations
from app.config import settings
import logging
logger = logging.getLogger("redis_service")

# -------------------------
# Configuration (override via settings)
# -------------------------
PREFIX = getattr(settings, "redis_prefix", "dits_chatbot:")
