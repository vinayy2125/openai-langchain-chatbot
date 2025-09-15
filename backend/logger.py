import os
import logging
from logging.handlers import RotatingFileHandler

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure the logger
logger = logging.getLogger("backend")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File Handler
file_handler = RotatingFileHandler("logs/backend.log", maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_event(event: str, **kwargs):
    """Helper to log structured events."""
    log_message = f"{event} | " + ", ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info(log_message)
