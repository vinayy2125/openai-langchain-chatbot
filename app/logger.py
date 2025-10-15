import logging
import sys
from logging.handlers import RotatingFileHandler
import os

PROJECT_ROOT = os.getcwd()
LOG_FILE = os.path.join(PROJECT_ROOT, 'app.log')

def get_logger(name: str = "chatbot") -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s - %(message)s"
    )
    while logger.handlers:
        logger.handlers.pop()
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
