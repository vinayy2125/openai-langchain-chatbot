import logging
import sys
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue
import os
import atexit

PROJECT_ROOT = os.getcwd()
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Shared handlers
_console_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None
_queue_handler: logging.Handler | None = None
_listener: QueueListener | None = None


def _get_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s - %(message)s"
    )


class AsciiOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = record.msg.encode("ascii", errors="replace").decode("ascii")
        return True


def _ensure_logging_system():
    """Initialize the background logging system if not already started."""
    global _console_handler, _file_handler, _queue_handler, _listener
    
    if _listener is not None:
        return

    # 1. Create actual handlers (workers)
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
        _console_handler.setFormatter(_get_formatter())
        _console_handler.addFilter(AsciiOnlyFilter())

    if _file_handler is None:
        _file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        _file_handler.setFormatter(_get_formatter())

    # 2. Create the queue and queue handler
    log_queue = Queue(-1)
    _queue_handler = QueueHandler(log_queue)

    # 3. Start the listener
    _listener = QueueListener(log_queue, _console_handler, _file_handler)
    _listener.start()
    
    # Register cleanup
    atexit.register(_listener.stop)


def get_logger(name: str = "__main__") -> logging.Logger:
    """Return a configured logger instance using non-blocking queue handler."""
    _ensure_logging_system()
    
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Attach queue handler only if the logger doesn't already have handlers
    if not logger.handlers:
        logger.addHandler(_queue_handler)

    # Prevent propagation to root logger to avoid duplication
    logger.propagate = False
    return logger


def attach_handlers_to_uvicorn() -> None:
    """Attach our shared queue handler to common ASGI/WSGI server loggers."""
    _ensure_logging_system()
    
    # Names of known server loggers to attach to
    target_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error", "gunicorn.access"]
    for lname in target_loggers:
        l = logging.getLogger(lname)
        # Clear default handlers to avoid mixing configs, then attach shared queue handler
        if l.handlers:
            l.handlers.clear()
        l.setLevel(LOG_LEVEL)
        l.addHandler(_queue_handler)
        l.propagate = False
 