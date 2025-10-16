import logging
import sys
from logging.handlers import RotatingFileHandler
import os

PROJECT_ROOT = os.getcwd()
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log")

# Shared (singleton) handlers so we don't create duplicates across modules
_console_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None


def _get_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s - %(message)s"
    )


class AsciiOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = record.msg.encode("ascii", errors="replace").decode("ascii")
        return True


def _get_console_handler() -> logging.Handler:
    global _console_handler
    if _console_handler is None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_get_formatter())
        ch.addFilter(AsciiOnlyFilter())
        _console_handler = ch
    return _console_handler


def _get_file_handler() -> logging.Handler:
    global _file_handler
    if _file_handler is None:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(_get_formatter())
        _file_handler = fh
    return _file_handler


def get_logger(name: str = "__main__") -> logging.Logger:
    """Return a configured logger instance using shared handlers.

    This avoids creating multiple handlers for the same logger and prevents
    duplicate log lines when modules call `get_logger` repeatedly.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Attach shared handlers only if the logger doesn't already have handlers
    if not logger.handlers:
        logger.addHandler(_get_console_handler())
        logger.addHandler(_get_file_handler())

    # Prevent propagation to root logger to avoid duplication with uvicorn/gunicorn
    logger.propagate = False
    return logger


def attach_handlers_to_uvicorn() -> None:
    """Attach our shared handlers to common ASGI/WSGI server loggers.

    This makes sure Uvicorn/Gunicorn log messages use the same handlers and
    file, and avoids duplicate console output.
    """
    # Names of known server loggers to attach to
    target_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error", "gunicorn.access"]
    for lname in target_loggers:
        l = logging.getLogger(lname)
        # Clear default handlers to avoid mixing configs, then attach shared ones
        if l.handlers:
            l.handlers.clear()
        l.setLevel(logging.INFO)
        l.addHandler(_get_console_handler())
        l.addHandler(_get_file_handler())
        l.propagate = False
 