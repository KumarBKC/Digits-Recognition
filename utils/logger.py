"""Logging setup for the digit recognition project."""

import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Generator, Optional

# Rotating log configuration
_MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB per log file
_BACKUP_COUNT = 3                  # keep 3 rotated backups
_LOG_LEVEL_NAMES = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _resolve_log_level() -> int:
    """Read console log level from DIGIT_LOG_LEVEL env var (default INFO)."""
    env_level = os.environ.get("DIGIT_LOG_LEVEL", "INFO").upper()
    return _LOG_LEVEL_NAMES.get(env_level, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with file and stream handlers.

    The first call creates the root ``digit_recognition`` logger; subsequent
    calls reuse the same handlers.  The file handler uses
    :class:`RotatingFileHandler` to cap log files at 5 MB with 3 backups.
    """
    root_logger_name = "digit_recognition"
    root_logger = logging.getLogger(root_logger_name)

    if not root_logger.handlers:
        root_logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_level = _resolve_log_level()

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(fmt)
        root_logger.addHandler(stream_handler)

        # Rotating file handler (writes to training.log, max 5 MB × 3 backups)
        log_dir = os.environ.get("DIGIT_LOG_DIR", os.getcwd())
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training.log")
        try:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=_MAX_LOG_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(fmt)
            root_logger.addHandler(file_handler)
        except OSError:
            # If we cannot create the log file, continue without it
            pass

    if name.startswith(root_logger_name):
        return logging.getLogger(name)
    return logging.getLogger(f"{root_logger_name}.{name}")


@contextmanager
def log_timer(
    description: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """Context manager that logs the elapsed wall-clock time of a block.

    Usage::

        with log_timer("Model inference", logger):
            result = predictor.predict(image)

    Args:
        description: Human-readable label for the timed operation.
        logger: Logger instance; if ``None``, uses the root project logger.
        level: Logging level for the timing message.

    Yields:
        None — the block executes normally.
    """
    if logger is None:
        logger = get_logger("timer")
    start = time.perf_counter()
    logger.log(level, "%s — started", description)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed >= 1.0:
            logger.log(level, "%s — completed in %.2fs", description, elapsed)
        else:
            logger.log(level, "%s — completed in %.1fms", description, elapsed * 1000)
