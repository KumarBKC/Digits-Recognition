"""Logging setup for the digit recognition project."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with file and stream handlers.

    The first call creates the root ``digit_recognition`` logger; subsequent
    calls reuse the same handlers.
    """
    root_logger_name = "digit_recognition"
    root_logger = logging.getLogger(root_logger_name)

    if not root_logger.handlers:
        root_logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(fmt)
        root_logger.addHandler(stream_handler)

        # File handler (writes to training.log in the current directory)
        log_path = os.path.join(os.getcwd(), "training.log")
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(fmt)
            root_logger.addHandler(file_handler)
        except OSError:
            # If we cannot create the log file, continue without it
            pass

    if name.startswith(root_logger_name):
        return logging.getLogger(name)
    return logging.getLogger(f"{root_logger_name}.{name}")
