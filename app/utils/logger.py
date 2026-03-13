"""Rotating file + console logger."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Create and return a logger with rotating file + console handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # Rotating file handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_path / "rag_chatbot.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
