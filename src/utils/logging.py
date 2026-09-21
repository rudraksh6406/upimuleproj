"""Structured logging utility for MuleGuard."""
import logging
import sys
from typing import Optional


def get_logger(name: str = "muleguard", level: Optional[int] = None) -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(level or logging.INFO)
    return logger
