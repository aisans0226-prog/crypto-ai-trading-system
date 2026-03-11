"""
utils/logger.py — Structured logging using Loguru
"""
import sys
import os
from loguru import logger
from config import settings


def setup_logger() -> None:
    """Configure loguru with console + rotating file sinks."""
    logger.remove()                              # remove default stderr sink

    # Console sink — coloured, human-readable
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File sink — JSON structured, rotated daily, retained 30 days
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation="00:00",          # rotate at midnight
        retention="30 days",
        compression="gz",
        serialize=True,            # JSON format
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logger initialised — level={}", settings.log_level)


# Run setup on import
setup_logger()
