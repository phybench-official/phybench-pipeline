"""
Centralized logging configuration for PhyBench Pipeline.

This module provides a configured loguru logger instance that outputs to both
console and file with appropriate formatting and rotation.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Remove default handler to prevent duplication
logger.remove()


def setup_logging(
    log_file: str | Path | None = None,
    log_level: str = "DEBUG",
    console_level: str = "INFO",
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        log_file: Path to the log file. If None, logs go to 'logs/phybench.log'
        log_level: Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        console_level: Console-specific logging level (if different from log_level)
    """
    if log_file is None:
        log_file = Path("logs") / "phybench.log"
    else:
        log_file = Path(log_file)

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    # Console logging with color and higher level filtering
    console_log_level = console_level or log_level
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=console_log_level,
        colorize=True,
        filter=lambda record: record["level"].no < 40,  # < ERROR level
    )

    # Separate handler for ERROR and CRITICAL to stderr
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level="ERROR",
        colorize=True,
    )

    # File logging with rotation, retention, and compression
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {process}:{thread} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="50 MB",
            retention="7 days",
            compression="gz",
            encoding="utf-8",
            enqueue=True,  # Thread-safe logging
        )

    logger.info(f"Logging initialized - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {Path(log_file).absolute()}")


def get_logger(name: str | None = None) -> Any:
    """
    Get a logger instance with optional name binding.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Module-level logger instance
log = logger
