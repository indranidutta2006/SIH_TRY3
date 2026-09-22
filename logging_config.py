"""Centralized logging infrastructure configuration.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Configures structured console logging and disk-backed rotating file handlers
at the INFO level by default.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import get_default_config

DEFAULT_LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_file_name: str = "system.log",
    level: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Initialize and configure global rotating file and stream loggers.

    Args:
        log_file_name: Name of the rotating log file stored under outputs/logs.
        level: Logging verbosity level. Defaults to config logging_level ('INFO').
        max_bytes: Maximum size in bytes before rotating log file (default 10MB).
        backup_count: Number of rotated history files to retain.

    Returns:
        Configured root application Logger instance.
    """
    config = get_default_config()
    resolved_level_name = level if level is not None else config.logging_level
    resolved_level = getattr(logging, resolved_level_name.upper(), logging.INFO)

    log_dir: Path = config.output_paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path: Path = log_dir / log_file_name

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    # Avoid duplicate handlers if re-invoked
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("maritime_system")
    logger.info("Logging initialized at level '%s'. Output file: %s", resolved_level_name, log_file_path)

    return logger
