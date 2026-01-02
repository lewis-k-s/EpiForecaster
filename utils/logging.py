"""
Logging utility for consistent logging configuration across the EpiForecaster project.

Provides a centralized setup_logging() function that should be called from the CLI
or any entry point to ensure consistent logging behavior across all modules.
"""

import logging
import sys
from typing import Literal


def setup_logging(
    level: int | Literal["DEBUG", "INFO", "WARNING", "ERROR"] = logging.INFO,
    format_str: str | None = None,
) -> None:
    """
    Configure logging for the EpiForecaster project.

    This function sets up root logging with a consistent format. It should be
    called once at application startup (e.g., from the CLI entry point).

    Args:
        level: Logging level. Can be an integer logging level or string name.
               Defaults to logging.INFO.
        format_str: Custom format string. If None, uses the default format:
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    Example:
        >>> from utils.logging import setup_logging
        >>> setup_logging(level="DEBUG")
        >>> logger = logging.getLogger(__name__)
        >>> logger.debug("This is a debug message")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if format_str is None:
        format_str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        stream=sys.stdout,
        force=False,
    )
