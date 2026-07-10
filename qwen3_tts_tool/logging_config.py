"""Centralized logging configuration with multi-level verbosity support.

Provides setup_logging() for configuring logging based on the verbosity
count from CLI arguments (-v, -vv, -vvv).
"""

import logging
import sys


def setup_logging(verbose_count: int = 0) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbose_count: Number of -v flags (0-3+)
            0: WARNING level (quiet mode)
            1: INFO level (normal verbose)
            2: DEBUG level (detailed debugging)
            3+: DEBUG + enable dependent library logging (trace mode)
    """
    if verbose_count == 0:
        level = logging.WARNING
    elif verbose_count == 1:
        level = logging.INFO
    elif verbose_count >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    if verbose_count >= 2:
        fmt = "[%(levelname)s] %(name)s:%(lineno)d - %(message)s"
    else:
        fmt = "[%(levelname)s] %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        stream=sys.stderr,
        force=True,
    )

    if verbose_count >= 3:
        logging.getLogger("transformers").setLevel(logging.DEBUG)
        logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
    else:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
