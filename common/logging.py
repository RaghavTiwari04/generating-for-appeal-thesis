"""Rich-formatted logger shared across modules."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from common.config import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=settings.log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
