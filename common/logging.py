"""Shared logger: rich in a terminal, plain in a log file.

RichHandler wraps every message to the console width and pads the remainder
with blank lines. Interactively that is fine. Under SLURM there is no terminal,
rich assumes 80 columns, and each measurement gets folded onto continuation
lines — so a grep for the metric name returns the label and leaves the number
on the next line. Batch logs are read with grep, so they get plain formatting.
"""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler

from common.config import settings

_CONFIGURED = False


def _handler() -> logging.Handler:
    if sys.stderr.isatty():
        return RichHandler(rich_tracebacks=True, markup=True)
    plain = logging.StreamHandler(sys.stdout)
    plain.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", "%H:%M:%S"))
    return plain


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=settings.log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[_handler()],
        )
        # httpx logs a line per request at INFO. The embedding and labelling
        # jobs make thousands, which buries the run's own output.
        for noisy in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
