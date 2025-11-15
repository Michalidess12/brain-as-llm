"""Logging helpers."""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Configure root logger with a simple format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=force,
    )


__all__ = ["setup_logging"]
