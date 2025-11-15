"""Configuration helpers for the brain-as-llm prototype."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""

    openai_api_key: Optional[str]
    openai_base_url: str
    small_model: str
    large_model: str
    default_temperature: float = 0.2
    max_tokens: Optional[int] = None


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings from environment variables once per process."""

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai")

    return Settings(
        openai_api_key=openai_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        small_model=os.getenv("BRAIN_SMALL_MODEL", "gpt-4o-mini"),
        large_model=os.getenv("BRAIN_LARGE_MODEL", "gpt-4o"),
        default_temperature=float(os.getenv("BRAIN_DEFAULT_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("BRAIN_MAX_TOKENS")) if os.getenv("BRAIN_MAX_TOKENS") else None,
    )


__all__ = ["Settings", "load_settings"]
