"""Base interfaces for LLM clients."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    """Abstract interface for chat-completion style LLM APIs."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a chat request and return a structured response."""


__all__ = ["LLMClient"]
