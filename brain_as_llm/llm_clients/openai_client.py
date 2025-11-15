"""OpenAI-compatible LLM client implementation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .base import LLMClient


class OpenAIClient(LLMClient):
    """Thin wrapper around the official OpenAI Python SDK."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        if OpenAI is None:  # pragma: no cover - exercised only without dependency
            raise RuntimeError(
                "openai package is required for OpenAIClient"  # noqa: TRY003
            ) from _IMPORT_ERROR
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        }

        return {
            "text": text,
            "usage": usage,
            "raw": response.model_dump() if hasattr(response, "model_dump") else response,
        }


__all__ = ["OpenAIClient"]
