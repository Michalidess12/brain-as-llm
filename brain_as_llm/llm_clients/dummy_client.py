"""Dummy LLM client useful for local development and unit tests."""
from __future__ import annotations

import itertools
import json
import time
from typing import Any, Dict, Iterable, List, Optional

from .base import LLMClient


class DummyLLMClient(LLMClient):
    """Rule-based LLM client returning deterministic responses."""

    def __init__(self, scripted_responses: Optional[Iterable[str]] = None) -> None:
        self._responses = list(scripted_responses or [])
        self._counter = itertools.count(1)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = self._default_response(messages, model=model)

        prompt_chars = sum(len(msg.get("content", "")) for msg in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(text) // 4 + 1)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        simulated_delay = min(0.002 * prompt_tokens, 0.05)
        time.sleep(simulated_delay)

        return {"text": text, "usage": usage, "raw": {"model": model, "messages": messages}}

    def _default_response(self, messages: List[Dict[str, str]], *, model: Optional[str]) -> str:
        last_message = messages[-1]["content"].lower()
        if "control json" in last_message or "strategy" in last_message or "budget" in last_message:
            plan = {
                "difficulty": "medium",
                "max_reasoning_passes": 2,
                "needs_full_context": False,
                "strategy": "cascade_small_then_big",
                "target_expert_tokens": 256,
                "target_latency_ms": 400,
                "speculation_mode": "off",
                "notes_for_reasoner": "Answer succinctly and cite facts.",
            }
            return json.dumps(plan)
        if "summary" in last_message or "chunk" in last_message:
            summary = {
                "summary": "Chunk summary",
                "facts": ["Fact A"],
                "entities": [{"name": "Example", "type": "concept", "description": "placeholder"}],
                "quotes": ['"Quoted detail"'],
                "notes_for_reasoner": "",
            }
            return json.dumps(summary)
        if "critique" in last_message:
            return "Refined: " + last_message[:100]

        response_prefix = "SMALL" if self._is_small_model(model) else "EXPERT"
        return f"{response_prefix} answer {next(self._counter)}: {messages[-1]['content'][:120]}"

    def _is_small_model(self, model: Optional[str]) -> bool:
        if not model:
            return True
        lowered = model.lower()
        return any(token in lowered for token in ["mini", "small", "lite"])


__all__ = ["DummyLLMClient"]


__all__ = ["DummyLLMClient"]
