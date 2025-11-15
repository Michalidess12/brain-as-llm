"""Baseline pipeline using a single LLM call."""
from __future__ import annotations

import textwrap
import time
from typing import Any, Dict, Optional

from ..llm_clients.base import LLMClient


def run_baseline_pipeline(
    llm_client: LLMClient,
    *,
    raw_text: str,
    question: str,
    model_name: str,
    temperature: float = 0.2,
    max_context_chars: int = 12000,
    max_tokens: Optional[int] = None,
    policy_name: str = "baseline_full_context",
) -> Dict[str, Any]:
    """Execute the baseline: one prompt with raw context."""

    truncated_context = raw_text[:max_context_chars]
    prompt = textwrap.dedent(
        f"""
        You are an expert analyst. Answer the QUESTION using the CONTEXT below.
        QUESTION: {question.strip()}
        CONTEXT:
        {truncated_context.strip()}
        """
    ).strip()

    start = time.perf_counter()
    response = llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    return {
        "answer": response.get("text", ""),
        "usage": response.get("usage", {}),
        "latency_seconds": latency,
        "prompt_chars": len(prompt),
        "policy_name": policy_name,
    }


__all__ = ["run_baseline_pipeline"]
