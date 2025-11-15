"""Utility helpers for metrics aggregation."""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List


def sum_usage(usages: Iterable[Dict[str, int]]) -> Dict[str, int]:
    """Sum token usage dictionaries."""

    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in total:
            total[key] += int(usage.get(key, 0))
    return total


def average_length(texts: List[str]) -> float:
    """Return the average length of the provided texts."""

    if not texts:
        return 0.0
    return statistics.mean(len(text) for text in texts)


__all__ = ["sum_usage", "average_length"]
