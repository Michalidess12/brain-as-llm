"""Simple policy analytics for brain-as-llm."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_results(paths: Iterable[Path]) -> List[Dict]:
    """Load JSONL experiment files into a flat list of dicts."""

    records: List[Dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def compute_policy_stats(records: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute average expert tokens/latency per policy name."""

    aggregates: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for record in records:
        for key in ("baseline", "brain"):
            entry = record.get(key)
            if not entry:
                continue
            policy = entry.get("policy_name") or record.get("policy_name") or f"{key}_unknown"
            usage = entry.get("usage", {})
            expert_tokens = usage.get("total_tokens", usage.get("reasoner_tokens", {}).get("total_tokens", 0))
            latency = entry.get("latency_seconds", 0.0)
            aggregates.setdefault(policy, {"tokens": 0.0, "latency": 0.0})
            counts[policy] = counts.get(policy, 0) + 1
            aggregates[policy]["tokens"] += expert_tokens
            aggregates[policy]["latency"] += latency

    stats: Dict[str, Dict[str, float]] = {}
    for policy, aggregate in aggregates.items():
        total = counts.get(policy, 1)
        stats[policy] = {
            "avg_expert_tokens": aggregate["tokens"] / total,
            "avg_expert_latency": aggregate["latency"] / total,
            "samples": total,
        }
    return stats


def recommend_policies_by_testcase(records: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    """Pick the most efficient policy per testcase based on expert metrics."""

    recommendations: Dict[str, Dict[str, float]] = {}
    for record in records:
        testcase = record.get("id") or record.get("doc_id") or "unknown"
        candidates = []
        for key in ("baseline", "brain"):
            entry = record.get(key)
            if not entry:
                continue
            policy = entry.get("policy_name") or record.get("policy_name") or f"{key}_unknown"
            if key == "brain":
                reasoner_usage = entry.get("usage", {}).get("reasoner_tokens", {}).get("total_tokens", 0)
            else:
                reasoner_usage = entry.get("usage", {}).get("total_tokens", 0)
            candidates.append(
                {
                    "policy_name": policy,
                    "expert_tokens": reasoner_usage,
                    "expert_latency": entry.get("latency_seconds", 0.0),
                }
            )
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item["expert_tokens"], item["expert_latency"]))
        recommendations[testcase] = candidates[0]
    return recommendations


__all__ = ["load_results", "compute_policy_stats", "recommend_policies_by_testcase"]
