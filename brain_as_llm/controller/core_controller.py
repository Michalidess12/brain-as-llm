"""Cost-aware controller that decides reasoning strategies."""
from __future__ import annotations

import json
import logging
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..llm_clients.base import LLMClient
from ..state.state_store import BaseStateStore, InMemoryStateStore

logger = logging.getLogger(__name__)


@dataclass
class BudgetContract:
    """Optional latency/cost constraints provided to the controller."""

    max_latency_ms: Optional[int] = None
    max_expert_tokens: Optional[int] = None
    priority: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_latency_ms": self.max_latency_ms,
            "max_expert_tokens": self.max_expert_tokens,
            "priority": self.priority,
        }


@dataclass
class ControlPlan:
    """Decision returned by the controller."""

    difficulty: str
    max_reasoning_passes: int
    needs_full_context: bool
    strategy: str
    target_expert_tokens: Optional[int]
    target_latency_ms: Optional[int]
    speculation_mode: str
    notes_for_reasoner: str
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "max_reasoning_passes": self.max_reasoning_passes,
            "needs_full_context": self.needs_full_context,
            "strategy": self.strategy,
            "target_expert_tokens": self.target_expert_tokens,
            "target_latency_ms": self.target_latency_ms,
            "speculation_mode": self.speculation_mode,
            "notes_for_reasoner": self.notes_for_reasoner,
            "usage": self.usage,
            "latency_seconds": self.latency_seconds,
        }


@dataclass
class ControllerConfig:
    system_prompt: str = (
        "You are a metacognitive controller that routes questions to a reasoning engine. "
        "Given a multi-resolution canvas, a question, optional state, and a budget contract, "
        "decide the reasoning strategy. Respond with JSON containing: "
        "difficulty (easy/medium/hard), max_reasoning_passes (1-3), needs_full_context (bool), "
        "strategy (small_only | cascade_small_then_big | full_brain), "
        "target_expert_tokens (int or null), target_latency_ms (int or null), "
        "speculation_mode (off | conservative | aggressive), and notes_for_reasoner."
    )
    temperature: float = 0.0
    max_tokens: Optional[int] = 256


class CoreController:
    """Small controller deciding how the reasoner should operate."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str,
        *,
        state_store: Optional[BaseStateStore] = None,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self._llm = llm_client
        self._model_name = model_name
        self._state_store = state_store or InMemoryStateStore()
        self._config = config or ControllerConfig()

    def plan(
        self,
        question: str,
        canvas: Dict[str, Any],
        *,
        budget: Optional[BudgetContract] = None,
    ) -> ControlPlan:
        """Return a control plan for the provided canvas and question."""

        state_snapshot = self._state_store.load_state() or {}
        prompt = self._build_prompt(question=question, canvas=canvas, state=state_snapshot, budget=budget)
        start = time.perf_counter()
        response = self._llm.chat(
            messages=[
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self._model_name,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        latency = time.perf_counter() - start
        plan_dict = self._parse_response(response["text"])
        plan = ControlPlan(
            **plan_dict,
            usage=response.get("usage", {}),
            latency_seconds=latency,
        )
        self._state_store.save_state({"last_difficulty": plan.difficulty, "last_strategy": plan.strategy})
        return plan

    def _build_prompt(
        self,
        *,
        question: str,
        canvas: Dict[str, Any],
        state: Dict[str, Any],
        budget: Optional[BudgetContract],
    ) -> str:
        notes = textwrap.shorten(str(canvas.get("notes_for_reasoner", "")), width=400, placeholder="...")
        summaries = canvas.get("summaries") or canvas.get("key_points") or []
        prompt_budget = budget.to_dict() if budget else {"priority": "normal"}
        return textwrap.dedent(
            f"""
            STATE: {state}
            QUESTION: {question.strip()}
            CANVAS SUMMARIES: {summaries[:6]}
            ENTITIES: {canvas.get('entities', [])[:5]}
            FACTS: {canvas.get('facts', [])[:5]}
            NOTES: {notes}
            BUDGET: {prompt_budget}
            Provide the control JSON.
            """
        ).strip()

    def _parse_response(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Controller returned non-JSON response; applying heuristic parsing")
            difficulty = "hard" if "hard" in text.lower() else ("easy" if "easy" in text.lower() else "medium")
            strategy = "full_brain" if difficulty == "hard" else "cascade_small_then_big"
            data = {
                "difficulty": difficulty,
                "max_reasoning_passes": 3 if difficulty == "hard" else 1,
                "needs_full_context": "full" in text.lower(),
                "strategy": strategy,
                "target_expert_tokens": None,
                "target_latency_ms": None,
                "speculation_mode": "off",
                "notes_for_reasoner": text.strip(),
            }
        defaults = {
            "difficulty": "medium",
            "max_reasoning_passes": 2,
            "needs_full_context": False,
            "strategy": "full_brain",
            "target_expert_tokens": None,
            "target_latency_ms": None,
            "speculation_mode": "off",
            "notes_for_reasoner": "",
        }
        for key, value in defaults.items():
            data.setdefault(key, value)
        data["max_reasoning_passes"] = max(1, min(3, int(data["max_reasoning_passes"])))
        if data["strategy"] not in {"small_only", "cascade_small_then_big", "full_brain"}:
            data["strategy"] = "full_brain"
        if data["speculation_mode"] not in {"off", "conservative", "aggressive"}:
            data["speculation_mode"] = "off"
        return data


__all__ = ["CoreController", "ControlPlan", "ControllerConfig", "BudgetContract"]
