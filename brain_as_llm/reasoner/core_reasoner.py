"\"\"\"Core multi-step reasoner implementation with adaptive strategies.\"\"\""
from __future__ import annotations

import logging
import time
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..controller.core_controller import ControlPlan
from ..llm_clients.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ReasonerConfig:
    system_prompt: str = (
        "You are an expert analyst. Given a compressed canvas and a question, reason carefully and "
        "provide clear, well-supported answers. Cite relevant key points when possible."
    )
    temperature: float = 0.2
    max_tokens: Optional[int] = 600


@dataclass
class StageReasoningResult:
    stage_name: str
    model: str
    final_answer: str
    intermediate_steps: List[str]
    usage: List[Dict[str, Any]]
    step_latencies: List[float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "model": self.model,
            "final_answer": self.final_answer,
            "intermediate_steps": self.intermediate_steps,
            "usage": self.usage,
            "step_latencies": self.step_latencies,
            "confidence": self.confidence,
        }


@dataclass
class ReasonerResult:
    final_answer: str
    intermediate_steps: List[str] = field(default_factory=list)
    usage: List[Dict[str, Any]] = field(default_factory=list)
    step_latencies: List[float] = field(default_factory=list)
    stages: List[StageReasoningResult] = field(default_factory=list)
    strategy_used: str = "full_brain"
    speculation_mode: str = "off"


class CoreReasoner:
    """Expert model that operates on the compressed canvas."""

    def __init__(
        self,
        *,
        small_client: LLMClient,
        large_client: LLMClient,
        small_model: str,
        large_model: str,
        config: Optional[ReasonerConfig] = None,
    ) -> None:
        self._small_client = small_client
        self._large_client = large_client
        self._small_model = small_model
        self._large_model = large_model
        self._config = config or ReasonerConfig()

    def reason(
        self,
        question: str,
        canvas: Dict[str, Any],
        plan: ControlPlan,
        *,
        raw_text: Optional[str] = None,
    ) -> ReasonerResult:
        """Run one or more reasoning passes and return the final answer."""

        stages: List[StageReasoningResult] = []
        strategy = plan.strategy or "full_brain"
        logger.debug("Reasoner strategy: %s", strategy)

        if strategy == "small_only":
            stage = self._run_stage(
                stage_name="small_only",
                client=self._small_client,
                model_name=self._small_model,
                question=question,
                canvas=canvas,
                plan=plan,
                raw_text=raw_text,
            )
            stages.append(stage)
        elif strategy == "cascade_small_then_big":
            small_stage = self._run_stage(
                stage_name="cascade_small",
                client=self._small_client,
                model_name=self._small_model,
                question=question,
                canvas=canvas,
                plan=plan,
                raw_text=raw_text,
                max_passes=min(2, plan.max_reasoning_passes),
            )
            stages.append(small_stage)
            if self._needs_escalation(small_stage):
                logger.debug("Escalating to expert model after small-stage confidence %.2f", small_stage.confidence)
                expert_stage = self._run_stage(
                    stage_name="cascade_expert",
                    client=self._large_client,
                    model_name=self._large_model,
                    question=question,
                    canvas=canvas,
                    plan=plan,
                    raw_text=raw_text,
                )
                stages.append(expert_stage)
        else:  # full_brain
            stage = self._run_stage(
                stage_name="full_brain",
                client=self._large_client,
                model_name=self._large_model,
                question=question,
                canvas=canvas,
                plan=plan,
                raw_text=raw_text,
            )
            stages.append(stage)

        flattened_usage: List[Dict[str, Any]] = []
        flattened_steps: List[str] = []
        flattened_latencies: List[float] = []
        for stage in stages:
            flattened_usage.extend(stage.usage)
            flattened_steps.extend(stage.intermediate_steps)
            flattened_latencies.extend(stage.step_latencies)

        final_answer = stages[-1].final_answer if stages else ""
        return ReasonerResult(
            final_answer=final_answer,
            intermediate_steps=flattened_steps,
            usage=flattened_usage,
            step_latencies=flattened_latencies,
            stages=stages,
            strategy_used=strategy,
            speculation_mode=plan.speculation_mode,
        )

    def _run_stage(
        self,
        *,
        stage_name: str,
        client: LLMClient,
        model_name: str,
        question: str,
        canvas: Dict[str, Any],
        plan: ControlPlan,
        raw_text: Optional[str],
        max_passes: Optional[int] = None,
    ) -> StageReasoningResult:
        total_passes = max_passes or plan.max_reasoning_passes
        formatted_canvas = self._format_canvas(canvas)
        if plan.needs_full_context and raw_text:
            formatted_canvas += "\n\n[Additional Context]\n" + textwrap.shorten(raw_text, width=1200, placeholder=" ...")
        notes = plan.notes_for_reasoner or ""
        intermediate: List[str] = []
        usage: List[Dict[str, Any]] = []
        latencies: List[float] = []
        previous_output = ""

        for step_index in range(1, total_passes + 1):
            prompt = self._build_step_prompt(
                step_index=step_index,
                total_steps=total_passes,
                question=question,
                canvas_text=formatted_canvas,
                prior=previous_output,
                notes=notes,
                plan=plan,
            )
            logger.debug("Reasoner executing %s step %s/%s", stage_name, step_index, total_passes)
            step_start = time.perf_counter()
            response = client.chat(
                messages=[
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            latencies.append(time.perf_counter() - step_start)
            text = response.get("text", "")
            intermediate.append(text)
            usage.append(response.get("usage", {}))
            previous_output = text

        confidence = self._estimate_confidence(intermediate[-1] if intermediate else "")
        return StageReasoningResult(
            stage_name=stage_name,
            model=model_name,
            final_answer=intermediate[-1] if intermediate else "",
            intermediate_steps=intermediate,
            usage=usage,
            step_latencies=latencies,
            confidence=confidence,
        )

    def _format_canvas(self, canvas: Dict[str, Any]) -> str:
        def _safe_join(items: List[Any]) -> str:
            if not items:
                return "(none)"
            return "\n - ".join(str(item) for item in items)

        summaries = _safe_join(canvas.get("summaries", []))
        entities_text = _safe_join(canvas.get("entities", []))
        facts_text = _safe_join(canvas.get("facts", []))
        quotes = _safe_join(canvas.get("quotes", []))
        notes = canvas.get("notes_for_reasoner", "") or ""
        return textwrap.dedent(
            f"""
            [Canvas]
            Summaries:\n - {summaries}
            Entities:\n - {entities_text}
            Facts:\n - {facts_text}
            Quotes:\n - {quotes}
            Notes:\n{notes}
            """
        ).strip()

    def _build_step_prompt(
        self,
        *,
        step_index: int,
        total_steps: int,
        question: str,
        canvas_text: str,
        prior: str,
        notes: str,
        plan: ControlPlan,
    ) -> str:
        include_canvas = step_index == 1 or total_steps == 1
        if total_steps == 1:
            instructions = "Provide a direct, well-structured answer."
        elif step_index == 1:
            instructions = "Draft reasoning with numbered arguments and highlight gaps."
        elif step_index == total_steps:
            instructions = "Produce the final polished answer that resolves earlier critiques."
        else:
            instructions = "Critique the previous draft and outline concrete improvements."

        budget_line = ""
        if plan.target_expert_tokens:
            budget_line += f"Target expert tokens: {plan.target_expert_tokens}. "
        if plan.target_latency_ms:
            budget_line += f"Target latency: {plan.target_latency_ms} ms. "
        prompt_parts = [
            f"QUESTION: {question.strip()}",
        ]
        if include_canvas:
            prompt_parts.append(canvas_text)
        prompt_parts.append(f"Controller notes: {notes}")
        prompt_parts.append(f"Strategy: {plan.strategy} | Speculation: {plan.speculation_mode}")
        if budget_line.strip():
            prompt_parts.append(budget_line.strip())
        prompt_parts.append(f"Step {step_index}/{total_steps}. {instructions}")
        if prior:
            prompt_parts.append(f"Prior output:\n{prior.strip()}")
        return "\n".join(part for part in prompt_parts if part)

    def _estimate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
        lowered = text.lower()
        confidence = min(1.0, max(0.1, len(text) / 500))
        if any(phrase in lowered for phrase in ["not sure", "uncertain", "unknown"]):
            confidence *= 0.5
        return confidence

    def _needs_escalation(self, stage: StageReasoningResult) -> bool:
        return stage.confidence < 0.6 or len(stage.final_answer.strip()) < 120


__all__ = ["CoreReasoner", "ReasonerConfig", "ReasonerResult", "StageReasoningResult"]
