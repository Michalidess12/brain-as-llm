"\"\"\"Brain-as-LLM pipeline orchestration.\"\"\""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..controller import BudgetContract, ControlPlan, CoreController, ControllerConfig
from ..encoder import Canvas, EncoderConfig, TextEncoder, TextEncoderResult
from ..llm_clients.base import LLMClient
from ..reasoner import CoreReasoner, ReasonerConfig, ReasonerResult
from ..state.canvas_store import CanvasStore
from ..utils.metrics import sum_usage


def run_brain_pipeline(
    *,
    small_client: LLMClient,
    large_client: LLMClient,
    raw_text: str,
    question: str,
    encoder_model: str,
    controller_model: str,
    reasoner_model: str,
    encoder_config: Optional[EncoderConfig] = None,
    controller_config: Optional[ControllerConfig] = None,
    reasoner_config: Optional[ReasonerConfig] = None,
    canvas_store: Optional[CanvasStore] = None,
    doc_id: Optional[str] = None,
    policy_name: str = "default_brain_v1",
    budget: Optional[BudgetContract] = None,
) -> Dict[str, Any]:
    """Run the encoder + controller + reasoner pipeline."""

    start = time.perf_counter()
    encoder = TextEncoder(small_client, encoder_model, encoder_config, canvas_store=canvas_store)
    controller = CoreController(small_client, controller_model, config=controller_config)
    reasoner = CoreReasoner(
        small_client=small_client,
        large_client=large_client,
        small_model=encoder_model,
        large_model=reasoner_model,
        config=reasoner_config,
    )

    encoder_result: TextEncoderResult = encoder.encode(raw_text=raw_text, question=question, doc_id=doc_id)
    plan: ControlPlan = controller.plan(question=question, canvas=encoder_result.canvas.to_dict(), budget=budget)
    reasoner_result: ReasonerResult = reasoner.reason(
        question=question,
        canvas=encoder_result.canvas.to_dict(),
        plan=plan,
        raw_text=raw_text,
    )

    encoder_usage = sum_usage(chunk["usage"] for chunk in encoder_result.chunk_summaries) if encoder_result.chunk_summaries else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    controller_usage = plan.usage
    expert_usage = sum_usage(reasoner_result.usage)
    reasoner_latency = sum(reasoner_result.step_latencies)
    total_latency = time.perf_counter() - start

    return {
        "answer": reasoner_result.final_answer,
        "plan": plan.to_dict(),
        "policy_name": policy_name,
        "strategy_used": reasoner_result.strategy_used,
        "speculation_mode": reasoner_result.speculation_mode,
        "debug": {
            "canvas": encoder_result.canvas.to_dict(),
            "intermediate_steps": reasoner_result.intermediate_steps,
            "chunk_summaries": encoder_result.chunk_summaries,
            "stages": [stage.to_dict() for stage in reasoner_result.stages],
            "encoder_from_cache": encoder_result.from_cache,
        },
        "usage": {
            "encoder_tokens": encoder_usage,
            "controller_tokens": controller_usage,
            "reasoner_tokens": expert_usage,
        },
        "latency_seconds": total_latency,
        "reasoner_latency_seconds": reasoner_latency,
    }


__all__ = ["run_brain_pipeline"]
