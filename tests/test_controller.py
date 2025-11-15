import json

from brain_as_llm.controller import CoreController, BudgetContract
from brain_as_llm.llm_clients.dummy_client import DummyLLMClient
from brain_as_llm.state import InMemoryStateStore


def _canvas() -> dict:
    return {
        "summaries": ["Budget cuts"],
        "facts": ["Fact"],
        "entities": [],
        "quotes": [],
        "notes_for_reasoner": "",
    }


def test_controller_returns_rich_plan():
    plan_json = json.dumps(
        {
            "difficulty": "easy",
            "max_reasoning_passes": 1,
            "needs_full_context": False,
            "strategy": "small_only",
            "target_expert_tokens": 128,
            "target_latency_ms": 200,
            "speculation_mode": "off",
            "notes_for_reasoner": "Answer succinctly.",
        }
    )
    controller = CoreController(
        DummyLLMClient([plan_json]),
        model_name="dummy-small",
        state_store=InMemoryStateStore(),
    )

    plan = controller.plan(
        question="Summarize",
        canvas=_canvas(),
        budget=BudgetContract(max_latency_ms=500, max_expert_tokens=256, priority="high"),
    )

    assert plan.strategy == "small_only"
    assert plan.max_reasoning_passes == 1
    assert plan.target_expert_tokens == 128
