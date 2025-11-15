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


def test_costcut_policy_triggers_passthrough_without_llm():
    controller = CoreController(
        DummyLLMClient([]),
        model_name="dummy-small",
        state_store=InMemoryStateStore(),
    )

    plan = controller.plan(
        question="Summarize in bullet points",
        canvas=_canvas(),
        policy_name="openai_brain_costcut_v1",
        raw_text="short notes here",
    )

    assert plan.strategy == "baseline_passthrough"
    assert plan.max_reasoning_passes == 1


def test_costcut_policy_overrides_max_passes_and_context():
    plan_json = json.dumps(
        {
            "difficulty": "medium",
            "max_reasoning_passes": 3,
            "needs_full_context": True,
            "strategy": "full_brain",
            "target_expert_tokens": 900,
            "target_latency_ms": 12000,
            "speculation_mode": "aggressive",
            "notes_for_reasoner": "",
        }
    )
    controller = CoreController(
        DummyLLMClient([plan_json]),
        model_name="dummy-small",
        state_store=InMemoryStateStore(),
    )
    plan = controller.plan(
        question="Compare approaches",
        canvas=_canvas(),
        policy_name="openai_brain_costcut_v1",
        raw_text="longer text that exceeds thresholds " * 20,
    )
    assert plan.max_reasoning_passes <= 2
    assert plan.needs_full_context is False
    assert plan.target_expert_tokens <= 400
