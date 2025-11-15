import json
from pathlib import Path

from typer.testing import CliRunner

from brain_as_llm.experiments.runner import app
from brain_as_llm.experiments.baseline_pipeline import run_baseline_pipeline
from brain_as_llm.experiments.brain_pipeline import run_brain_pipeline
from brain_as_llm.llm_clients.dummy_client import DummyLLMClient
from brain_as_llm.state.canvas_store import CanvasStore


def test_brain_pipeline_runs_with_dummy_client(tmp_path: Path):
    summary = json.dumps(
        {
            "summary": "Key",
            "facts": [],
            "entities": [],
            "quotes": [],
            "notes_for_reasoner": "",
        }
    )
    plan = json.dumps(
        {
            "difficulty": "medium",
            "max_reasoning_passes": 1,
            "needs_full_context": False,
            "strategy": "small_only",
            "target_expert_tokens": None,
            "target_latency_ms": None,
            "speculation_mode": "off",
            "notes_for_reasoner": "",
        }
    )
    dummy = DummyLLMClient([summary, plan, "SMALL answer final"])
    store = CanvasStore(tmp_path)
    result = run_brain_pipeline(
        small_client=dummy,
        large_client=dummy,
        raw_text="Some text",
        question="What is key?",
        encoder_model="dummy-small",
        controller_model="dummy-small",
        reasoner_model="dummy-large",
        canvas_store=store,
        doc_id="doc-1",
        policy_name="test_policy",
    )
    assert result["policy_name"] == "test_policy"
    assert result["strategy_used"] == "small_only"
    assert result["answer"].startswith("SMALL")


def test_cli_runner_creates_results(tmp_path: Path):
    raw_path = tmp_path / "doc.txt"
    raw_path.write_text("Short raw text", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "t1", "raw_text_path": str(raw_path), "question": "Q?"}),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(cases_path),
            "--use-dummy",
            "--output-dir",
            str(tmp_path / "out"),
            "--policy-name",
            "test_policy",
        ],
    )
    assert result.exit_code == 0, result.output
    files = list((tmp_path / "out").glob("experiments_*.jsonl"))
    assert files, "Expected experiments output file"
    content = files[0].read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(content)
    assert record["policy_name"] == "test_policy"


def test_loop_command_runs_min_iterations(tmp_path: Path):
    raw_path = tmp_path / "doc.txt"
    raw_path.write_text("Short raw text", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps({"id": "t1", "raw_text_path": str(raw_path), "question": "Q?"}),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "loop",
            str(cases_path),
            "--use-dummy",
            "--output-dir",
            str(tmp_path / "loop_out"),
            "--min-iterations",
            "2",
            "--max-iterations",
            "2",
            "--policy-name",
            "policy_loop",
        ],
    )
    assert result.exit_code == 0, result.output
    files = sorted((tmp_path / "loop_out").glob("loop_*_iter*.jsonl"))
    assert len(files) == 2, "Expected two loop output files"


def test_analyze_policies_command(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sample_record = {
        "id": "t1",
        "baseline": {"usage": {"total_tokens": 100}, "latency_seconds": 0.1, "policy_name": "baseline"},
        "brain": {
            "usage": {"reasoner_tokens": {"total_tokens": 80}},
            "latency_seconds": 0.08,
            "policy_name": "brain_v1",
        },
    }
    (results_dir / "results.jsonl").write_text(json.dumps(sample_record), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analyze-policies",
            "--results-glob",
            str(results_dir / "*.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_baseline_pipeline_works_with_dummy_client():
    dummy = DummyLLMClient(["Simple baseline answer"])
    result = run_baseline_pipeline(
        dummy,
        raw_text="Context",
        question="Q?",
        model_name="dummy-large",
        policy_name="baseline_test",
    )
    assert result["answer"].startswith("Simple")
    assert result["policy_name"] == "baseline_test"
