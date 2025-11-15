"""CLI entry point for running experiments."""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer

from ..config import load_settings
from ..controller import BudgetContract
from ..llm_clients.dummy_client import DummyLLMClient
from ..llm_clients.openai_client import OpenAIClient
from ..policies import policy_manager
from ..state.canvas_store import CanvasStore
from ..utils.logging_utils import setup_logging
from .baseline_pipeline import run_baseline_pipeline
from .brain_pipeline import run_brain_pipeline

app = typer.Typer(help="Run baseline vs brain-as-llm experiments")


@app.command()
def run(
    testcases_path: Path = typer.Argument(..., help="Path to JSONL testcases"),
    output_dir: Path = typer.Option(Path("results"), help="Directory for experiment outputs"),
    use_dummy: bool = typer.Option(False, help="Use the dummy LLM client instead of real APIs"),
    policy_name: str = typer.Option("default_brain_v1", help="Policy name for tagging outputs"),
    canvas_store_dir: Path = typer.Option(Path("data/canvas_store"), help="Directory for cached canvases"),
) -> None:
    """Run the configured experiments and save JSONL results."""

    setup_logging()
    cases = _load_cases(testcases_path)
    if not cases:
        raise typer.BadParameter("No testcases found in file")

    settings = load_settings()
    small_client, large_client = _build_clients(use_dummy, settings)
    canvas_store = CanvasStore(canvas_store_dir)

    records, metrics = _execute_cases(
        cases=cases,
        small_client=small_client,
        large_client=large_client,
        settings=settings,
        canvas_store=canvas_store,
        policy_name=policy_name,
    )
    output_path = _write_results(records, output_dir, prefix="experiments")

    typer.secho(f"Saved results to {output_path}", fg=typer.colors.GREEN)
    summary = _summarize(metrics)
    typer.echo(json.dumps(summary, indent=2))
    _print_human_summary(summary)


@app.command()
def loop(
    testcases_path: Path = typer.Argument(..., help="Path to JSONL testcases"),
    output_dir: Path = typer.Option(Path("results/loop"), help="Directory for looped experiment outputs"),
    use_dummy: bool = typer.Option(True, help="Use the dummy LLM client (default for looped testing)"),
    min_iterations: int = typer.Option(10, help="Minimum iterations to run before checking expectations"),
    max_iterations: int = typer.Option(20, help="Maximum iterations to run"),
    policy_name: str = typer.Option("default_brain_v1", help="Policy name for tagging outputs"),
    canvas_store_dir: Path = typer.Option(Path("data/canvas_store"), help="Directory for cached canvases"),
) -> None:
    """Run repeated experiments until efficiency expectations are met."""

    if max_iterations < min_iterations:
        raise typer.BadParameter("max_iterations must be >= min_iterations")

    setup_logging()
    cases = _load_cases(testcases_path)
    if not cases:
        raise typer.BadParameter("No testcases found in file")

    settings = load_settings()
    small_client, large_client = _build_clients(use_dummy, settings)
    canvas_store = CanvasStore(canvas_store_dir)

    successful_iteration: Optional[int] = None

    for iteration in range(1, max_iterations + 1):
        records, metrics = _execute_cases(
            cases=cases,
            small_client=small_client,
            large_client=large_client,
            settings=settings,
            canvas_store=canvas_store,
            policy_name=policy_name,
        )
        output_path = _write_results(records, output_dir, prefix="loop", iteration=iteration)
        summary = _summarize(metrics)
        typer.echo(f"Iteration {iteration} saved to {output_path}")
        typer.echo(f"Summary: {json.dumps(summary)}")

        if iteration >= min_iterations and _expectations_met(metrics):
            successful_iteration = iteration
            typer.secho(f"Efficiency expectations met on iteration {iteration}", fg=typer.colors.GREEN)
            break

    if successful_iteration is None:
        typer.secho("Expectations not satisfied within allotted iterations.", fg=typer.colors.RED)


@app.command("analyze-policies")
def analyze_policies(
    results_glob: str = typer.Option("results/**/*.jsonl", help="Glob for results JSONL files"),
) -> None:
    """Aggregate prior runs and recommend policies."""

    paths = [Path(p) for p in glob.glob(results_glob, recursive=True)]
    if not paths:
        typer.secho("No results found for analysis.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    records = policy_manager.load_results(paths)
    if not records:
        typer.secho("Result files were empty.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    stats = policy_manager.compute_policy_stats(records)
    recommendations = policy_manager.recommend_policies_by_testcase(records)

    typer.echo("Policy performance summary:")
    for name, metric in stats.items():
        typer.echo(
            f"- {name}: expert_tokens={metric['avg_expert_tokens']:.2f}, "
            f"expert_latency={metric['avg_expert_latency']:.4f}s over {metric['samples']} samples"
        )

    typer.echo("\nPer-testcase recommendations:")
    for testcase, rec in recommendations.items():
        typer.echo(f"- {testcase}: best_policy={rec['policy_name']} (tokens={rec['expert_tokens']}, latency={rec['expert_latency']:.4f}s)")


def _build_clients(use_dummy: bool, settings: Any) -> Tuple[Any, Any]:
    if use_dummy:
        dummy = DummyLLMClient()
        return dummy, dummy
    if not settings.openai_api_key:
        raise typer.BadParameter("OPENAI_API_KEY needs to be configured or use --use-dummy/loop default")
    client = OpenAIClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return client, client


def _execute_cases(
    *,
    cases: List[Dict[str, Any]],
    small_client: Any,
    large_client: Any,
    settings: Any,
    canvas_store: CanvasStore,
    policy_name: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    lines: List[str] = []
    per_case_metrics: List[Dict[str, Any]] = []

    for case in cases:
        raw_path = Path(case["raw_text_path"])
        raw_text = raw_path.read_text(encoding="utf-8")
        question = case["question"]
        doc_id = case.get("doc_id") or case.get("id") or raw_path.stem
        budget = _budget_from_case(case)

        baseline_policy = f"{policy_name}_baseline"
        baseline_result = run_baseline_pipeline(
            large_client,
            raw_text=raw_text,
            question=question,
            model_name=settings.large_model,
            temperature=settings.default_temperature,
            max_tokens=settings.max_tokens,
            policy_name=baseline_policy,
        )
        brain_result = run_brain_pipeline(
            small_client=small_client,
            large_client=large_client,
            raw_text=raw_text,
            question=question,
            encoder_model=settings.small_model,
            controller_model=settings.small_model,
            reasoner_model=settings.large_model,
            canvas_store=canvas_store,
            doc_id=doc_id,
            policy_name=policy_name,
            budget=budget,
        )
        record = {
            "id": case.get("id"),
            "question": question,
            "baseline": baseline_result,
            "brain": brain_result,
            "expected_notes": case.get("expected_notes"),
            "doc_id": doc_id,
            "policy_name": policy_name,
        }
        per_case_metrics.append(
            {
                "baseline_tokens": baseline_result["usage"].get("total_tokens", 0),
                "baseline_latency": baseline_result["latency_seconds"],
                "brain_tokens": _brain_total_tokens(brain_result["usage"]),
                "brain_reasoner_tokens": brain_result["usage"]["reasoner_tokens"].get("total_tokens", 0),
                "brain_latency": brain_result["latency_seconds"],
                "brain_reasoner_latency": brain_result.get("reasoner_latency_seconds", 0.0),
                "id": case.get("id"),
            }
        )
        lines.append(json.dumps(record))

    return lines, per_case_metrics


def _budget_from_case(case: Dict[str, Any]) -> Optional[BudgetContract]:
    budget_data = case.get("budget")
    if not budget_data:
        return None
    return BudgetContract(
        max_latency_ms=budget_data.get("max_latency_ms"),
        max_expert_tokens=budget_data.get("max_expert_tokens"),
        priority=budget_data.get("priority", "normal"),
    )


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _brain_total_tokens(usage: Dict[str, Dict[str, int]]) -> int:
    keys = ["encoder_tokens", "controller_tokens", "reasoner_tokens"]
    total = 0
    for key in keys:
        total += int(usage.get(key, {}).get("total_tokens", 0))
    return total


def _summarize(metrics: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    data = list(metrics)
    if not data:
        return {}
    return {
        "cases": len(data),
        "avg_baseline_tokens": mean(item["baseline_tokens"] for item in data),
        "avg_brain_tokens": mean(item["brain_tokens"] for item in data),
        "avg_baseline_latency": mean(item["baseline_latency"] for item in data),
        "avg_brain_latency": mean(item["brain_latency"] for item in data),
        "avg_brain_reasoner_tokens": mean(item["brain_reasoner_tokens"] for item in data),
        "avg_brain_reasoner_latency": mean(item["brain_reasoner_latency"] for item in data),
    }


def _print_human_summary(summary: Dict[str, float]) -> None:
    if not summary:
        return
    cases = int(summary.get("cases", 0))
    baseline_tokens = summary.get("avg_baseline_tokens", 0.0)
    brain_tokens = summary.get("avg_brain_reasoner_tokens", summary.get("avg_brain_tokens", 0.0))
    baseline_latency = summary.get("avg_baseline_latency", 0.0)
    brain_latency = summary.get("avg_brain_latency", 0.0)

    token_delta = brain_tokens - baseline_tokens
    latency_delta = brain_latency - baseline_latency

    typer.echo(
        f"Phase 1 summary over {cases} cases -> "
        f"expert tokens baseline={baseline_tokens:.1f}, brain={brain_tokens:.1f} (Δ {token_delta:+.1f}); "
        f"latency baseline={baseline_latency:.3f}s, brain={brain_latency:.3f}s (Δ {latency_delta:+.3f}s)"
    )


def _expectations_met(metrics: Iterable[Dict[str, Any]]) -> bool:
    for item in metrics:
        if item["brain_reasoner_tokens"] > item["baseline_tokens"]:
            return False
        if item["brain_reasoner_latency"] > item["baseline_latency"]:
            return False
    return True


def _write_results(records: List[str], output_dir: Path, *, prefix: str, iteration: Optional[int] = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    iteration_suffix = f"_iter{iteration:02d}" if iteration is not None else ""
    path = output_dir / f"{prefix}_{timestamp}{iteration_suffix}.jsonl"
    path.write_text("\n".join(records), encoding="utf-8")
    return path


if __name__ == "__main__":
    app()
