# brain-as-llm

Research prototype that evaluates a brain-inspired workflow for large language models:

1. **Encoder / Compressor** condenses long context into a structured "canvas" of key points, entities, quotes, and notes.
2. **Controller / Cost Brain** inspects the canvas, budgets, and history to choose number of reasoning passes, cascade strategy, speculation mode, and latency/token targets.
3. **Reasoner** (small-only, cascade, or expert) performs one or more passes over the canvas to produce the final answer.

The repository also includes a baseline single-call pipeline and an experiment runner to compare metrics such as token usage and latency.

```
raw text ──▶ encoder ──▶ canvas ──▶ controller ──▶ control plan ──▶ reasoner ──▶ answer
             │                              ▲                         │
             └────────────── debug info ────┴──── optional full text ─┘
```

## Project layout

```
brain_as_llm/
  config.py            # Env-driven settings and model names
  llm_clients/         # Base interface, OpenAI client, and dummy client for tests
  encoder/             # TextEncoder that builds the canvas
  controller/          # CoreController deciding difficulty & reasoning steps
  reasoner/            # Multi-step CoreReasoner
  state/               # Simple JSON/in-memory state stores
  experiments/
    baseline_pipeline.py
    brain_pipeline.py
    runner.py          # Typer CLI to run experiments on JSONL testcases
  utils/               # Logging + metrics helpers
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables for your LLM provider (example shown for OpenAI-compatible APIs):

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional override
export BRAIN_SMALL_MODEL="gpt-4o-mini"
export BRAIN_LARGE_MODEL="gpt-4o"
```

## Running experiments

1. Prepare a JSONL file with testcases. Each line should look like:

```json
{
  "id": "test_001",
  "raw_text_path": "data/doc1.txt",
  "question": "What are the main risks mentioned?",
  "expected_notes": "Optional manual annotation"
}
```

2. Run the CLI:

```bash
python -m brain_as_llm.experiments.runner run data/testcases.jsonl --output-dir results --policy-name default_brain_v1
```

Use `--use-dummy` for an offline smoke test without external API calls. Canvases are cached in `data/canvas_store` by default; override with `--canvas-store-dir`.

The CLI saves detailed outputs to `results/experiments_<timestamp>.jsonl` and prints aggregate metrics (average tokens and latency for both pipelines).

## Phase 1 – OpenAI baseline vs brain experiment

Follow these steps to reproduce the Phase 1 validation run that compares the single-call baseline against the brain pipeline on the same workload:

1. **Install + test locally**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   pytest -q
   ```

2. **Configure OpenAI credentials and model choices**

   ```bash
   export OPENAI_API_KEY="sk-..."                     # do not commit this value
   export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional override
   export BRAIN_SMALL_MODEL="gpt-4o-mini"
   export BRAIN_LARGE_MODEL="gpt-4o"
   ```

   Leave `--use-dummy` unset so the runner talks to the real OpenAI backend.

3. **Create/verify `data/testcases.jsonl`**

   The repo ships with a 18-line JSONL file referencing documents in `data/`. Edit or extend it as needed; the format is:

   ```json
   {"id": "product_summary_bullets", "doc_id": "product_strategy", "raw_text_path": "data/product_strategy.txt", "question": "..."}
   ```

4. **Run the combined baseline + brain experiment**

   ```bash
   python -m brain_as_llm.experiments.runner run data/testcases.jsonl \
       --output-dir results \
       --policy-name openai_brain_v1
   ```

   The CLI runs baseline and brain back-to-back on every testcase, saves a JSONL report such as `results/experiments_20240101_120000.jsonl`, and prints a summary comparing token usage and latency across both strategies.

5. **Review outputs**

   * Inspect the JSONL file for per-case answers, plan metadata, token usage, and latency.
   * Check the final console summary that now spells out baseline vs brain averages plus deltas to confirm the OpenAI wiring behaved as expected.

6. **Interpret results later**

   Use the new helper command to re-summarize any saved JSONL file without rerunning the workloads:

   ```bash
   python -m brain_as_llm.experiments.runner interpret-results results/experiments_20250101_120000.jsonl
   ```

   This prints the same JSON summary object plus the human-readable baseline-vs-brain comparison banner.

Use `--use-dummy` only when you explicitly want to bypass network calls (e.g., local smoke tests or CI).

### Looping experiments

To run 10–20 iterations automatically (stopping once the brain pipeline matches or beats baseline latency/tokens), use:

```bash
python -m brain_as_llm.experiments.runner loop data/testcases.jsonl --use-dummy --policy-name cost_brain_v2
```

Each iteration is stored under `results/loop/loop_<timestamp>_iterXX.jsonl`.

### Policy analytics

Aggregate historical runs and compare policies:

```bash
python -m brain_as_llm.experiments.runner analyze-policies --results-glob "results/**/*.jsonl"
```

The policy manager computes average expert tokens/latency per `policy_name` and recommends the best historical policy per testcase/workload.

## Testing

```
pytest
```

Tests rely on the dummy LLM client, so they run without network access.

## Next steps

- Improve prompts for encoder/controller, tweak chunking heuristics, and add smarter aggregation.
- Persist richer state (e.g., conversation memory) and expose more telemetry for manual evaluation.
- Plug in alternative LLM backends (Anthropic, DeepSeek, local models) via the shared `LLMClient` interface.
