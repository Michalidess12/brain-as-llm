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

## Phase 1 – OpenAI baseline vs brain experiment

Follow this checklist to validate real OpenAI models end-to-end:

1. **Install & test**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pytest -q
   ```
   (The tests rely on the dummy backend so they run offline.)

2. **Configure OpenAI credentials**
   ```bash
   export OPENAI_API_KEY="sk-your-key"   # or set OPENAI_API_KEY in your shell profile
   export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
   export BRAIN_SMALL_MODEL="gpt-4o-mini"
   export BRAIN_LARGE_MODEL="gpt-4o"
   ```
   If you already exported a variable named `openai`, the config loader will fall back to it, but `OPENAI_API_KEY` is recommended.

3. **Prepare testcases**
   - Use `data/testcases.jsonl` (16 prompts across the bundled product and risk briefs) or create your own JSONL file following the same schema.

4. **Run the OpenAI-backed comparison**
   ```bash
   python -m brain_as_llm.experiments.runner run data/testcases.jsonl \
     --output-dir results \
     --policy-name openai_brain_v1
   ```
   (Do **not** pass `--use-dummy`; absence of the flag means real OpenAI calls.)

5. **Inspect outputs**
   - Detailed per-testcase records: `results/experiments_<timestamp>.jsonl`
   - CLI prints a JSON summary plus a short human-readable line comparing average expert tokens & latency for baseline vs. brain.

This workflow is our Phase 1 sanity check that both the single-shot baseline and the brain-as-LLM pipeline succeed against production models.

## Testing

```
pytest
```

Tests rely on the dummy LLM client, so they run without network access.

## Next steps

- Improve prompts for encoder/controller, tweak chunking heuristics, and add smarter aggregation.
- Persist richer state (e.g., conversation memory) and expose more telemetry for manual evaluation.
- Plug in alternative LLM backends (Anthropic, DeepSeek, local models) via the shared `LLMClient` interface.
