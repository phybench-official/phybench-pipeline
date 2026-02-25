# Model Judge

LLM-as-Judge evaluation module for PhyBench Pipeline. Scores the **reasoning process** of model solutions using a judge LLM, complementing the numeric EED scoring in the `evaluation` module.

## Overview

While the `evaluation` module compares final answers using Expression Edit Distance (EED), `model_judge` evaluates the *quality of the solution process* — physical reasoning, mathematical derivation, and completeness — by prompting a capable judge LLM.

### Pipeline Position

```
api_caller  →  evaluation   (EED score on final answer)
           ↘  model_judge   (LLM score on solution process)
```

Both `evaluation` and `model_judge` read from `data/model_solutions/` independently.

## Quick Start

```bash
# 1. Ensure config.toml has a provider that includes your judge model
# 2. Run api_caller first to generate model solutions
python -m phybench.api_caller

# 3. Run model_judge
python -m phybench.model_judge

# 4. Override judge model or input file via CLI
python -m phybench.model_judge --judge-model claude-opus-4-6 \
    --model-solutions-file test_gpt-4o
```

## Configuration

All settings live under `[model_judge]` in `config.toml`. Key sections:

### `[model_judge.model]`

```toml
[model_judge.model]
judge_model = "claude-opus-4-6"   # Must be listed in a [[providers]] block
```

Recommended judge models (in order of preference):

| Model | Notes |
|-------|-------|
| `claude-opus-4-6` | Best reasoning quality, recommended default |
| `claude-sonnet-4-6` | Faster, slightly lower quality |
| `gpt-4o` | Good alternative if using OpenAI provider |
| `o3-mini` | Strong physics reasoning, higher cost |

### `[model_judge.paths]`

```toml
[model_judge.paths]
model_solutions_dir = "data/model_solutions"
model_solutions_file = "{input_file}_{model}"   # e.g. test_gpt-4o
gt_dir = "data/ground_truth"
gt_file = "test.json"
output_dir = "data/judge_results"
output_file = "{input_file}_{model}_judge"      # e.g. test_gpt-4o_judge
```

### `[model_judge.execution]`

```toml
[model_judge.execution]
num_consumers = 5      # Concurrent judge API calls
chat_timeout = 300     # Seconds per call
max_retries = 3
max_task_queue_size = 100
```

### `[model_judge.prompt]`

The system prompt is the core of the judge. The default prompt instructs the judge to evaluate three dimensions and respond in strict JSON:

```toml
[model_judge.prompt]
system_prompt = """You are an expert physics professor evaluating a student's solution to a physics problem.
Your task is to assess the quality of the reasoning process, not just the final answer.
Evaluate the solution on three dimensions:
1. Physical Reasoning (0-100): Correct application of physics principles, laws, and concepts.
2. Mathematical Derivation (0-100): Correctness and rigor of mathematical steps and algebra.
3. Completeness (0-100): Whether all necessary steps are shown and the solution is well-structured.

Respond ONLY with a valid JSON object in this exact format:
{
  "physical_reasoning_score": <int 0-100>,
  "math_derivation_score": <int 0-100>,
  "completeness_score": <int 0-100>,
  "overall_score": <int 0-100>,
  "commentary": "<concise evaluation in 2-4 sentences>"
}"""
```

**Prompt design notes:**
- The user turn always includes: problem statement, reference answer, and the full model solution text.
- `temperature=0` is hardcoded for reproducibility.
- The judge is explicitly told to evaluate *process*, not just whether the final answer matches.
- JSON-only output format prevents free-text contamination and enables reliable parsing.

### Full prompt structure (both turns)

Every judge call sends exactly two messages:

**System message** — the `system_prompt` from config (see above).

**User message** — assembled at runtime from the ground truth and solution data:

```
## Problem
<problem statement from ground truth>

## Reference Answer
<reference answer from ground truth>

## Student Solution
<full model_solution text from api_caller output>
```

The judge sees the complete solution process (not just the boxed final answer), which is what enables process-level scoring.

## Input / Output Format

### Input: model_solutions JSON (from `api_caller`)

Each item must have:
```json
{
  "id": 42,
  "model": "gpt-4o",
  "model_solution": "<full reasoning text with LaTeX>",
  "model_answer": "\\frac{1}{2}mv^2"
}
```

### Input: ground truth JSON

Each item must have:
```json
{
  "id": 42,
  "content": "<problem statement>",
  "answer": "$$\\frac{1}{2}mv^2$$"
}
```

### Output: judge_results JSON

Each item written to `data/judge_results/`:
```json
{
  "id": 42,
  "model": "gpt-4o",
  "judge_model": "claude-opus-4-6",
  "physical_reasoning_score": 85,
  "math_derivation_score": 90,
  "completeness_score": 75,
  "overall_score": 83,
  "commentary": "The student correctly identified the relevant conservation law and applied it properly. The algebraic manipulation is correct but an intermediate step is skipped. The solution would benefit from explicitly stating boundary conditions.",
  "time_taken": 4.21
}
```

## Scoring Rubric

| Dimension | What it measures |
|-----------|-----------------|
| `physical_reasoning_score` | Correct identification and application of physics laws, principles, and physical intuition |
| `math_derivation_score` | Algebraic correctness, valid manipulations, no mathematical errors |
| `completeness_score` | All steps shown, solution is self-contained, no unjustified leaps |
| `overall_score` | Holistic quality — judge's integrated assessment (not a simple average) |

Scores are integers 0–100. A score of 0 means completely wrong or missing; 100 means flawless.

## Reproducibility Notes

- `temperature=0` is hardcoded in all API calls.
- The exact `system_prompt` used is stored in `config.toml` and logged at run start.
- Each output item records `judge_model`, so results from different judges are distinguishable.
- For research use, run each problem 3× with different seeds and average to reduce variance.
- The judge prompt version should be tracked in git alongside result files.

## Module Structure

```
phybench/model_judge/
├── __init__.py       # empty
├── __main__.py       # entry point: python -m phybench.model_judge
├── main.py           # typer CLI, async orchestration
├── client.py         # AsyncOpenAI calls, JSON parsing, file I/O
└── README.md         # this file
```
