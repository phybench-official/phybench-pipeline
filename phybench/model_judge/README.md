# Model Judge

LLM-as-Judge evaluation module for PhyBench Pipeline. Evaluates both **answer accuracy** and **solution process quality** by comparing student solutions against reference solutions, complementing the numeric EED scoring in the `evaluation` module.

## Overview

While the `evaluation` module compares final answers using Expression Edit Distance (EED), `model_judge` evaluates:
1. **Answer accuracy** (most important, 50% weight) — does the final answer match the reference?
2. **Solution process quality** (50% weight) — physical reasoning, mathematical derivation, and completeness compared to the reference solution

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

| Model               | Notes                                       |
| ------------------- | ------------------------------------------- |
| `claude-opus-4-6`   | Best reasoning quality, recommended default |
| `claude-sonnet-4-6` | Faster, slightly lower quality              |
| `gpt-4o`            | Good alternative if using OpenAI provider   |
| `o3-mini`           | Strong physics reasoning, higher cost       |

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

The system prompt is the core of the judge. The prompt uses strict evaluation criteria with a harsh grading philosophy. The judge model (default: gpt-4o) evaluates four dimensions by comparing against reference solutions, with answer accuracy as the most important criterion.

```toml
[model_judge.prompt]
system_prompt = """Harsh physics grader: Compare student vs reference solution strictly.

Score 0-100 on:
- Answer (50%): Match reference
- Physics (25%): Match reference
- Math (15%): Match reference
- Complete (10%): Match reference

Rules: Deduct for deviations. Wrong answer ≤40. Different approach ≤70. Be harsh.

Overall = 0.5×answer + 0.25×physics + 0.15×math + 0.1×complete

JSON: {"answer_accuracy_score": <int>, "physical_reasoning_score": <int>, "math_derivation_score": <int>, "completeness_score": <int>, "overall_score": <int>, "commentary": "<text>"}"""
```

**Prompt design notes:**

- The user turn includes: problem statement, **reference solution process**, reference answer, and student solution.
- `temperature=0` is hardcoded for reproducibility.
- The **judge model (gpt-4o) is responsible for determining answer correctness** by comparing mathematical equivalence.
- **Answer accuracy is weighted at 50%**, making it the most important criterion.
- The prompt uses a **harsh grading philosophy** with strict deductions for all deviations from reference.
- Wrong answers cap the overall_score at 40.
- Different approaches (even if valid) are capped at 50 to ensure reference-based evaluation.
- JSON-only output format prevents free-text contamination and enables reliable parsing.
- **Extremely concise**: Only 546 characters, making it easy to understand and maintain.


### Full prompt structure (both turns)

Every judge call sends exactly two messages:

**System message** — the `system_prompt` from config (see above).

**User message** — assembled at runtime from the ground truth and solution data:

```
## Problem
<problem statement from ground truth>

## Reference Solution
<reference solution process from ground truth>

## Reference Answer
<reference answer from ground truth>

## Student Solution
<full model_solution text from api_caller output>
```

The judge sees both the reference solution process and the student's complete solution, enabling comparison-based evaluation with emphasis on answer accuracy.

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
  "solution": "<reference solution process>",
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
  "answer_accuracy_score": 95,
  "physical_reasoning_score": 85,
  "math_derivation_score": 90,
  "completeness_score": 75,
  "overall_score": 88,
  "commentary": "The student's answer matches the reference answer. The solution correctly applies conservation of energy. Minor: one intermediate step could be more explicit.",
  "time_taken": 4.21
}
```

## Scoring Rubric

| Dimension                  | Weight | What it measures                                                                           |
| -------------------------- | ------ | ------------------------------------------------------------------------------------------ |
| `answer_accuracy_score`    | 50%    | **MOST IMPORTANT**: Does the final answer match the reference answer? |
| `physical_reasoning_score` | 25%    | Does the physics approach match the reference solution? |
| `math_derivation_score`    | 15%    | Do the mathematical steps match the reference derivation? |
| `completeness_score`       | 10%    | Are all reference elements present in the solution? |
| `overall_score`            | 100%   | Weighted combination: 50% answer + 25% physics + 15% math + 10% completeness |

Scores are integers 0–100. **Strict** scoring rules:
- **Wrong final answer → overall_score ≤ 40**
- Correct answer but different approach → overall_score ≤ 50
- Deduct points for all deviations from reference
- High scores require close match with reference

### Evaluation Philosophy

- Solutions must match the reference in both answer and approach to score well
- Any deviation from reference results in deductions
- Different approaches (even if valid) score lower to ensure reference-based evaluation
- Harsh grading: default to moderate-low scores unless solution closely matches reference

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
