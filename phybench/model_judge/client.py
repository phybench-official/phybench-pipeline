import asyncio
import json
import re
import time
from typing import Any
from openai import AsyncOpenAI
from pydantic import BaseModel
from loguru import logger
from phybench.settings import ModelJudgePromptSettings

file_lock = asyncio.Lock()


class SolutionItem(BaseModel):
    id: int
    model: str
    model_solution: str   # full reasoning text from api_caller output
    model_answer: str     # extracted boxed answer


class GroundTruthItem(BaseModel):
    id: int
    content: str          # problem statement
    answer: str           # reference answer


def parse_judge_response(response_text: str) -> dict[str, Any]:
    """Extract JSON from judge LLM response. Falls back to zero scores on failure."""
    match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    logger.warning(f"Could not parse judge response as JSON: {response_text[:200]}")
    return {
        "physical_reasoning_score": 0,
        "math_derivation_score": 0,
        "completeness_score": 0,
        "overall_score": 0,
        "commentary": f"Parse error: {response_text[:100]}",
    }


async def judge_solution(
    client: AsyncOpenAI,
    solution: SolutionItem,
    gt: GroundTruthItem,
    judge_model: str,
    prompt_settings: ModelJudgePromptSettings,
    timeout: float = 300.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call judge LLM for one solution. Returns result dict."""
    user_prompt = (
        f"## Problem\n{gt.content}\n\n"
        f"## Reference Answer\n{gt.answer}\n\n"
        f"## Student Solution\n{solution.model_solution}"
    )
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": prompt_settings.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                timeout=timeout,
            )
            elapsed = time.time() - t0
            raw = response.choices[0].message.content or ""
            scores = parse_judge_response(raw)
            return {
                "id": solution.id,
                "model": solution.model,
                "judge_model": judge_model,
                "time_taken": elapsed,
                **scores,
            }
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for id={solution.id}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
    return {
        "id": solution.id,
        "model": solution.model,
        "judge_model": judge_model,
        "physical_reasoning_score": 0,
        "math_derivation_score": 0,
        "completeness_score": 0,
        "overall_score": 0,
        "commentary": "Error: all retries failed",
        "time_taken": 0.0,
    }


async def write_result(result: dict[str, Any], output_file: str) -> None:
    """Append one result to the output JSON file (thread-safe)."""
    async with file_lock:
        from pathlib import Path
        existing = []
        if Path(output_file).exists():
            with open(output_file, encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(result)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
