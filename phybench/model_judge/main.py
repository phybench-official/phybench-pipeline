import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from openai import AsyncOpenAI

from phybench.config_loader import get_settings
from phybench.settings import AppSettings
from .client import SolutionItem, GroundTruthItem, judge_solution, write_result

app = typer.Typer()


@app.command()
def run(
    config_file: Annotated[str, typer.Option("--config-file")] = "config.toml",
    judge_model: Annotated[str | None, typer.Option("--judge-model")] = None,
    model_solutions_file: Annotated[str | None, typer.Option("--model-solutions-file")] = None,
):
    settings: AppSettings = get_settings(config_file)
    jset = settings.model_judge

    if judge_model:
        jset.model.judge_model = judge_model
    if model_solutions_file:
        jset.paths.model_solutions_file = model_solutions_file

    solutions_path = Path(jset.paths.model_solutions_dir) / f"{jset.paths.model_solutions_file}.json"
    gt_path = Path(jset.paths.gt_dir) / jset.paths.gt_file
    output_path = Path(jset.paths.output_dir) / f"{jset.paths.output_file}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solutions_data = json.loads(solutions_path.read_text(encoding="utf-8"))
    solutions = [SolutionItem.model_validate(x) for x in solutions_data]
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_dict = {x["id"]: GroundTruthItem.model_validate(x) for x in gt_data}

    provider = next((p for p in settings.providers if jset.model.judge_model in p.models), None)
    if not provider:
        logger.error(f"No provider found for judge model '{jset.model.judge_model}'")
        raise typer.Exit(1)

    client = AsyncOpenAI(api_key=provider.api_key, base_url=provider.base_url)

    async def run_all():
        sem = asyncio.Semaphore(jset.execution.num_consumers)

        async def bounded(sol: SolutionItem):
            async with sem:
                gt = gt_dict.get(sol.id)
                if not gt:
                    logger.warning(f"No ground truth for id={sol.id}, skipping")
                    return
                result = await judge_solution(
                    client, sol, gt,
                    jset.model.judge_model,
                    jset.prompt,
                    timeout=jset.execution.chat_timeout,
                    max_retries=jset.execution.max_retries,
                )
                await write_result(result, str(output_path))
                logger.info(f"Judged id={sol.id} overall={result['overall_score']}")

        await asyncio.gather(*[bounded(s) for s in solutions])

    asyncio.run(run_all())
    logger.success(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    app()
