import asyncio
import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm

from phybench.config_loader import get_settings
from phybench.logging_config import setup_logging
from phybench.path_resolver import PathResolver
from phybench.settings import APICallerPromptSettings, AppSettings, ProviderSettings

from .client import (
    ProblemItem,
    create_async_client,
    generate_solution_data,
    read_problems,
)

task_queue: "queue.Queue[dict[str, Any] | None]"
result_queue: "queue.Queue[dict[str, Any] | None]"

app: typer.Typer = typer.Typer()


def get_provider_for_model(
    model_name: str, providers: list[ProviderSettings]
) -> ProviderSettings | None:
    """Finds the correct provider configuration for a given model name."""
    for provider in providers:
        if model_name in provider.models:
            return provider
    return None


def is_error_solution(solution: dict[str, Any]) -> bool:
    """
    Checks if the provided solution dictionary indicates an error during generation.

    Args:
        solution: The solution dictionary.

    Returns:
        True if the solution contains an error message, False otherwise.
    """
    sol_text: Any | None = solution.get("model_solution")

    if isinstance(sol_text, str) and sol_text.startswith("Error"):
        return True

    if solution.get("error"):
        return True

    if not sol_text or (isinstance(sol_text, str) and sol_text.strip() == ""):
        return True

    return False


def producer(
    problems: list[ProblemItem],
    model: str,
    repeat_count: int,
    output_file: Path,
    pbar_desc: str = "Producing tasks",
) -> None:
    """
    Populates the task_queue with problems to be processed.

    Args:
        problems: A list of problem dictionaries.
        model: The model name to use for these tasks.
        repeat_count: How many times each problem should be processed.
        output_file: Path to output file for checking existing solutions.
        pbar_desc: Description for the tqdm progress bar.
    """

    if not problems:
        logger.warning("No problems to process")
        return

    logger.info("🔍 Checking for existing solutions to avoid duplicates...")
    completed_tasks = check_existing_solutions(output_file)
    completed_for_model = completed_tasks.get(model, set())

    total_possible_tasks = len(problems) * repeat_count
    tasks_to_add = []
    skipped_count = 0

    for repeat_idx in range(repeat_count):
        for problem in problems:
            problem_id = problem.id
            task_key = f"{problem_id}_{repeat_idx}"

            if task_key in completed_for_model:
                skipped_count += 1
                continue

            task = {
                "problem": problem,
                "model": model,
                "repeat_idx": repeat_idx,
                "output_file": str(output_file),
            }
            tasks_to_add.append(task)

    logger.info("📊 Task Summary:")
    logger.info(f"  - Total possible tasks: {total_possible_tasks}")
    logger.info(f"  - Already completed: {skipped_count}")
    logger.info(f"  - Tasks to process: {len(tasks_to_add)}")

    if not tasks_to_add:
        logger.info("All tasks already completed. Nothing to do.")
        return

    with tqdm(total=len(tasks_to_add), desc=pbar_desc, unit="task") as pbar:
        for task in tasks_to_add:
            task_queue.put(task)
            pbar.update(1)

    logger.info("Producer: Finished enqueuing all tasks")


async def consumer_task_processor(
    client: AsyncOpenAI,
    chat_timeout: float,
    max_retries: int,
    prompt_settings: APICallerPromptSettings,
) -> None:
    """
    Continuously fetches tasks from task_queue, processes them using generate_solution_data,
    retries on failure, and puts successful or final-error results onto result_queue.

    Args:
        client: An active AsyncOpenAI client.
        chat_timeout: Timeout in seconds for the API call.
        max_retries: Maximum number of retries for a task.
    """

    while True:
        try:
            task = task_queue.get(timeout=1.0)
            if task is None:
                break

            problem = task["problem"]
            model = task["model"]
            repeat_idx = task["repeat_idx"]

            retry_count = 0
            while retry_count <= max_retries:
                try:
                    solution_data = await generate_solution_data(
                        client,
                        problem,
                        model,
                        repeat_idx,
                        timeout=chat_timeout,
                        prompt_settings=prompt_settings,
                    )

                    if not is_error_solution(solution_data):
                        result_queue.put(solution_data)
                        break
                    elif retry_count == max_retries:
                        result_queue.put(solution_data)
                        break
                    else:
                        retry_count += 1
                        await asyncio.sleep(min(2.0**retry_count, 30.0))

                except Exception as e:
                    if retry_count == max_retries:
                        error_solution = {
                            "id": problem.get("id", "N/A"),
                            "model": model,
                            "model_solution": f"Error after {max_retries} retries: {str(e)}",
                            "model_answer": "",
                            "timestamp": time.time(),
                            "time_taken": 0.0,
                            "repeat_index": repeat_idx,
                            "error_message": str(e),
                        }
                        result_queue.put(error_solution)
                        break
                    else:
                        retry_count += 1
                        await asyncio.sleep(min(2.0**retry_count, 30.0))

            task_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            break


def run_consumer_loop(
    api_key: str,
    base_url: str,
    chat_timeout: float,
    max_retries: int,
    prompt_settings: APICallerPromptSettings,
) -> None:
    """
    Wrapper to run the asyncio event loop for a single consumer.
    It creates and closes an API client for this consumer's lifecycle.
    """

    async def actual_processing_loop() -> None:
        client = create_async_client(api_key, base_url)
        try:
            await consumer_task_processor(
                client, chat_timeout, max_retries, prompt_settings
            )
        finally:
            await client.close()

    try:
        asyncio.run(actual_processing_loop())
    except Exception as e:
        logger.error(f"Consumer loop error: {e}")


def sync_write_solutions(solutions: list[dict[str, Any]], output_file: Path) -> None:
    """
    Synchronously writes a list of solution dictionaries to a JSON file with backup/recovery mechanism.

    Args:
        solutions: A list of solution dictionaries (representing the entire dataset to be written).
        output_file: The path to the output JSON file.
    """

    max_retries = 3

    for attempt in range(max_retries):
        try:
            backup_file = output_file.with_suffix(f"{output_file.suffix}.backup")

            if output_file.exists():
                shutil.copy2(output_file, backup_file)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(solutions, f, indent=2, ensure_ascii=False)

            if backup_file.exists():
                backup_file.unlink()

            return

        except Exception as e:
            logger.error(f"Write attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))
            else:
                backup_file = output_file.with_suffix(f"{output_file.suffix}.backup")
                if backup_file.exists():
                    shutil.copy2(backup_file, output_file)
                    logger.info(f"Restored from backup: {backup_file}")
                    backup_file.unlink()


def result_writer(
    output_file: Path,
    total_tasks_expected: int,
    batch_size: int = 10,
    pbar_desc: str = "Writing results",
) -> None:
    """
    Fetches processed solutions from result_queue and writes them to the output file.
    It loads existing solutions from the file first, appends new ones, and writes all back at the end.

    Args:
        output_file: The path to the output JSON file.
        total_tasks_expected: The total number of tasks that consumers are expected to process.
                              Used for the tqdm progress bar total.
        batch_size: Number of solutions to buffer in memory (currently not directly tied to write frequency).
        pbar_desc: Description for the tqdm progress bar.
    """

    existing_solutions: list[dict[str, Any]] = []
    if output_file.exists():
        try:
            with open(output_file, encoding="utf-8") as f:
                existing_solutions = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing solutions: {e}")

    all_solutions_for_file: list[dict[str, Any]] = existing_solutions

    current_run_buffer: list[dict[str, Any]] = []

    processed_count = 0
    success_count = 0
    error_count = 0
    total_time = 0.0

    with tqdm(total=total_tasks_expected, desc=pbar_desc, unit="result") as pbar:
        while processed_count < total_tasks_expected:
            try:
                solution = result_queue.get(timeout=5.0)
                if solution is None:
                    break

                current_run_buffer.append(solution)
                processed_count += 1

                if is_error_solution(solution):
                    error_count += 1
                else:
                    success_count += 1

                time_taken = solution.get("time_taken", 0.0)
                total_time += time_taken

                avg_time = total_time / processed_count if processed_count > 0 else 0.0
                success_rate = (
                    (success_count / processed_count) * 100
                    if processed_count > 0
                    else 0.0
                )

                pbar.set_description(
                    f"{pbar_desc} | Success: {success_rate:.1f}% | Avg: {avg_time:.1f}s"
                )
                pbar.update(1)

                result_queue.task_done()

            except queue.Empty:
                logger.debug("Timeout waiting for results. Continuing...")
                continue
            except Exception as e:
                logger.error(f"Error in result writer: {e}")
                break

    all_solutions_for_file.extend(current_run_buffer)

    logger.info(f"📝 Writing {len(all_solutions_for_file)} solutions to {output_file}")
    sync_write_solutions(all_solutions_for_file, output_file)

    logger.info("📊 Final Statistics:")
    logger.info(f"  - Processed: {processed_count}")
    logger.info(f"  - Successful: {success_count}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(
        f"  - Success Rate: {success_count / processed_count * 100:.1f}%"
        if processed_count > 0
        else "  - Success Rate: 0%"
    )
    logger.info(
        f"  - Average Time: {total_time / processed_count:.1f}s"
        if processed_count > 0
        else "  - Average Time: 0s"
    )


def check_existing_solutions(output_file: Path) -> dict[str, set[str]]:
    """
    Checks existing solutions in the output file and returns a dictionary
    mapping model names to sets of completed task keys.

    Args:
        output_file: Path to the output JSON file

    Returns:
        Dictionary with model names as keys and sets of completed task keys as values
    """
    completed_tasks: dict[str, set[str]] = {}

    if not output_file.exists():
        return completed_tasks

    try:
        with open(output_file, encoding="utf-8") as f:
            existing_solutions = json.load(f)

        for solution in existing_solutions:
            model = solution.get("model", "unknown")
            problem_id = solution.get("id", "N/A")
            repeat_idx = solution.get("repeat_index", 0)

            task_key = f"{problem_id}_{repeat_idx}"

            if model not in completed_tasks:
                completed_tasks[model] = set()
            completed_tasks[model].add(task_key)

    except Exception as e:
        logger.warning(f"Could not check existing solutions: {e}")

    return completed_tasks


async def validate_model(
    api_key: str, base_url: str, model: str, timeout: float = 10.0
) -> bool:
    """
    Validates that the specified model is available and working by making a simple test API call.

    Args:
        api_key: The API key for authentication
        base_url: The base URL for the API
        model: The model name to validate
        timeout: Timeout for the validation call in seconds

    Returns:
        True if the model is valid and accessible, False otherwise
    """
    try:
        client = create_async_client(api_key, base_url)
        try:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                timeout=timeout,
            )
            return True
        finally:
            await client.close()
    except Exception as e:
        logger.error(f"Model validation failed for '{model}': {e}")
        return False


@app.command()  # type: ignore
def main(
    config_file: Annotated[
        Path,
        typer.Option(
            "--config-file",
            "-c",
            help="Path to the configuration file (e.g., config.toml)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("config.toml"),
    model: Annotated[
        str | None,
        typer.Option(help="Model to use for API calls (must be in config)"),
    ] = None,
    input_dir: Annotated[
        Path | None, typer.Option(help="Directory containing input JSON files")
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option(help="Input problems filename (JSON)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory to store output files (individual model solution files)"
        ),
    ] = None,
    output_file: Annotated[
        str | None,
        typer.Option(help="Output filename for model solutions (JSON)."),
    ] = None,
    repeat_count: Annotated[
        int | None, typer.Option(help="Number of times to run each problem.")
    ] = None,
    num_consumers: Annotated[
        int | None, typer.Option(help="Number of concurrent API calls.")
    ] = None,
    chat_timeout: Annotated[
        float | None, typer.Option(help="API call timeout (seconds).")
    ] = None,
    max_retries: Annotated[
        int | None, typer.Option(help="Max retries for failed API calls.")
    ] = None,
    log_dir: Annotated[
        Path | None, typer.Option(help="Directory to store log files")
    ] = None,
    log_file: Annotated[
        str | None,
        typer.Option(help="Log filename."),
    ] = None,
) -> None:
    global task_queue, result_queue
    try:
        settings: AppSettings = get_settings(str(config_file))
        logger.info(f"Loaded settings from {config_file}")
    except FileNotFoundError as e:
        logger.error(f"Missing config file: {e}")
        if str(config_file) == "config.toml":
            logger.error(
                "Please create a 'config.toml' file or use --config-file to specify a path."
            )
        raise typer.Exit(code=1) from e

    # Override settings with CLI options if provided
    if model:
        settings.api_caller.model.model = model
    if input_dir:
        settings.api_caller.paths.input_dir = str(input_dir)
    if input_file:
        settings.api_caller.paths.input_file = input_file
    if output_dir:
        settings.api_caller.paths.output_dir = str(output_dir)
    if output_file:
        settings.api_caller.paths.output_file = output_file
    if repeat_count is not None:
        settings.api_caller.execution.repeat_count = repeat_count
    if num_consumers is not None:
        settings.api_caller.execution.num_consumers = num_consumers
    if chat_timeout is not None:
        settings.api_caller.execution.chat_timeout = int(chat_timeout)
    if max_retries is not None:
        settings.api_caller.execution.max_retries = max_retries
    if log_dir:
        settings.logging.log_dir = str(log_dir)
    if log_file:
        settings.logging.log_file = log_file

    if (
        not settings.api_caller.model.model
        or settings.api_caller.model.model.strip() == ""
    ):
        logger.error("No model specified. Use --model or set model in config.")
        raise typer.Exit(code=1)

    resolver = PathResolver(
        settings.api_caller.model.model,
        settings.api_caller.paths.input_dir,
        settings.api_caller.paths.input_file,
        settings.api_caller.paths.output_dir,
        settings.api_caller.paths.output_file,
        settings.evaluation.paths.gt_dir,
        settings.evaluation.paths.gt_file,
        settings.evaluation.paths.model_answers_dir,
        settings.evaluation.paths.model_answers_file,
        settings.evaluation.paths.output_dir,
        settings.evaluation.paths.output_file,
        settings.logging.log_dir,
        settings.logging.log_file,
    )

    setup_logging(
        log_file=resolver.get_log_file(),
        log_level=settings.logging.file_level,
        console_level=settings.logging.console_level,
    )

    input_file_path = resolver.get_api_caller_input_file()

    provider = get_provider_for_model(
        settings.api_caller.model.model, settings.providers
    )

    if not provider:
        logger.error(
            f"No provider found for model '{settings.api_caller.model.model}'. Please add it to your config.toml"
        )
        raise typer.Exit(code=1)

    logger.info(
        f"🔍 Validating model '{settings.api_caller.model.model}' using provider '{provider.name}'..."
    )
    if not asyncio.run(
        validate_model(
            provider.api_key, provider.base_url, settings.api_caller.model.model
        )
    ):
        logger.error(
            f"Model '{settings.api_caller.model.model}' is not available. Please check your model name and API configuration."
        )
        raise typer.Exit(code=1)
    logger.success(
        f"✅ Model '{settings.api_caller.model.model}' validated successfully."
    )

    problems = read_problems(str(input_file_path))
    if not problems:
        logger.error("No problems loaded. Exiting.")
        return

    output_file_path = resolver.get_api_caller_output_file()

    task_queue = queue.Queue(maxsize=settings.api_caller.execution.max_task_queue_size)
    result_queue = queue.Queue()

    total_tasks = len(problems) * settings.api_caller.execution.repeat_count

    logger.info("🚀 Starting parallel processing:")
    logger.info(f"  - Model: {settings.api_caller.model.model}")
    logger.info(f"  - Problems: {len(problems)}")
    logger.info(f"  - Repeat count: {settings.api_caller.execution.repeat_count}")
    logger.info(f"  - Total tasks: {total_tasks}")
    logger.info(f"  - Consumers: {settings.api_caller.execution.num_consumers}")
    logger.info(f"  - Output: {output_file_path}")

    producer_thread = threading.Thread(
        target=producer,
        args=(
            problems,
            settings.api_caller.model.model,
            settings.api_caller.execution.repeat_count,
            output_file_path,
            "Producing tasks",
        ),
    )

    consumer_threads = []
    for _ in range(settings.api_caller.execution.num_consumers):
        thread = threading.Thread(
            target=run_consumer_loop,
            args=(
                provider.api_key,
                provider.base_url,
                settings.api_caller.execution.chat_timeout,
                settings.api_caller.execution.max_retries,
                settings.api_caller.prompt,
            ),
        )
        consumer_threads.append(thread)

    writer_thread = threading.Thread(
        target=result_writer,
        args=(output_file_path, total_tasks, 10, "Processing results"),
    )

    producer_thread.start()

    for thread in consumer_threads:
        thread.start()

    writer_thread.start()

    producer_thread.join()
    logger.info("Producer finished")

    task_queue.join()
    logger.info("All tasks completed")

    for _ in range(settings.api_caller.execution.num_consumers):
        task_queue.put(None)

    for thread in consumer_threads:
        thread.join()
    logger.info("All consumers finished")

    result_queue.put(None)
    writer_thread.join()
    logger.info("Writer finished")

    logger.success("🎉 All processing completed!")


if __name__ == "__main__":
    app()
