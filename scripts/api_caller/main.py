import argparse
import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from api_config import ApiConfig, load_api_config  # noqa: E402
from client import (  # noqa: E402
    create_async_client,
    generate_solution_data,
    initialize_globals_from_config,
    read_problems,
)
from openai import AsyncOpenAI  # noqa: E402

APP_CONFIG: ApiConfig | None = None

task_queue: "queue.Queue[dict[str, Any] | None]"
result_queue: "queue.Queue[dict[str, Any] | None]"


def get_output_file(
    target_dir_path: Path, model_name: str, input_filename: str, output_template: str
) -> Path:
    """
    Generates the full path for the output JSON file for a given model using a template.

    Args:
        target_dir_path: The directory where the output file will be saved.
        model_name: The name of the model, used to create the filename.
        input_filename: The input filename (with or without .json extension).
        output_template: The output filename template with [input_file] and [model] placeholders.

    Returns:
        The absolute path to the output JSON file.
    """
    sanitized_model_name = model_name.replace("/", "_").replace(":", "_")

    # Extract filename without extension from input_filename (handle both with/without .json)
    input_path = Path(input_filename)
    input_base = input_path.stem

    # Replace placeholders in the template
    output_filename = output_template.replace("[input_file]", input_base).replace(
        "[model]", sanitized_model_name
    )

    # Ensure .json extension
    if not output_filename.endswith(".json"):
        output_filename += ".json"

    return target_dir_path / output_filename


def is_error_solution(solution: dict[str, Any]) -> bool:
    """
    Checks if the provided solution dictionary indicates an error during generation.

    Args:
        solution: The solution dictionary.

    Returns:
        True if the solution contains an error message, False otherwise.
    """
    sol_text: Any | None = solution.get("solution")

    if isinstance(sol_text, str) and sol_text.startswith("Error"):
        return True

    if solution.get("error"):
        return True

    if not sol_text or (isinstance(sol_text, str) and sol_text.strip() == ""):
        return True

    return False


def producer(
    problems: list[dict[str, Any]],
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
        print("No problems to process.")
        return

    print("🔍 Checking for existing solutions to avoid duplicates...")
    completed_tasks = check_existing_solutions(output_file)
    completed_for_model = completed_tasks.get(model, set())

    total_possible_tasks = len(problems) * repeat_count
    tasks_to_add = []
    skipped_count = 0

    for repeat_idx in range(repeat_count):
        for problem in problems:
            problem_id = problem.get("id", "N/A")
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

    print("📊 Task Summary:")
    print(f"  - Total possible tasks: {total_possible_tasks}")
    print(f"  - Already completed: {skipped_count}")
    print(f"  - Tasks to process: {len(tasks_to_add)}")

    if not tasks_to_add:
        print("All tasks already completed. Nothing to do.")
        return

    with tqdm(total=len(tasks_to_add), desc=pbar_desc, unit="task") as pbar:
        for task in tasks_to_add:
            task_queue.put(task)
            pbar.update(1)

    print("Producer: Finished enqueuing all tasks.")


async def consumer_task_processor(
    client: AsyncOpenAI, chat_timeout: float, max_retries: int
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
                        client, problem, model, repeat_idx, timeout=chat_timeout
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
                            "solution": f"Error after {max_retries} retries: {str(e)}",
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
            print(f"Consumer error: {e}")
            break


def run_consumer_loop(
    api_key: str, base_url: str, chat_timeout: float, max_retries: int
) -> None:
    """
    Wrapper to run the asyncio event loop for a single consumer.
    It creates and closes an API client for this consumer's lifecycle.
    """

    async def actual_processing_loop() -> None:
        client = create_async_client(api_key, base_url)
        try:
            await consumer_task_processor(client, chat_timeout, max_retries)
        finally:
            await client.close()

    try:
        asyncio.run(actual_processing_loop())
    except Exception as e:
        print(f"Consumer loop error: {e}")


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
                import shutil

                shutil.copy2(output_file, backup_file)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(solutions, f, indent=2, ensure_ascii=False)

            if backup_file.exists():
                backup_file.unlink()

            return

        except Exception as e:
            print(f"Write attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))
            else:
                backup_file = output_file.with_suffix(f"{output_file.suffix}.backup")
                if backup_file.exists():
                    import shutil

                    shutil.copy2(backup_file, output_file)
                    print(f"Restored from backup: {backup_file}")
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
            print(f"Warning: Could not load existing solutions: {e}")

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
                print("Timeout waiting for results. Continuing...")
                continue
            except Exception as e:
                print(f"Error in result writer: {e}")
                break

    all_solutions_for_file.extend(current_run_buffer)

    print(f"\n📝 Writing {len(all_solutions_for_file)} solutions to {output_file}...")
    sync_write_solutions(all_solutions_for_file, output_file)

    print("\n📊 Final Statistics:")
    print(f"  - Processed: {processed_count}")
    print(f"  - Successful: {success_count}")
    print(f"  - Errors: {error_count}")
    print(
        f"  - Success Rate: {success_count / processed_count * 100:.1f}%"
        if processed_count > 0
        else "  - Success Rate: 0%"
    )
    print(
        f"  - Average Time: {total_time / processed_count:.1f}s"
        if processed_count > 0
        else "  - Average Time: 0s"
    )


def parse_args(config: ApiConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel API caller for physics problems"
    )

    # Build default input file path from folder + filename
    default_input_path = None
    if config.input_folder and config.input_file:
        default_input_path = str(Path(config.input_folder) / config.input_file)

    parser.add_argument(
        "--input-file",
        default=default_input_path,
        help="Path to the input JSON file that contains problems (should contain fields: id, tag, content, solution, answer)",
    )
    parser.add_argument(
        "--output-dir",
        default=config.output_dir,
        help="Directory to store output files (individual model solution files)",
    )
    parser.add_argument("--model", default=config.model, help="Model name to use")
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=config.repeat_count,
        help="Number of times to repeat each problem",
    )
    parser.add_argument(
        "--num-consumers",
        type=int,
        default=config.num_consumers,
        help="Number of consumer threads",
    )

    return parser.parse_args()


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
        print(f"Warning: Could not check existing solutions: {e}")

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
        print(f"Model validation failed for '{model}': {e}")
        return False


def main() -> None:
    global APP_CONFIG, task_queue, result_queue

    APP_CONFIG = load_api_config()
    args = parse_args(APP_CONFIG)
    if not args.input_file:
        print(
            "Error: No input file specified. Use --input-file or set input_file in config."
        )
        return

    if not args.output_dir:
        print(
            "Error: No output directory specified. Use --output-dir or set output_dir in config."
        )
        return

    if not args.model or args.model.strip() == "":
        print("Error: No model specified. Use --model or set model in config.")
        return

    if not APP_CONFIG.api_key:
        print("Error: No API key specified. Please set api_key in config.")
        return

    if not APP_CONFIG.base_url:
        print("Error: No base URL specified. Please set base_url in config.")
        return

    print(f"🔍 Validating model '{args.model}'...")
    if not asyncio.run(
        validate_model(APP_CONFIG.api_key, APP_CONFIG.base_url, args.model)
    ):
        print(
            f"Error: Model '{args.model}' is not available or not working. Please check your model name and API configuration."
        )
        return
    print(f"✅ Model '{args.model}' validated successfully.")

    initialize_globals_from_config(APP_CONFIG.openai_o_model_keywords)
    problems = read_problems(args.input_file)
    if not problems:
        print("No problems loaded. Exiting.")
        return

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Get input filename from the input file path
    input_filename = Path(args.input_file).name
    output_template = APP_CONFIG.output_file or "[input_file]_[model]"

    output_file = get_output_file(
        output_dir_path, args.model, input_filename, output_template
    )

    task_queue = queue.Queue(maxsize=APP_CONFIG.max_task_queue_size or 0)
    result_queue = queue.Queue()

    total_tasks = len(problems) * args.repeat_count

    print("🚀 Starting parallel processing:")
    print(f"  - Model: {args.model}")
    print(f"  - Problems: {len(problems)}")
    print(f"  - Repeat count: {args.repeat_count}")
    print(f"  - Total tasks: {total_tasks}")
    print(f"  - Consumers: {args.num_consumers}")
    print(f"  - Output: {output_file}")

    producer_thread = threading.Thread(
        target=producer,
        args=(problems, args.model, args.repeat_count, output_file, "Producing tasks"),
    )

    consumer_threads = []
    for _ in range(args.num_consumers):
        thread = threading.Thread(
            target=run_consumer_loop,
            args=(
                APP_CONFIG.api_key,
                APP_CONFIG.base_url,
                APP_CONFIG.chat_timeout,
                APP_CONFIG.max_retries,
            ),
        )
        consumer_threads.append(thread)

    writer_thread = threading.Thread(
        target=result_writer, args=(output_file, total_tasks, 10, "Processing results")
    )

    producer_thread.start()

    for thread in consumer_threads:
        thread.start()

    writer_thread.start()

    producer_thread.join()
    print("Producer finished.")

    task_queue.join()
    print("All tasks completed.")

    for _ in range(args.num_consumers):
        task_queue.put(None)

    for thread in consumer_threads:
        thread.join()
    print("All consumers finished.")

    result_queue.put(None)
    writer_thread.join()
    print("Writer finished.")

    print("🎉 All processing completed!")


if __name__ == "__main__":
    main()
