import threading
import queue
import asyncio
from tqdm import tqdm
import os
import json
import argparse
import time
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Callable, Coroutine

# Ensure config_loader and api_run can be imported
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import from local modules
from config_loader import load_config, AppConfig, CONFIG_FILE_NAME
from api_run import (
    read_problems,
    generate_solution_data,
    create_async_client,
    extract_boxed_answer, # Kept for completeness, though not directly used in this file's main flow
    initialize_globals_from_config
)
from openai import AsyncOpenAI # Needed for type hint

# Global AppConfig instance - to be loaded in main()
APP_CONFIG: Optional[AppConfig] = None

# Queues for inter-thread communication - size will be set from APP_CONFIG
task_queue: "queue.Queue[Optional[Dict[str, Any]]]"
result_queue: "queue.Queue[Optional[Dict[str, Any]]]"


def get_output_file(target_dir_path: Path, model_name: str) -> Path:
    """
    Generates the full path for the output JSON file for a given model.

    Args:
        target_dir: The directory where the output file will be saved.
        model_name: The name of the model, used to create the filename.

    Returns:
        The absolute path to the output JSON file.
    """
    sanitized_model_name = model_name.replace("/", "_").replace(":", "_")
    return target_dir_path / f"{sanitized_model_name}.json"


def is_error_solution(solution: Dict[str, Any]) -> bool:
    """
    Checks if the provided solution dictionary indicates an error during generation.

    Args:
        solution: The solution dictionary.

    Returns:
        True if the solution contains an error message, False otherwise.
    """
    sol_text: Optional[Any] = solution.get("solution") # Can be str or None
    
    # Check if sol_text is a string and starts with "Error"
    if isinstance(sol_text, str) and sol_text.startswith("Error"):
        return True
    
    # Check if explicit error field exists
    if solution.get("error"):
        return True
    
    # Check if solution is empty but should have content
    if not sol_text or (isinstance(sol_text, str) and sol_text.strip() == ""):
        return True
        
    return False


def producer(
    problems: List[Dict[str, Any]],
    model: str,
    repeat_times: int,
    output_file: Path,
    pbar_desc: str = "Producing tasks"
) -> None:
    """
    Populates the task_queue with problems to be processed.

    Args:
        problems: A list of problem dictionaries.
        model: The model name to use for these tasks.
        repeat_times: How many times each problem should be processed.
        output_file: Path to output file for checking existing solutions.
        pbar_desc: Description for the tqdm progress bar.
    """
    if not problems:
        print("Producer: No problems to produce.")
        return
    
    # Check existing solutions to avoid duplicates
    print("🔍 Checking for existing solutions to avoid duplicates...")
    completed_tasks = check_existing_solutions(output_file)
    completed_for_model = completed_tasks.get(model, set())
    
    # Calculate total tasks and filter out completed ones
    total_possible_tasks = len(problems) * repeat_times
    tasks_to_add = []
    skipped_count = 0
    
    for repeat_idx in range(repeat_times):
        for problem in problems:
            problem_id = problem.get("id")
            
            # Skip if this problem/model combination is already completed
            if problem_id in completed_for_model:
                skipped_count += 1
                continue
                
            tasks_to_add.append({"problem": problem, "model": model, "repeat_idx": repeat_idx})
    
    print(f"📊 Task Summary:")
    print(f"  - Total possible tasks: {total_possible_tasks}")
    print(f"  - Already completed: {skipped_count}")
    print(f"  - Tasks to process: {len(tasks_to_add)}")
    
    if not tasks_to_add:
        print("✅ All tasks already completed. Nothing to process.")
        return
        
    with tqdm(total=len(tasks_to_add), desc=pbar_desc, unit="task") as pbar:
        for task in tasks_to_add:
            task_queue.put(task)
            pbar.update(1)
            
    print("Producer: Finished enqueuing all tasks.")


async def consumer_task_processor(
    client: AsyncOpenAI, # OpenAI client instance
    chat_timeout: float,
    max_retries: int
) -> None:
    """
    Continuously fetches tasks from task_queue, processes them using generate_solution_data,
    retries on failure, and puts successful or final-error results onto result_queue.
    Enhanced version with structured error handling and progressive retry delays.

    Args:
        client: An active AsyncOpenAI client.
        chat_timeout: Timeout in seconds for the API call.
        max_retries: Maximum number of retries for a task.
    """
    while True:
        task_info = task_queue.get()
        if task_info is None:  # Sentinel to stop the consumer
            task_queue.task_done() # Important to mark sentinel as done
            break

        problem_id = task_info["problem"].get("id", "UnknownID")
        model_name = task_info["model"]
        current_solution: Optional[Dict[str, Any]] = None
        last_error = None

        for attempt in range(max_retries):
            try:
                # Use the centralized generate_solution_data function
                current_solution = await generate_solution_data(
                    async_client_instance=client, # Pass the client
                    problem=task_info["problem"],
                    model=task_info["model"],
                    repeat_idx=task_info["repeat_idx"],
                    timeout=chat_timeout
                )

                if not is_error_solution(current_solution):
                    result_queue.put(current_solution)
                    break  # Success, exit retry loop
                else:
                    # Enhanced error information extraction
                    error_info = current_solution.get("solution", "Unknown error")
                    last_error = error_info
                    
                    print(f"Task for problem {problem_id} (model {model_name}) failed attempt {attempt + 1}/{max_retries}. Error: {error_info}")
                    
                    if attempt < max_retries - 1:
                        # Progressive retry delay: 1s, 2s, 4s, etc.
                        delay = 2 ** attempt  # Exponential backoff
                        await asyncio.sleep(delay)
                    else: 
                        # Last attempt failed - create structured error solution
                        print(f"Task for problem {problem_id} (model {model_name}) failed after {max_retries} attempts. Storing structured error result.")
                        
                        # Create enhanced error solution with metadata
                        structured_error_solution = {
                            "id": problem_id,
                            "model": model_name,
                            "solution": f"Error after {max_retries} attempts: {last_error}",
                            "answer": "",
                            "time_taken": current_solution.get("time_taken", 0.0),
                            "repeat_idx": task_info["repeat_idx"],
                            "tokens": current_solution.get("tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "error": last_error,
                            "error_metadata": {
                                "max_retries": max_retries,
                                "failed_attempts": max_retries,
                                "final_attempt_error": last_error,
                                "error_type": "max_retries_exceeded"
                            }
                        }
                        
                        result_queue.put(structured_error_solution)
                        
            except Exception as e:
                last_error = f"Unexpected error in consumer: {type(e).__name__}: {e}"
                print(f"Unexpected error processing problem {problem_id} (attempt {attempt + 1}/{max_retries}): {last_error}")
                
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    # Create emergency error solution
                    emergency_error_solution = {
                        "id": problem_id,
                        "model": model_name,
                        "solution": f"Critical error after {max_retries} attempts: {last_error}",
                        "answer": "",
                        "time_taken": 0.0,
                        "repeat_idx": task_info["repeat_idx"],
                        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error": last_error,
                        "error_metadata": {
                            "max_retries": max_retries,
                            "failed_attempts": max_retries,
                            "final_attempt_error": last_error,
                            "error_type": "critical_exception"
                        }
                    }
                    
                    result_queue.put(emergency_error_solution)

        task_queue.task_done()


def run_consumer_loop(
    api_key: str, 
    base_url: str, 
    chat_timeout: float, 
    max_retries: int
) -> None:
    """
    Wrapper to run the asyncio event loop for a single consumer.
    It creates and closes an API client for this consumer's lifecycle.
    """
    # print(f"Consumer thread {threading.get_ident()} starting.") # Debug
    async def actual_processing_loop():
        # print(f"Consumer {threading.get_ident()}: Creating API client.") # Debug
        client = create_async_client(api_key=api_key, base_url=base_url)
        try:
            await consumer_task_processor(client, chat_timeout, max_retries)
        finally:
            # print(f"Consumer {threading.get_ident()}: Closing API client.") # Debug
            await client.close()
            # print(f"Consumer {threading.get_ident()}: API client closed.") # Debug
            
    try:
        asyncio.run(actual_processing_loop())
    except Exception as e:
        print(f"Error in consumer loop (thread {threading.get_ident()}): {e}")
    # print(f"Consumer thread {threading.get_ident()} finished.") # Debug


def sync_write_solutions(solutions: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Synchronously writes a list of solution dictionaries to a JSON file with backup/recovery mechanism.
    Enhanced version with backup creation and error recovery.

    Args:
        solutions: A list of solution dictionaries (representing the entire dataset to be written).
        output_file: The path to the output JSON file.
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Ensure the directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            backup_file = output_file.with_suffix(f"{output_file.suffix}.backup")
            if output_file.exists():
                try:
                    import shutil
                    shutil.copy2(output_file, backup_file)
                except Exception:
                    pass  # Backup failure doesn't affect main flow
            
            # Write solutions to file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(solutions, f, indent=4, ensure_ascii=False)
            
            # Verify write was successful
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    json.load(f)  # Verify JSON format is correct
                print(f"✅ Successfully wrote {len(solutions)} solutions to {output_file}")
                return
            except Exception:
                if attempt < max_retries - 1:
                    print(f"⚠️  Write verification failed, retrying {attempt + 1}/{max_retries}")
                    time.sleep(0.5)
                    continue
                else:
                    raise
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Write failed, retrying {attempt + 1}/{max_retries}: {e}")
                time.sleep(1)
            else:
                print(f"❌ Error writing solutions (final failure): {e}")
                # Try to write to emergency backup file
                emergency_file = output_file.with_suffix(f"{output_file.suffix}.emergency_{int(time.time())}")
                try:
                    with open(emergency_file, "w", encoding="utf-8") as f:
                        json.dump(solutions, f, indent=4, ensure_ascii=False)
                    print(f"🆘 Written to emergency backup: {emergency_file}")
                except Exception:
                    print(f"💥 Emergency backup failed")
                break


def result_writer(
    output_file: Path,
    total_tasks_expected: int,
    batch_size: int = 10, # This batch_size is for buffer, actual write is at the end
    pbar_desc: str = "Writing results"
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
    existing_solutions: List[Dict[str, Any]] = []
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    loaded_data = json.loads(content)
                    if isinstance(loaded_data, list):
                        existing_solutions = loaded_data
                    else:
                        print(f"Warning: Existing content in {output_file} is not a list. Starting fresh.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing JSON from {output_file}. Starting with an empty list.")
        except Exception as e:
            print(f"Warning: Could not read {output_file} ({e}). Starting with an empty list.")
    
    # This list will accumulate all solutions: those initially in the file + new ones from this run.
    all_solutions_for_file: List[Dict[str, Any]] = existing_solutions
    
    # Buffer for results from this specific run before appending to all_solutions_for_file
    # This is mostly for managing updates to the progress bar or potential intermittent saves if implemented.
    # Given current single final save, buffer isn't strictly for batch *writing*.
    current_run_buffer: List[Dict[str, Any]] = [] 
    
    # Enhanced statistics tracking
    processed_count = 0
    success_count = 0
    error_count = 0
    total_time = 0.0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    error_types = {}
    success_rate = 0.0
    avg_time = 0.0
    
    with tqdm(total=total_tasks_expected, desc=pbar_desc, unit="result") as pbar:
        while True:
            try:
                # Timeout to prevent indefinite blocking if producer/consumers have issues
                # and to allow checking processed_count against total_tasks_expected
                result = result_queue.get(timeout=5.0) 
            except queue.Empty:
                # Check if all expected tasks are processed. This is a failsafe.
                # The main exit condition is the None sentinel.
                if processed_count >= total_tasks_expected:
                    print("Result Writer: Queue empty and all expected tasks processed. Preparing to exit.")
                    break 
                continue # Continue waiting if not all tasks accounted for

            if result is None:  # Sentinel to stop the writer
                result_queue.task_done() # Mark sentinel as done
                break

            current_run_buffer.append(result)
            processed_count += 1
            
            # Enhanced statistics tracking
            is_error = is_error_solution(result)
            if is_error:
                error_count += 1
                # Track error types
                if "error_metadata" in result and "error_type" in result["error_metadata"]:
                    error_type = result["error_metadata"]["error_type"]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                status_icon = "❌"
            else:
                success_count += 1
                status_icon = "✅"
            
            # Track timing and token usage
            time_taken = result.get("time_taken", 0.0)
            total_time += time_taken
            
            tokens = result.get("tokens", {})
            if isinstance(tokens, dict):
                total_tokens["prompt_tokens"] += tokens.get("prompt_tokens", 0)
                total_tokens["completion_tokens"] += tokens.get("completion_tokens", 0) 
                total_tokens["total_tokens"] += tokens.get("total_tokens", 0)
            
            # Calculate real-time statistics
            success_rate = (success_count / processed_count) * 100 if processed_count > 0 else 0
            avg_time = total_time / processed_count if processed_count > 0 else 0
            
            # Enhanced progress bar description with real-time stats
            problem_id = result.get("id", "?")
            model = result.get("model", "?")
            pbar.set_description(
                f"{status_icon} P{problem_id} | Success: {success_rate:.1f}% | Avg: {avg_time:.1f}s | Tokens: {total_tokens['total_tokens']:,}"
            )
            pbar.update(1)
            result_queue.task_done()
            
            # Optional: if you wanted to write in batches (not current design which writes once at end)
            # if len(current_run_buffer) >= batch_size:
            #     all_solutions_for_file.extend(current_run_buffer)
            #     current_run_buffer.clear()
            #     # sync_write_solutions(all_solutions_for_file, output_file) # Intermittent write

    # Append any remaining items from this run's buffer
    if current_run_buffer:
        all_solutions_for_file.extend(current_run_buffer)
    
    # Final write of all accumulated solutions
    if all_solutions_for_file: # Only write if there's something to write
        print(f"Result Writer: Writing {len(all_solutions_for_file)} total solutions to {output_file}.")
        sync_write_solutions(all_solutions_for_file, output_file)
    else:
        print("Result Writer: No solutions to write.")
    
    # Print comprehensive final statistics
    print("\n" + "="*60)
    print("📊 PROCESSING STATISTICS SUMMARY")
    print("="*60)
    print(f"✅ Total Processed: {processed_count:,}")
    print(f"✅ Successful: {success_count:,} ({success_rate:.1f}%)")
    print(f"❌ Errors: {error_count:,} ({(error_count/processed_count)*100 if processed_count > 0 else 0:.1f}%)")
    print(f"⏱️  Average Time: {avg_time:.2f}s per task")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print(f"🔤 Total Tokens: {total_tokens['total_tokens']:,}")
    print(f"   - Prompt: {total_tokens['prompt_tokens']:,}")
    print(f"   - Completion: {total_tokens['completion_tokens']:,}")
    
    if error_types:
        print(f"\n❌ Error Breakdown:")
        for error_type, count in error_types.items():
            percentage = (count / error_count) * 100 if error_count > 0 else 0
            print(f"   - {error_type}: {count} ({percentage:.1f}%)")
    
    print("="*60)
    print("Result Writer: Finished.")


def parse_args(config: AppConfig) -> argparse.Namespace:
    """
    Parses command-line arguments, using AppConfig for defaults.

    Args:
        config: The application configuration object.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run API calls for physics problems in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help message
    )
    parser.add_argument('--model', type=str, default=config.default_model,
                        help='Model name to use (e.g., gpt-4o). Overrides DEFAULT_MODEL in config.ini.')
    parser.add_argument('--bench_file', type=str, default=config.default_bench_file,
                        help='The input JSON file with problems. Overrides DEFAULT_BENCH_FILE in config.ini.')
    parser.add_argument('--target_dir', type=str, default=config.default_target_dir,
                        help='The directory to save output files. Overrides DEFAULT_TARGET_DIR in config.ini.')
    parser.add_argument('--num_consumers', type=int, default=config.num_consumers,
                        help='Number of consumer threads. Overrides NUM_CONSUMERS in config.ini.')
    parser.add_argument('--chat_timeout', type=float, default=config.chat_timeout,
                        help='API call timeout in seconds. Overrides CHAT_TIMEOUT in config.ini.')
    parser.add_argument('--repeat_times', type=int, default=config.repeat_times,
                        help='Number of times to process each problem. Overrides REPEAT_TIMES in config.ini.')
    parser.add_argument('--max_retries', type=int, default=config.max_retries,
                        help='Max retries for a failed API call. Overrides MAX_RETRIES in config.ini.')
    parser.add_argument('--config_file', type=str, default=str(current_dir / CONFIG_FILE_NAME),
                        help=f'Path to the configuration file (default: {current_dir / CONFIG_FILE_NAME}).')
    
    args = parser.parse_args()

    # Post-parsing validation for critical fields
    if not args.model:
        parser.error("Model name is required. Set via --model or DEFAULT_MODEL in config.ini.")
    if not args.bench_file:
        parser.error("Benchmark file path is required. Set via --bench_file or DEFAULT_BENCH_FILE in config.ini.")
    if not args.target_dir:
        parser.error("Target directory is required. Set via --target_dir or DEFAULT_TARGET_DIR in config.ini.")    
    if not config.api_key:
        parser.error("API_KEY is missing in config.ini. Please add it to the [API] section.")

    args.config_file_path = Path(args.config_file)
    if not args.config_file_path.is_absolute():
        args.config_file_path = (current_dir / args.config_file_path).resolve()
    if args.bench_file: # Check if it was provided
        args.bench_file_path = Path(args.bench_file)
        if not args.bench_file_path.is_absolute():
            args.bench_file_path = (current_dir / args.bench_file_path).resolve()
    if args.target_dir: # Check if it was provided
        args.target_dir_path = Path(args.target_dir)
        if not args.target_dir_path.is_absolute():
            args.target_dir_path = (current_dir / args.target_dir_path).resolve()
    

    return args


def check_existing_solutions(output_file: Path) -> Dict[str, set]:
    """
    Checks existing solutions in the output file and returns a dictionary of completed tasks.
    Enhanced version from problem_solve.py for duplicate detection and prevention.

    Args:
        output_file: Path to the JSON file containing existing solutions.

    Returns:
        Dictionary mapping model names to sets of completed problem IDs.
    """
    completed_tasks = {}
    
    if not output_file.exists():
        return completed_tasks
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return completed_tasks
            
            solutions = json.loads(content)
            
        for solution in solutions:
            problem_id = solution.get("id")
            model = solution.get("model")
            answer = solution.get("answer", "")
            
            # Only consider tasks with non-empty answers as completed
            # This prevents re-processing failed tasks that should be retried
            if problem_id is not None and model and answer.strip():
                if model not in completed_tasks:
                    completed_tasks[model] = set()
                completed_tasks[model].add(problem_id)
                
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  Warning: Could not read existing solutions from {output_file}: {e}")
        return {}
    
    return completed_tasks


def main() -> None:
    """
    Main function to set up and run the producer-consumer system.
    """
    global APP_CONFIG, task_queue, result_queue # Allow modification of global queue variables

    # Load configuration first, possibly overridden by --config_file arg if we parse it early
    # Here, we'll use the default config file name or one specified if we enhance parse_args for it.
    
    # Initial minimal parsing for config_file path
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config_file', type=str, default=CONFIG_FILE_NAME)
    cli_args, _ = temp_parser.parse_known_args()
    
    config_file_to_load = Path(cli_args.config_file)
    if not config_file_to_load.is_absolute():
        config_file_to_load = (current_dir / config_file_to_load).resolve()
    
    APP_CONFIG = load_config(config_file_to_load)
    
    # Now parse all arguments with config providing defaults
    args = parse_args(APP_CONFIG) # Pass loaded APP_CONFIG for defaults

    # Initialize globals in api_run.py module
    initialize_globals_from_config(
        APP_CONFIG.openai_o_model_keywords
    )

    # Update dynamic global constants based on final config/args
    # MAX_RETRIES and MAX_TASKS (queue size) are now sourced from args/APP_CONFIG
    effective_max_retries = args.max_retries
    effective_max_task_queue_size = APP_CONFIG.max_task_queue_size # From config only for now

    # Initialize queues with configured size
    task_queue = queue.Queue(maxsize=effective_max_task_queue_size)
    result_queue = queue.Queue() # Result queue typically doesn't need strict max size or can be larger

    args.target_dir_path.mkdir(parents=True, exist_ok=True)
    output_file = get_output_file(args.target_dir_path, args.model)
    
    problems: List[Dict[str, Any]] = read_problems(args.bench_file_path)
    if not problems:
        print(f"No problems loaded from {args.bench_file_path}. Exiting.")
        return
    
    # sampled_indices = [4, 69, 50, 37, 8, 86, 88, 45, 43, 72, 30, 19, 12, 7, 46, 60, 92, 58, 78, 40, 63, 29, 15, 87, 77, 44, 35, 94, 66, 84, 59, 64]
    # sampled_problems = [problems[i] for i in sampled_indices]
    sampled_problems = problems

    print(f"Total problems to process: {len(sampled_problems)} (each repeated {args.repeat_times} times)")
    print(f"Using model: {args.model}")
    print(f"Output will be saved to: {output_file}")
    print(f"Number of consumer workers: {args.num_consumers}")
    print(f"API call timeout: {args.chat_timeout}s")
    print(f"Max retries per task: {effective_max_retries}")
    print(f"Task queue size: {effective_max_task_queue_size}")

    # Check for existing solutions to avoid duplicates
    existing_solutions = check_existing_solutions(output_file)
    if existing_solutions:
        print(f"Found existing solutions for model(s): {', '.join(existing_solutions.keys())}")
        for model, problem_ids in existing_solutions.items():
            print(f"  Model {model}: {len(problem_ids)} completed problem(s).")

        # Optionally: remove already completed problems from the workload
        if args.skip_existing:
            print("Skipping existing problems in this run.")
            sampled_problems = [p for p in sampled_problems if p.get("id") not in existing_solutions.get(args.model, set())]

    total_tasks_to_produce = len(sampled_problems) * args.repeat_times

    # Start the producer thread
    producer_thread = threading.Thread(
        target=producer,
        args=(sampled_problems, args.model, args.repeat_times, output_file),
        daemon=True 
    )
    producer_thread.start()

    # Start consumer threads
    consumer_threads: List[threading.Thread] = []
    if not APP_CONFIG.api_key or not APP_CONFIG.base_url: # Should be caught by parse_args or earlier
        print("Error: API Key or Base URL is not configured. Cannot start consumers.")
        return

    for i in range(args.num_consumers):
        thread = threading.Thread(
            target=run_consumer_loop,
            args=(APP_CONFIG.api_key, APP_CONFIG.base_url, args.chat_timeout, effective_max_retries),
            daemon=True,
            name=f"Consumer-{i+1}"
        )
        thread.start()
        consumer_threads.append(thread)

    # Start the result writer thread
    writer_thread = threading.Thread(
        target=result_writer,
        args=(output_file, total_tasks_to_produce),
        daemon=True,
        name="ResultWriter"
    )
    writer_thread.start()

    # Wait for producer to finish (all tasks enqueued)
    producer_thread.join()

    # Signal consumers to stop by putting None sentinels in task_queue
    for _ in range(args.num_consumers):
        task_queue.put(None)
    
    print("All tasks enqueued. Waiting for consumers to process...")
    # Wait for all tasks in task_queue to be processed by consumers
    task_queue.join() # Waits until task_done() has been called for all items, including sentinels
    
    print("All tasks processed by consumers.")

    # Signal result_writer to stop
    result_queue.put(None)
    
    # Wait for writer thread to finish
    writer_thread.join()

    print("Joining consumer threads...")
    for t_idx, t in enumerate(consumer_threads):
       t.join(timeout=10) # Add timeout to prevent indefinite hanging
       if t.is_alive():
           print(f"Warning: Consumer thread {t.name} did not terminate cleanly.")
    print("Processing complete.")


if __name__ == "__main__":
    print("Starting parallel API processing...")
    main()
