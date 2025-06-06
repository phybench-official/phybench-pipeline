import json
import time
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from tqdm import tqdm
from typing import List, Dict, Any, Optional

_NORMALIZED_OPENAI_O_MODELS: Optional[set[str]] = None


def initialize_globals_from_config(openai_o_model_keywords: List[str]) -> None:
    """Initializes global settings from configuration."""
    global _NORMALIZED_OPENAI_O_MODELS

    _NORMALIZED_OPENAI_O_MODELS = {
        kw.lower().replace(" ", "").replace("-", "") for kw in openai_o_model_keywords
    }


def create_async_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Creates an asynchronous OpenAI client."""
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def read_problems(filename: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON file containing physics problems.
    Each problem should contain at least "id" and "content" fields.

    Args:
        filename: The path to the JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents a problem.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' is not a valid JSON file.")
        return []

    return data


def extract_boxed_answer(solution_text: str) -> str:
    """
    Extracts the content within the first \\boxed{...} structure from model output text (supports nested braces).
    Returns empty string if not found.

    Args:
        solution_text: The text containing the solution.

    Returns:
        The extracted content within the first \\boxed{...}, or an empty string if not found.
    """
    start_marker = r"\boxed{"
    start_index = solution_text.find(start_marker)
    if start_index == -1:
        return ""
    index = start_index + len(start_marker)
    brace_level = 1
    content_chars: List[str] = []

    while index < len(solution_text) and brace_level > 0:
        char = solution_text[index]
        if char == "{":
            brace_level += 1
        elif char == "}":
            brace_level -= 1

        if brace_level > 0:
            content_chars.append(char)
        index += 1

    return "".join(content_chars).strip()


file_lock = asyncio.Lock()


async def write_solution(solution: Dict[str, Any], output_filename: str) -> bool:
    """
    Asynchronously writes a single solution to JSON file with backup/recovery mechanism.

    Args:
        solution: The solution dictionary to write.
        output_filename: The path to the output JSON file.

    Returns:
        True if write was successful, False otherwise.
    """
    max_retries = 3

    for attempt in range(max_retries):
        async with file_lock:
            try:
                existing_solutions = []
                if Path(output_filename).exists():
                    with open(output_filename, "r", encoding="utf-8") as f:
                        existing_solutions = json.load(f)

                existing_solutions.append(solution)

                backup_filename = f"{output_filename}.backup"
                if Path(output_filename).exists():
                    import shutil

                    shutil.copy2(output_filename, backup_filename)

                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(existing_solutions, f, indent=2, ensure_ascii=False)

                if Path(backup_filename).exists():
                    Path(backup_filename).unlink()

                return True

            except Exception as e:
                print(f"Attempt {attempt + 1} failed to write solution: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (2**attempt))
                else:
                    backup_filename = f"{output_filename}.backup"
                    if Path(backup_filename).exists():
                        import shutil

                        shutil.copy2(backup_filename, output_filename)
                        print(f"Restored from backup: {backup_filename}")

    return False


def _is_openai_o_model(model_name: str) -> bool:
    """
    Checks if the model is an OpenAI 'o' series model that might benefit from 'reasoning_effort'.
    Requires `initialize_globals_from_config` to be called first.
    """
    if _NORMALIZED_OPENAI_O_MODELS is None:
        return False
    normalized_model_name = model_name.lower().replace(" ", "").replace("-", "")
    return any(kw in normalized_model_name for kw in _NORMALIZED_OPENAI_O_MODELS)


async def generate_solution_data(
    async_client_instance: AsyncOpenAI,
    problem: Dict[str, Any],
    model: str,
    repeat_idx: Optional[int],
    timeout: Optional[float] = 1200.0,
) -> Dict[str, Any]:
    """
    Generates a solution for a given problem using the specified model with non-streaming API.
    Enhanced version with physics-specific prompt, model-specific parameters, thinking support,
    robust error handling, token tracking, and timestamp functionality.

    Args:
        async_client_instance: An active AsyncOpenAI client instance.
        problem: The problem dictionary, containing "id" and "content" or "translatedContent".
        model: The name of the model to use.
        repeat_idx: The repetition index for this attempt (if any).
        timeout: Optional timeout in seconds for the API call (default 1200 for non-streaming).

    Returns:
        A dictionary containing the solution details, including any errors encountered.
    """
    enhanced_prompt = """You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer. 
Your final answer must be enclosed strictly within a single \\boxed{} command. 
The final answer must be a single, fully simplified, and directly parseable LaTeX expression. 
Do NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression. 
Use standard LaTeX conventions rigorously."""

    question_text: str = problem.get("translatedContent", problem.get("content", ""))
    if not question_text:
        return {
            "id": problem.get("id", "N/A"),
            "model": model,
            "solution": "Error: No question content found",
            "boxed_answer": "",
            "timestamp": time.time(),
            "time_taken": 0.0,
            "repeat_index": repeat_idx,
            "error_message": "Missing question content",
        }

    full_prompt = f"{enhanced_prompt}\nQuestion: {question_text}\n\nPlease provide the solution in LaTeX format, ensuring that the final boxed answer is clear and concise."

    start_time = time.time()

    try:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "timeout": timeout,
        }

        if _is_openai_o_model(model):
            params["reasoning_effort"] = "high"

        response: ChatCompletion = await async_client_instance.chat.completions.create(
            **params
        )

        solution_content: str = response.choices[0].message.content or ""
        boxed_answer = extract_boxed_answer(solution_content)

        time_taken = time.time() - start_time

        usage: Optional[CompletionUsage] = response.usage

        result = {
            "id": problem.get("id", "N/A"),
            "model": model,
            "solution": solution_content,
            "boxed_answer": boxed_answer,
            "timestamp": time.time(),
            "time_taken": time_taken,
            "repeat_index": repeat_idx,
        }

        if usage:
            result.update(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )

        return result

    except Exception as e:
        time_taken = time.time() - start_time
        error_msg = f"Error generating solution: {str(e)}"

        return {
            "id": problem.get("id", "N/A"),
            "model": model,
            "solution": error_msg,
            "boxed_answer": "",
            "timestamp": time.time(),
            "time_taken": time_taken,
            "repeat_index": repeat_idx,
            "error_message": str(e),
        }


async def process_problem(
    async_client_instance: AsyncOpenAI,
    problem: Dict[str, Any],
    model: str,
    output_filename: str,
    pbar: Optional[tqdm] = None,
    repeat_idx: Optional[int] = None,
    api_timeout: Optional[float] = 1200.0,
) -> Dict[str, Any]:
    """
    Processes a single problem, calls model to generate solution, and writes to file.
    Manages progress display (console or tqdm) and file writing.

    Args:
        async_client_instance: An active AsyncOpenAI client instance.
        problem: The problem dictionary.
        model: The model name.
        output_filename: Path to the output JSON file.
        pbar: Optional tqdm progress bar instance.
        repeat_idx: Optional repetition index.
        api_timeout: Optional timeout for the API call.

    Returns:
        The solution dictionary that was generated and written.
    """
    problem_id = problem.get("id", "N/A")
    status_msg_key = f"problem {problem_id}" + (
        f" (attempt {repeat_idx + 1})" if repeat_idx is not None else ""
    )
    status_msg = f"Using {model} for {status_msg_key}..."

    if pbar:
        pbar.set_description(status_msg)
    else:
        print(status_msg)

    solution_data = await generate_solution_data(
        async_client_instance, problem, model, repeat_idx, timeout=api_timeout
    )

    elapsed_time = solution_data.get("time_taken", 0.0)

    solution_content = solution_data.get("solution")
    is_error = isinstance(solution_content, str) and solution_content.startswith(
        "Error"
    )

    if is_error:
        status_suffix = f"❌ Error after {elapsed_time:.1f}s"
    else:
        status_suffix = f"✅ Completed in {elapsed_time:.1f}s"

    final_status = f"{status_msg} {status_suffix}"

    if pbar:
        pbar.set_description(final_status)
        pbar.update(1)
    else:
        print(final_status)

    write_success = await write_solution(solution_data, output_filename)
    if not write_success:
        print(f"Warning: Failed to write solution for problem {problem_id}")

    return solution_data
