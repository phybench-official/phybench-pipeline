import json
import asyncio
import aiofiles
import os
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from tqdm import tqdm
import time
from typing import List, Dict, Any, Optional

# Configuration for prompts and model specifics will be passed or initialized
# Global variable for normalized o_models, to be initialized by a function
_NORMALIZED_OPENAI_O_MODELS: Optional[set[str]] = None
SYSTEM_PROMPT_CONTENT_GLOBAL: str = "" # To be set by initialize_prompts
USER_PROMPT_TEMPLATE_GLOBAL: str = "" # To be set by initialize_prompts


def initialize_globals_from_config(
    openai_o_model_keywords: List[str],
    system_prompt: str,
    user_prompt: str
) -> None:
    """Initializes global settings from configuration."""
    global _NORMALIZED_OPENAI_O_MODELS, SYSTEM_PROMPT_CONTENT_GLOBAL, USER_PROMPT_TEMPLATE_GLOBAL
    
    _NORMALIZED_OPENAI_O_MODELS = {
        kw.lower().replace(" ", "").replace("-", "") for kw in openai_o_model_keywords
    }
    SYSTEM_PROMPT_CONTENT_GLOBAL = system_prompt
    USER_PROMPT_TEMPLATE_GLOBAL = user_prompt


def create_async_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """
    Creates an asynchronous OpenAI client.
    Args:
        api_key: The API key.
        base_url: The base URL for the API.
    Returns:
        An instance of AsyncOpenAI.
    """
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
        with open(filename, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file '{filename}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Ensure it's a valid JSON file.")
        return []
    
    # # 只保留id在idList中的题目
    # idList = set([512, 309, 463, 286, 454, 320, 581, 213, 25, 467, 288, 584, 390, 351])
    # data = [problem for problem in data if problem["id"] in idList]
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
        return ""    # Position after \boxed{
    index = start_index + len(start_marker)
    brace_level = 1  # Already inside one brace
    content_chars: List[str] = []

    while index < len(solution_text) and brace_level > 0:
        char = solution_text[index]
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1

        # Collect characters only when inside braces
        if brace_level > 0:
            content_chars.append(char)
        index += 1

    return "".join(content_chars).strip()


# Global async lock to prevent concurrent file write conflicts
file_lock = asyncio.Lock()

async def write_solution(solution: Dict[str, Any], output_filename: str) -> None:
    """
    Asynchronously writes a single solution to JSON file, using file_lock to prevent concurrent write conflicts.
    Does not overwrite solutions with same id/model, keeps all results for multiple test runs.

    Args:
        solution: The solution dictionary to write.
        output_filename: The path to the output JSON file.
    """
    try:
        async with file_lock:
            # Read existing solutions
            if os.path.exists(output_filename):
                async with aiofiles.open(output_filename, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    try:
                        solutions = json.loads(content) if content else []
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse existing content in {output_filename}. Initializing as empty list.")
                        solutions = []
            else:
                solutions = []

            # Append directly, do not overwrite
            solutions.append(solution)

            # Write back to file
            async with aiofiles.open(output_filename, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(solutions, indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"Error writing solution: {e} for file {output_filename}")


def _is_openai_o_model(model_name: str) -> bool:
    """
    Checks if the model is an OpenAI 'o' series model that might benefit from 'reasoning_effort'.
    Requires `initialize_globals_from_config` to be called first.
    """
    if _NORMALIZED_OPENAI_O_MODELS is None: # This should not happen if initialize_globals_from_config is called at startup.
        print("Warning: _NORMALIZED_OPENAI_O_MODELS not initialized. 'reasoning_effort' may not be applied correctly.")
        return False
    normalized_model_name = model_name.lower().replace(" ", "").replace("-", "")
    return any(kw in normalized_model_name for kw in _NORMALIZED_OPENAI_O_MODELS)


async def generate_solution_data(
    async_client_instance: AsyncOpenAI,
    problem: Dict[str, Any],
    model: str,
    repeat_idx: Optional[int],
    timeout: Optional[float] = 1200.0
) -> Dict[str, Any]:
    """
    Generates a solution for a given problem using the specified model.
    This function handles the API call and data extraction but does not write to files
    or manage progress bars.

    Args:
        async_client_instance: An active AsyncOpenAI client instance.
        problem: The problem dictionary, containing "id" and "content" or "translatedContent".
        model: The name of the model to use.
        repeat_idx: The repetition index for this attempt (if any).
        timeout: Optional timeout in seconds for the API call.

    Returns:
        A dictionary containing the solution details, including any errors encountered.
    """
    if not USER_PROMPT_TEMPLATE_GLOBAL or ("{question_text}" not in USER_PROMPT_TEMPLATE_GLOBAL) or not SYSTEM_PROMPT_CONTENT_GLOBAL:
        # This should not happen if initialize_globals_from_config is called.
        print(f"Warning: Global prompts not initialized. Current user prompt:\n{USER_PROMPT_TEMPLATE_GLOBAL}\nCurrent system prompt:\n{SYSTEM_PROMPT_CONTENT_GLOBAL}")

    question_text: str = problem.get("translatedContent", problem.get("content", ""))
    if not question_text:
        return {
            "id": problem.get("id", "unknown_id"),
            "model": model,
            "solution": "Error: Problem content ('content' or 'translatedContent') is missing or empty.",
            "answer": "",
            "time_taken": 0.0,
            "repeat_idx": repeat_idx,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        
    prompt: str = USER_PROMPT_TEMPLATE_GLOBAL.replace('{question_text}', question_text)

    solution_text: Optional[str] = None
    elapsed_time: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error_message: Optional[str] = None

    start_time = time.time()
    try:
        extra_params: Dict[str, Any] = {}
        if _is_openai_o_model(model):
            extra_params["reasoning_effort"] = "high"        # Call model API
        response: ChatCompletion = await async_client_instance.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CONTENT_GLOBAL},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            timeout=timeout,
            **extra_params
        )

        if response.choices and response.choices[0].message:
            message_obj: ChatCompletionMessage = response.choices[0].message
            solution_text = message_obj.content
        else:
            solution_text = None 

        if solution_text is None:
            error_message = "Error: Received no valid response or empty solution text from the API."
            print(f"Problem {problem.get('id', 'N/A')} with model {model}: Empty solution_text received in API response.")
        
        # Calculate elapsed_time and tokens only on success
        elapsed_time = time.time() - start_time
        usage: Optional[CompletionUsage] = getattr(response, "usage", None)
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

    except Exception as e:
        error_message = f"Error generating solution: {type(e).__name__}: {e}"
        elapsed_time = time.time() - start_time

    final_boxed_answer = extract_boxed_answer(solution_text) if solution_text else ""
      # Determine final solution content
    final_solution_content = (
        error_message or 
        (solution_text.strip() if solution_text else 
         "Error: Unknown issue, solution text is None without explicit error message.")
    )

    return {
        "id": problem.get("id", "unknown_id"),
        "model": model,
        "solution": final_solution_content,
        "answer": final_boxed_answer,
        "time_taken": elapsed_time,
        "repeat_idx": repeat_idx,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def process_problem(
    async_client_instance: AsyncOpenAI, # Added client instance parameter
    problem: Dict[str, Any],
    model: str,
    output_filename: str,
    pbar: Optional[tqdm] = None,
    repeat_idx: Optional[int] = None,
    api_timeout: Optional[float] = 1200.0
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
    problem_id = problem.get("id", "N/A") # Safer access to problem id
    status_msg_key = f"problem {problem_id}" + (f" (attempt {repeat_idx + 1})" if repeat_idx is not None else "")
    status_msg = f"Using {model} for {status_msg_key}..."

    if pbar:
        pbar.set_description_str(status_msg, refresh=True)
    else:
        print(f"\n--- {status_msg} ---\n")

    solution_data = await generate_solution_data(
        async_client_instance, problem, model, repeat_idx, timeout=api_timeout # Pass client
    )

    elapsed_time = solution_data.get("time_taken", 0.0)
    
    # Check if the solution field indicates an error
    solution_content = solution_data.get("solution")
    # A more robust check for error: check if error_message was set internally,
    # or if the solution_content (which now is the error message if one occurred) starts with "Error"
    is_error = isinstance(solution_content, str) and solution_content.startswith("Error")


    if is_error:
        error_info = solution_content # solution_content is already the error message
        completion_msg = f"Error with {model} for {status_msg_key}: {error_info}. Took {elapsed_time:.2f}s"
        if pbar:
            # tqdm doesn't like newlines in description, keep it concise
            pbar.set_description_str(f"Error {status_msg_key}", refresh=True)
            # Optionally print full error if not using pbar or if more detail is needed
            # pbar.write(completion_msg) # Alternative to print for tqdm
        else:
            print(completion_msg)
    else:
        completion_msg = f"{model} completed {status_msg_key}. Took {elapsed_time:.2f}s"
        if pbar:
            pbar.set_description_str(completion_msg, refresh=True)
        else:
            print(f"\n--- {completion_msg} ---\n")
            if solution_content:
                 print(solution_content)
    # Write solution
    await write_solution(solution_data, output_filename)

    if pbar:
        pbar.update(1)

    return solution_data
