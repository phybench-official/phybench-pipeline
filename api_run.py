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

# Global variable for normalized o_models, to be initialized
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
        with open(filename, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file '{filename}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Ensure it's a valid JSON file.")
        return []

    # # 只保留id在idList中的题目
    # idList = set([512, 309])
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
        try:
            async with file_lock:
                # Read existing solutions
                solutions = []
                if os.path.exists(output_filename):
                    try:
                        async with aiofiles.open(output_filename, 'r', encoding='utf-8') as f:
                            content = await f.read()
                            if content.strip():
                                solutions = json.loads(content)
                    except (json.JSONDecodeError, Exception):
                        # If file is corrupted, try to restore from backup
                        backup_filename = output_filename + ".backup"
                        if os.path.exists(backup_filename):
                            try:
                                async with aiofiles.open(backup_filename, 'r', encoding='utf-8') as f:
                                    content = await f.read()
                                    solutions = json.loads(content) if content.strip() else []
                            except Exception:
                                solutions = []
                        else:
                            solutions = []

                # Check if the same ID/model already exists in the file
                # If so, replace it. Otherwise, append as a new entry.
                updated = False
                for i, existing_solution in enumerate(solutions):
                    if (existing_solution["id"] == solution["id"]
                        and existing_solution["model"] == solution["model"]):
                        solutions[i] = solution
                        updated = True
                        break
                
                if not updated:
                    solutions.append(solution)

                # Create backup before writing
                backup_filename = output_filename + ".backup"
                if os.path.exists(output_filename):
                    try:
                        async with aiofiles.open(output_filename, 'r', encoding='utf-8') as src:
                            backup_content = await src.read()
                        async with aiofiles.open(backup_filename, 'w', encoding='utf-8') as dst:
                            await dst.write(backup_content)
                    except Exception:
                        pass  # Backup failure doesn't affect main flow

                # Write the updated list back to the file
                json_content = json.dumps(solutions, indent=4, ensure_ascii=False)
                async with aiofiles.open(output_filename, 'w', encoding='utf-8') as f:
                    await f.write(json_content)
                
                # Verify write was successful
                try:
                    async with aiofiles.open(output_filename, 'r', encoding='utf-8') as f:
                        verify_content = await f.read()
                        json.loads(verify_content)  # Verify JSON format is correct
                    
                    print(f"✅ Saved: Problem {solution['id']}, Model {solution['model']}")
                    return True
                    
                except Exception:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Write verification failed, retrying {attempt + 1}/{max_retries}")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        raise

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Write failed, retrying {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(1)
            else:
                print(f"❌ Error writing solution (final failure): {e}")
                # Try to write to emergency backup file
                emergency_filename = f"{output_filename}.emergency_{int(time.time())}"
                try:
                    async with aiofiles.open(emergency_filename, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps([solution], indent=4, ensure_ascii=False))
                    print(f"🆘 Written to emergency backup: {emergency_filename}")
                except Exception:
                    print(f"💥 Emergency backup failed")
                return False
    
    return False


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
    # Use enhanced physics prompt instead of legacy prompts
    enhanced_prompt = """You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer. 
Your final answer must be enclosed strictly within a single \\boxed{} command. 
The final answer must be a single, fully simplified, and directly parseable LaTeX expression. 
Do NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression. 
Use standard LaTeX conventions rigorously."""

    question_text: str = problem.get("translatedContent", problem.get("content", ""))
    if not question_text:
        return {
            "id": problem.get("id", "unknown_id"),
            "model": model,
            "solution": "Error: Problem content ('content' or 'translatedContent') is missing or empty.",
            "answer": "",
            "time_taken": 0.0,
            "repeat_idx": repeat_idx,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    full_prompt = f"{enhanced_prompt}\nQuestion: {question_text}\n\nPlease provide the solution in LaTeX format, ensuring that the final boxed answer is clear and concise."

    start_time = time.time()
    
    try:
        # Prepare model-specific parameters
        model_name_clean = model.replace("-high", "")
        extra_params = {}
        
        if _is_openai_o_model(model):
            if model.endswith("-high"):
                extra_params["reasoning_effort"] = "high"
            
        # Enable thinking for supported models
        extra_body = {"enable_thinking": True}
        
        # Use non-streaming API call
        response = await async_client_instance.chat.completions.create(
            model=model_name_clean,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            timeout=timeout,
            extra_body=extra_body,
            **extra_params
        )

        elapsed_time = time.time() - start_time

        # Extract solution text from response
        solution_text = ""
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                solution_text = message.content

        # Extract token usage information
        usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, 'usage') and response.usage:
            usage_info = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            }

        final_boxed = extract_boxed_answer(solution_text)

        return {
            "id": problem.get("id", "unknown_id"),
            "model": model,
            "solution": solution_text.strip(),
            "answer": final_boxed,
            "time_taken": elapsed_time,
            "repeat_idx": repeat_idx,
            "tokens": usage_info or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_message = f"Error generating solution: {type(e).__name__}: {e}"
        
        return {
            "id": problem.get("id", "unknown_id"),
            "model": model,
            "solution": error_message,
            "answer": "",
            "time_taken": elapsed_time,
            "repeat_idx": repeat_idx,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
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
