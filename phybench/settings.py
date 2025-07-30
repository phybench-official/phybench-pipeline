from pydantic import BaseModel


# --- Provider Settings ---
class ProviderSettings(BaseModel):
    name: str
    base_url: str
    api_key: str
    models: list[str]


# --- Logging Settings ---
class LoggingSettings(BaseModel):
    log_dir: str = "logs"
    log_file: str = "{api_caller_input_file}"
    console_level: str = "INFO"
    file_level: str = "DEBUG"


# --- API Caller Settings ---
class APICallerModelSettings(BaseModel):
    model: str = ""
    openai_o_model_keywords: list[str] = []


class APICallerPathsSettings(BaseModel):
    input_dir: str = "data/ground_truth"
    input_file: str = "test.json"
    output_dir: str = "data/model_solutions"
    output_file: str = "{input_file}_{model}"


class APICallerExecutionSettings(BaseModel):
    num_consumers: int = 10
    chat_timeout: int = 1200
    repeat_count: int = 1
    max_retries: int = 5
    max_task_queue_size: int = 100


class APICallerPromptSettings(BaseModel):
    prefix: str = """You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer.
Your final answer must be enclosed strictly within a single \boxed{} command.
The final answer must be a single, fully simplified, and directly parseable LaTeX expression.
Do NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression.
Use standard LaTeX conventions rigorously."""
    suffix: str = "Please provide the solution in LaTeX format, ensuring that the final boxed answer is clear and concise."


class APICallerSettings(BaseModel):
    model: APICallerModelSettings
    paths: APICallerPathsSettings
    execution: APICallerExecutionSettings
    prompt: APICallerPromptSettings


# --- Evaluation Settings ---


class EvaluationPathsSettings(BaseModel):
    gt_dir: str = "data/ground_truth"
    gt_file: str = "{api_caller_input_file}"
    model_answers_dir: str = "data/model_solutions"
    model_answers_file: str = "{api_caller_output_file}"
    output_dir: str = "data/evaluation_results"
    output_file: str = "{api_caller_input_file}"


class EvaluationEEDSettings(BaseModel):
    initial_score: float = 60.0
    scoring_slope: float = 100.0
    insert_cost: dict[str, float] = {
        "number": 1.0,
        "symbol": 1.0,
        "operator": 1.0,
        "function": 1.0,
    }
    delete_cost: dict[str, float] = {
        "number": 1.0,
        "symbol": 1.0,
        "operator": 1.0,
        "function": 1.0,
    }
    update_cost: dict[str, float] = {
        "number": 1.0,
        "symbol": 1.0,
        "operator": 1.0,
        "function": 1.0,
    }
    change_type_cost: float = 1.0
    bar_size: float = 5.0
    discount_slope: float = 0.6
    simplify_time_limit: int = 30
    equals_time_limit: int = 10


class EvaluationExecutionSettings(BaseModel):
    num_processes: int = 0
    skip_problem_ids: list[int] = []


class EvaluationSettings(BaseModel):
    paths: EvaluationPathsSettings
    eed: EvaluationEEDSettings
    execution: EvaluationExecutionSettings


# --- Main Settings ---


class AppSettings(BaseModel):
    providers: list[ProviderSettings]
    logging: LoggingSettings
    api_caller: APICallerSettings
    evaluation: EvaluationSettings
