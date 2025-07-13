from pydantic import Field, BaseModel

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

class APICallerSettings(BaseModel):
    paths: APICallerPathsSettings
    execution: APICallerExecutionSettings

# --- Evaluation Settings ---

class EvaluationPathsSettings(BaseModel):
    gt_dir: str = "data/ground_truth"
    gt_file: str = "{api_caller_input_file}"
    model_answers_dir: str = "data/model_solutions"
    model_answers_file: str = "{api_caller_output_file}"
    output_dir: str = "data/evaluation_results"
    output_file: str = "{api_caller_input_file}"

class EvaluationScoringSettings(BaseModel):
    initial_score: int = 60
    scoring_slope: int = 100

class EvaluationExecutionSettings(BaseModel):
    num_processes: int = 0 # 0 for auto-detect

class EvaluationSettings(BaseModel):
    paths: EvaluationPathsSettings
    scoring: EvaluationScoringSettings
    execution: EvaluationExecutionSettings

# --- Main Settings ---

class AppSettings(BaseModel):
    providers: list[ProviderSettings]
    logging: LoggingSettings
    api_caller: APICallerSettings
    evaluation: EvaluationSettings
