import configparser
from pathlib import Path
from typing import Final

CONFIG_FILE_NAME: Final[str] = "config.ini"


class ApiConfig:
    """Holds application configuration."""

    def __init__(self) -> None:
        self.base_url: str | None = None
        self.api_key: str | None = None

        self.model: str | None = None
        self.openai_o_model_keywords: list[str] = []

        self.user_prompt: str = """You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer.
Your final answer must be enclosed strictly within a single \\boxed{} command.
The final answer must be a single, fully simplified, and directly parseable LaTeX expression.
Do NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression.
Use standard LaTeX conventions rigorously."""

        self.input_file: str | None = None
        self.output_file: str | None = None
        self.num_consumers: int | None = None
        self.chat_timeout: float | None = None
        self.repeat_count: int | None = None
        self.max_retries: int | None = None
        self.max_task_queue_size: int | None = None


def load_api_config(config_file_path: Path = Path(CONFIG_FILE_NAME)) -> ApiConfig:
    """Loads configuration from an INI file."""
    parser = configparser.ConfigParser()
    config = ApiConfig()

    if not config_file_path.is_file():
        raise FileNotFoundError(
            f"Configuration file '{config_file_path}' not found. "
            "Please create a config.ini file with the required configuration."
        )

    parser.read(config_file_path, encoding="utf-8")

    if "api_caller.api" in parser:
        config.base_url = parser["api_caller.api"].get("base_url")
        config.api_key = parser["api_caller.api"].get("api_key")

    if "api_caller.model" in parser:
        model_raw = parser["api_caller.model"].get("model")
        config.model = model_raw.strip() if model_raw and model_raw.strip() else None

        o_model_kw_str = parser["api_caller.model"].get("openai_o_model_keywords", "")
        if o_model_kw_str:
            config.openai_o_model_keywords = [
                kw.strip() for kw in o_model_kw_str.split(",") if kw.strip()
            ]

    if "api_caller.paths" in parser:
        config.input_file = parser["api_caller.paths"].get("input_file")
        config.output_file = parser["api_caller.paths"].get("output_dir")

    if "api_caller.execution" in parser:
        config.num_consumers = parser["api_caller.execution"].getint("num_consumers")
        config.chat_timeout = parser["api_caller.execution"].getfloat("chat_timeout")
        config.repeat_count = parser["api_caller.execution"].getint("repeat_count")
        config.max_retries = parser["api_caller.execution"].getint("max_retries")
        config.max_task_queue_size = parser["api_caller.execution"].getint(
            "max_task_queue_size"
        )

    return config
