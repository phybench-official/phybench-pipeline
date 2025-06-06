import configparser
from typing import List, Optional
from pathlib import Path

CONFIG_FILE_NAME = "config.ini"


class AppConfig:
    """Holds application configuration."""

    def __init__(self):
        self.base_url: Optional[str] = None
        self.api_key: Optional[str] = None

        self.model: Optional[str] = None
        self.openai_o_model_keywords: List[str] = []

        self.user_prompt: str = """You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer. 
Your final answer must be enclosed strictly within a single \\boxed{} command. 
The final answer must be a single, fully simplified, and directly parseable LaTeX expression. 
Do NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression. 
Use standard LaTeX conventions rigorously."""

        self.bench_file: Optional[str] = None
        self.target_dir: Optional[str] = None
        self.num_consumers: int = 100
        self.chat_timeout: float = 1200.0
        self.repeat_times: int = 1
        self.max_retries: int = 5
        self.max_task_queue_size: int = 100


def load_config(config_file_path: Path = Path(CONFIG_FILE_NAME)) -> AppConfig:
    """Loads configuration from an INI file."""
    parser = configparser.ConfigParser()
    config = AppConfig()

    if not config_file_path.is_file():
        print(
            f"Warning: Configuration file '{config_file_path}' not found. Using default values."
        )
        print("Please create a config.ini file based on the example format.")
    else:
        parser.read(config_file_path, encoding="utf-8")

    if "api_caller.api" in parser:
        config.base_url = parser["api_caller.api"].get("base_url")
        config.api_key = parser["api_caller.api"].get("api_key")

    config.base_url = config.base_url or "https://api.gpt.ge/v1"

    if "api_caller.model" in parser:
        model_raw = parser["api_caller.model"].get("model")
        config.model = (
            model_raw.strip()
            if model_raw and model_raw.strip()
            else None
        )

        o_model_kw_str = parser["api_caller.model"].get("openai_o_model_keywords", "")
        if o_model_kw_str:
            config.openai_o_model_keywords = [
                kw.strip() for kw in o_model_kw_str.split(",") if kw.strip()
            ]

    config.model = config.model or "gpt-4o"
    if not config.openai_o_model_keywords:
        config.openai_o_model_keywords = [
            "o3-mini",
            "o3 (high)",
            "o3-high",
            "o3-mini (high)",
            "o4-mini",
            "o4-mini (high)",
            "o1",
            "o1-preview-all",
        ]

    if "api_caller.paths" in parser:
        config.bench_file = parser["api_caller.paths"].get("bench_file")
        config.target_dir = parser["api_caller.paths"].get("target_dir")

    if "api_caller.execution" in parser:
        config.num_consumers = parser["api_caller.execution"].getint("num_consumers", 10)
        config.chat_timeout = parser["api_caller.execution"].getfloat("chat_timeout", 1200.0)
        config.repeat_times = parser["api_caller.execution"].getint("repeat_times", 1)
        config.max_retries = parser["api_caller.execution"].getint("max_retries", 5)
        config.max_task_queue_size = parser["api_caller.execution"].getint("max_task_queue_size", 100)
    else:
        config.num_consumers = 10
        config.chat_timeout = 1200.0
        config.repeat_times = 1
        config.max_retries = 5
        config.max_task_queue_size = 100

    return config
