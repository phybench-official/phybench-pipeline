import configparser
from typing import List, Optional
from pathlib import Path

CONFIG_FILE_NAME = "config.ini"

class AppConfig:
    """Holds application configuration."""
    def __init__(self):
        # API settings
        self.base_url: Optional[str] = None
        self.api_key: Optional[str] = None
        
        # Model settings
        self.default_model: Optional[str] = None
        self.openai_o_model_keywords: List[str] = []

        # Enhanced physics prompt
        self.enhanced_physics_prompt: str = """You are a physics expert. Carefully read the following question and provide a clear, step-by-step solution leading clearly to the final answer. 
Your final answer must be enclosed strictly within a single \\boxed{} command. 
The final answer must be a single, fully simplified, and directly parseable LaTeX expression. 
Do NOT include integral symbol, multiple lines, piecewise cases, summation symbols, or textual explanations inside the boxed expression. 
Use standard LaTeX conventions rigorously."""

        # Execution settings
        self.default_bench_file: Optional[str] = None
        self.default_target_dir: Optional[str] = None
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
        print(f"Warning: Configuration file '{config_file_path}' not found. Using default values.")
        print("Please create a config.ini file based on the example format.")
    else:
        parser.read(config_file_path, encoding='utf-8')

    # API section
    if "API" in parser:
        config.base_url = parser["API"].get("BASE_URL")
        config.api_key = parser["API"].get("API_KEY")
    
    # Set fallbacks
    config.base_url = config.base_url or "https://api.gpt.ge/v1"

    # SETTINGS section
    if "SETTINGS" in parser:
        config.default_model = parser["SETTINGS"].get("DEFAULT_MODEL")
        
        o_model_kw_str = parser["SETTINGS"].get("OPENAI_O_MODEL_KEYWORDS", "")
        if o_model_kw_str:
            config.openai_o_model_keywords = [kw.strip() for kw in o_model_kw_str.split(',') if kw.strip()]

    # Fallbacks for SETTINGS
    config.default_model = config.default_model or "gpt-4o"
    if not config.openai_o_model_keywords:
        config.openai_o_model_keywords = [
            "o3-mini", "o3 (high)", "o3-high", "o3-mini (high)", 
            "o4-mini", "o4-mini (high)", "o1", "o1-preview-all"
        ]

    # EXECUTION section
    if "EXECUTION" in parser:
        config.default_bench_file = parser["EXECUTION"].get("DEFAULT_BENCH_FILE")
        config.default_target_dir = parser["EXECUTION"].get("DEFAULT_TARGET_DIR")
        config.num_consumers = parser["EXECUTION"].getint("NUM_CONSUMERS", 10)
        config.chat_timeout = parser["EXECUTION"].getfloat("CHAT_TIMEOUT", 1200.0)
        config.repeat_times = parser["EXECUTION"].getint("REPEAT_TIMES", 1)
        config.max_retries = parser["EXECUTION"].getint("MAX_RETRIES", 5)
        config.max_task_queue_size = parser["EXECUTION"].getint("MAX_TASK_QUEUE_SIZE", 100)
    # Fallbacks for EXECUTION
    else: # Set programmatic defaults if EXECUTION section is missing
        config.num_consumers = 10
        config.chat_timeout = 1200.0
        config.repeat_times = 1
        config.max_retries = 5
        config.max_task_queue_size = 100
        # default_bench_file and default_target_dir can remain None if not set

    return config
