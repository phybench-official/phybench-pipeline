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
        self.model_list: List[str] = [] # General purpose model list

        # Prompt settings
        self.system_prompt_content: str = ""
        self.user_prompt_template: str = ""

        # Execution settings
        self.default_bench_file: Optional[str] = None
        self.default_target_dir: Optional[str] = None
        self.num_consumers: int = 100
        self.chat_timeout: float = 1200.0
        self.repeat_times: int = 1
        self.max_retries: int = 5
        self.max_task_queue_size: int = 100
        self.result_writer_batch_size: int = 10


def load_config(config_file_path: Path = Path(CONFIG_FILE_NAME)) -> AppConfig:
    """Loads configuration from an INI file."""
    parser = configparser.ConfigParser()
    config = AppConfig()

    if not config_file_path.is_file():
        print(f"Warning: Configuration file '{config_file_path}' not found. Using default values and attempting to create a template.")
        # Create a default config file if it doesn't exist to guide the user
        _create_default_config_template(config_file_path, parser)
        # Proceed with hardcoded defaults for this run if template creation failed or to ensure operation
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
        
        model_list_str = parser["SETTINGS"].get("MODEL_LIST", "")
        if model_list_str:
            config.model_list = [m.strip() for m in model_list_str.split(',') if m.strip()]

    # Fallbacks for SETTINGS
    config.default_model = config.default_model or "gpt-4o"
    if not config.openai_o_model_keywords:
        config.openai_o_model_keywords = [
            "o3-mini", "o3 (high)", "o3-high", "o3-mini (high)", 
            "o4-mini", "o4-mini (high)", "o1", "o1-preview-all"
        ]
    if not config.model_list and config.default_model:
        config.model_list = [config.default_model]


    # PROMPTS section
    if "PROMPTS" in parser:
        config.system_prompt_content = parser["PROMPTS"].get("SYSTEM_PROMPT_CONTENT", "")
        config.user_prompt_template = parser["PROMPTS"].get("USER_PROMPT_TEMPLATE", "")
    # Fallbacks for PROMPTS
    config.system_prompt_content = config.system_prompt_content or (
        "You are an expert in solving physics problems. "
        "Offer a clear step-by-step explanation. Then provide "
        "the final result enclosed in \\boxed{}."
    )
    config.user_prompt_template = config.user_prompt_template or (
        "Please read the following question "
        "and provide a step-by-step solution. Put your final answer, which must be a readable latex formula, in "
        "a structure \\boxed{}.\n\n"
        "Question: {question_text}\n\nAnswer:"
    )

    # EXECUTION section
    if "EXECUTION" in parser:
        config.default_bench_file = parser["EXECUTION"].get("DEFAULT_BENCH_FILE")
        config.default_target_dir = parser["EXECUTION"].get("DEFAULT_TARGET_DIR")
        config.num_consumers = parser["EXECUTION"].getint("NUM_CONSUMERS", 10)
        config.chat_timeout = parser["EXECUTION"].getfloat("CHAT_TIMEOUT", 1200.0)
        config.repeat_times = parser["EXECUTION"].getint("REPEAT_TIMES", 1)
        config.max_retries = parser["EXECUTION"].getint("MAX_RETRIES", 5)
        config.max_task_queue_size = parser["EXECUTION"].getint("MAX_TASK_QUEUE_SIZE", 100)
        config.result_writer_batch_size = parser["EXECUTION"].getint("RESULT_WRITER_BATCH_SIZE", 10)
    # Fallbacks for EXECUTION
    else: # Set programmatic defaults if EXECUTION section is missing
        config.num_consumers = 10
        config.chat_timeout = 1200.0
        config.repeat_times = 1
        config.max_retries = 5
        config.max_task_queue_size = 100
        config.result_writer_batch_size = 10
        # default_bench_file and default_target_dir can remain None if not set

    return config

def _create_default_config_template(config_path: Path, parser: configparser.ConfigParser):
    """Creates a template config.ini file if one doesn't exist."""
    parser["API"] = {
        "BASE_URL": "https://api.gpt.ge/v1",
        "API_KEY": "YOUR_API_KEY_HERE"
    }
    parser["SETTINGS"] = {
        "DEFAULT_MODEL": "gpt-4o",
        "OPENAI_O_MODEL_KEYWORDS": "o3-mini, o3 (high), o3-high, o3-mini (high), o4-mini, o4-mini (high), o1, o1-preview-all",
        "MODEL_LIST": "gpt-4o"
    }
    parser["PROMPTS"] = {
        "SYSTEM_PROMPT_CONTENT": "You are an expert in solving physics problems. Offer a clear step-by-step explanation. Then provide the final result enclosed in \\boxed{}.",
        "USER_PROMPT_TEMPLATE": "Please read the following question and provide a step-by-step solution. Put your final answer, which must be a readable latex formula, in a structure \\boxed{}.\n\nQuestion: {question_text}\n\nAnswer:"
    }
    parser["EXECUTION"] = {
        "DEFAULT_BENCH_FILE": "test.json",
        "DEFAULT_TARGET_DIR": "test_results",
        "NUM_CONSUMERS": "10",
        "CHAT_TIMEOUT": "1200.0",
        "REPEAT_TIMES": "1",
        "MAX_RETRIES": "5",
        "MAX_TASK_QUEUE_SIZE": "100",
        "RESULT_WRITER_BATCH_SIZE": "10"
    }
    try:
        if not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            parser.write(f)
        print(f"A default configuration template has been written to '{config_path}'.")
        print("IMPORTANT: Please edit this file to include your API_KEY and verify other settings.")
    except IOError as e:
        print(f"Error: Could not write default configuration file '{config_path}': {e}")
