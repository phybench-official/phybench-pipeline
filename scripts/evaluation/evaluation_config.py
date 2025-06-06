import configparser
from typing import List, Optional
from pathlib import Path

CONFIG_FILE_NAME = "config.ini"


class EvaluationConfig:
    """Holds grading configuration."""

    def __init__(self):
        self.default_gt_file: Optional[str] = None
        self.default_gen_file: Optional[str] = None
        self.default_output_dir: Optional[str] = None
        self.default_scoring_params: List[int] = [60, 100]
        self.num_processes: int = 0  # 0 means auto-detect
        self.log_file: str = "evaluation_logs.txt"


def load_evaluation_config(config_file_path: Path = Path(CONFIG_FILE_NAME)) -> EvaluationConfig:
    """Loads grading configuration from an INI file."""
    parser = configparser.ConfigParser()
    config = EvaluationConfig()

    if not config_file_path.is_file():
        print(
            f"Warning: Configuration file '{config_file_path}' not found. Using default values."
        )
        return config
    
    parser.read(config_file_path, encoding="utf-8")

    if "EVALUATION" in parser:
        config.default_gt_file = parser["EVALUATION"].get("DEFAULT_GT_FILE")
        config.default_gen_file = parser["EVALUATION"].get("DEFAULT_GEN_FILE") 
        config.default_output_dir = parser["EVALUATION"].get("DEFAULT_OUTPUT_DIR")
        
        scoring_params_str = parser["EVALUATION"].get("DEFAULT_SCORING_PARAMS", "60,100")
        try:
            config.default_scoring_params = [int(x.strip()) for x in scoring_params_str.split(",")]
        except ValueError:
            print(f"Warning: Invalid scoring params '{scoring_params_str}', using default [60, 100]")
            config.default_scoring_params = [60, 100]
            
        config.num_processes = parser["EVALUATION"].getint("NUM_PROCESSES", 0)
        config.log_file = parser["EVALUATION"].get("LOG_FILE", "evaluation_logs.txt")

    return config
