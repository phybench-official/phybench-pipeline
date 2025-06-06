import configparser
from typing import List, Optional
from pathlib import Path

CONFIG_FILE_NAME = "config.ini"


class EvaluationConfig:
    """Holds grading configuration."""

    def __init__(self):
        self.gt_file: Optional[str] = None
        self.gen_file: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.log_file: str = "evaluation_logs.txt"
        self.initial_score: int = 60
        self.scoring_slope: int = 100
        self.num_processes: int = 0  # 0 means auto-detect


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

    if "evaluation.paths" in parser:
        config.gt_file = parser["evaluation.paths"].get("gt_file")
        config.gen_file = parser["evaluation.paths"].get("gen_file") 
        config.output_dir = parser["evaluation.paths"].get("output_dir")
        config.log_file = parser["evaluation.paths"].get("log_file", "evaluation_logs.txt")
        
    if "evaluation.scoring" in parser:
        config.initial_score = parser["evaluation.scoring"].getint("initial_score", 60)
        config.scoring_slope = parser["evaluation.scoring"].getint("scoring_slope", 100)
            
    if "evaluation.execution" in parser:
        config.num_processes = parser["evaluation.execution"].getint("num_processes", 0)

    return config
