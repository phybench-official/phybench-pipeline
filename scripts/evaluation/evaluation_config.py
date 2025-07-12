import configparser
from pathlib import Path
from typing import Final

CONFIG_FILE_NAME: Final[str] = "config.ini"


class EvaluationConfig:
    """Holds grading configuration."""

    def __init__(self) -> None:
        self.gt_file: str | None = None
        self.model_answers_file: str | None = None
        self.output_file: str | None = None
        self.log_file: str | None = None
        self.initial_score: int | None = None
        self.scoring_slope: int | None = None
        self.num_processes: int | None = None


def load_evaluation_config(
    config_file_path: Path = Path(CONFIG_FILE_NAME),
) -> EvaluationConfig:
    """Loads grading configuration from an INI file."""
    parser = configparser.ConfigParser()
    config = EvaluationConfig()

    if not config_file_path.is_file():
        raise FileNotFoundError(
            f"Configuration file '{config_file_path}' not found. "
            "Please create a config.ini file with the required evaluation configuration."
        )

    parser.read(config_file_path, encoding="utf-8")

    if "evaluation.paths" in parser:
        config.gt_file = parser["evaluation.paths"].get("gt_file")
        config.model_answers_file = parser["evaluation.paths"].get("model_answers_file")
        config.output_file = parser["evaluation.paths"].get("output_file")
        config.log_file = parser["evaluation.paths"].get("log_file")

    if "evaluation.scoring" in parser:
        config.initial_score = parser["evaluation.scoring"].getint("initial_score")
        config.scoring_slope = parser["evaluation.scoring"].getint("scoring_slope")

    if "evaluation.execution" in parser:
        config.num_processes = parser["evaluation.execution"].getint("num_processes")

    return config
