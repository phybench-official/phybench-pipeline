#!/usr/bin/env python3
"""
Entry point for the evaluation module when run as a package.
Usage: python -m phybench.evaluation [arguments]
"""

import sys
import warnings
from pathlib import Path

from .main import (
    expand_template_placeholders,
    get_evaluation_output_file,
    get_file_path_with_normalization,
    load_evaluation_config,
    main,
    main_cli,
)

# Suppress the frozen runpy warning that occurs when running as module
# This warning is an expected behavior, and does not indicate an error
warnings.filterwarnings(
    "ignore", message=".*found in sys.modules.*", category=RuntimeWarning
)


def main_entry() -> None:
    """Main entry point for the grading package."""
    if len(sys.argv) > 1:
        main_cli()
    else:
        try:
            config = load_evaluation_config()
            print("Running with config parameters...")

            if not config.gt_folder or not config.gt_file:
                raise ValueError(
                    "Ground truth folder and file must be specified in config.ini"
                )
            if not config.model_answers_folder or not config.model_answers_file:
                raise ValueError(
                    "Model answers folder and file must be specified in config.ini"
                )
            if not config.output_dir or not config.output_file:
                raise ValueError(
                    "Output directory and file template must be specified in config.ini"
                )
            if config.initial_score is None:
                raise ValueError(
                    "Initial score (initial_score) must be specified in config.ini"
                )
            if config.scoring_slope is None:
                raise ValueError(
                    "Scoring slope (scoring_slope) must be specified in config.ini"
                )

            # Build file paths with template expansion (same logic as parse_args)
            gt_file_expanded = expand_template_placeholders(
                config.gt_file,
                config.api_caller_model or "",
                config.api_caller_input_file or "",
                config.api_caller_output_file or "",
            )
            gt_file_path = get_file_path_with_normalization(
                config.gt_folder, gt_file_expanded
            )

            model_answers_file_expanded = expand_template_placeholders(
                config.model_answers_file,
                config.api_caller_model or "",
                config.api_caller_input_file or "",
                config.api_caller_output_file or "",
            )
            model_answers_file_path = get_file_path_with_normalization(
                config.model_answers_folder, model_answers_file_expanded
            )

            # Generate output file path using template
            output_dir_path = Path(config.output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)

            output_file_path = get_evaluation_output_file(
                output_dir_path,
                gt_file_expanded,
                model_answers_file_expanded,
                config.output_file,
                api_caller_model=config.api_caller_model or "",
                api_caller_input_file=config.api_caller_input_file or "",
                api_caller_output_file=config.api_caller_output_file or "",
            )

            scoring_params = [config.initial_score, config.scoring_slope]
            log_file = config.log_file or "evaluation_logs.txt"

            print(f"  - Ground truth file: {gt_file_path}")
            print(f"  - Model answers file: {model_answers_file_path}")
            print(f"  - Output file: {output_file_path}")
            print(f"  - Scoring parameters: {scoring_params}")
            print(f"  - Log file: {log_file}")

            main(
                gt_file_path,
                model_answers_file_path,
                str(output_file_path),
                scoring_params,
                log_file,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Configuration error: {e}")
            print("Please check your config.ini file or use command line arguments.")
            sys.exit(1)


if __name__ == "__main__":
    main_entry()
