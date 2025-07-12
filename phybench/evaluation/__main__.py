#!/usr/bin/env python3
"""
Entry point for the evaluation module when run as a package.
Usage: python -m phybench.evaluation [arguments]
"""

import sys
import warnings
from pathlib import Path

from phybench.logging_config import get_logger, setup_logging

from .main import (
    load_evaluation_config,
    main,
    main_cli,
    resolve_file_path,
    resolve_log_file_path,
    resolve_output_file_path,
)

logger = get_logger(__name__)

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

            # Setup logging early with config-based log file
            log_file_path = resolve_log_file_path(
                None, config.log_dir, config.log_file, config
            )
            setup_logging(
                log_file=log_file_path,
                log_level=config.file_level or "DEBUG",
                console_level=config.console_level or "INFO",
            )

            logger.info("Running evaluation with config parameters...")

            if not config.gt_dir or not config.gt_file:
                raise ValueError(
                    "Ground truth directory and file must be specified in config.ini"
                )
            if not config.model_answers_dir or not config.model_answers_file:
                raise ValueError(
                    "Model answers directory and file must be specified in config.ini"
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

            # Use consistent path resolution logic as CLI
            gt_file_path = resolve_file_path(
                None,  # no full path override
                config.gt_dir,
                config.gt_file,
                config,
                "ground_truth.json",
            )

            model_answers_file_path = resolve_file_path(
                None,  # no full path override
                config.model_answers_dir,
                config.model_answers_file,
                config,
                "model_answers.json",
            )

            output_file_path = resolve_output_file_path(
                None,  # no full path override
                config.output_dir,
                config.output_file,
                Path(gt_file_path).name,
                Path(model_answers_file_path).name,
                config,
            )

            log_file_path = resolve_log_file_path(
                None,  # no full path override
                config.log_dir,
                config.log_file,
                config,
            )

            # Validate required inputs
            if not Path(gt_file_path).exists():
                logger.error(f"Ground truth file not found: {gt_file_path}")
                logger.error(
                    "Please check your configuration or create the required files."
                )
                return

            if not Path(model_answers_file_path).exists():
                logger.error(f"Model answers file not found: {model_answers_file_path}")
                logger.error(
                    "Please check your configuration or create the required files."
                )
                return

            # Create output and log directories if needed
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

            scoring_params = [config.initial_score, config.scoring_slope]

            logger.info("🎯 Starting evaluation process:")
            logger.info(f"  - Ground truth file: {gt_file_path}")
            logger.info(f"  - Model answers file: {model_answers_file_path}")
            logger.info(f"  - Output file: {output_file_path}")
            logger.info(f"  - Log file: {log_file_path}")
            logger.info(f"  - Scoring parameters: {scoring_params}")

            main(
                gt_file_path,
                model_answers_file_path,
                str(output_file_path),
                scoring_params,
                log_file_path,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Configuration error: {e}")
            logger.error(
                "Please check your config.ini file or use command line arguments."
            )
            sys.exit(1)


if __name__ == "__main__":
    main_entry()
