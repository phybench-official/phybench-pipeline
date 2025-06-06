#!/usr/bin/env python3
"""
Entry point for the grading module when run as a package.
Usage: python -m scripts.grading [arguments]
"""

import sys
from .main import main_cli, main, load_evaluation_config

def main_entry():
    """Main entry point for the grading package."""
    if len(sys.argv) > 1:
        main_cli()
    else:
        try:
            config = load_evaluation_config()
            print("Running with config parameters...")
            
            if not config.gt_file:
                raise ValueError("Ground truth file (gt_file) must be specified in config.ini")
            if not config.gen_file:
                raise ValueError("Generated file (gen_file) must be specified in config.ini")
            if not config.output_dir:
                raise ValueError("Output directory (output_dir) must be specified in config.ini")
            if config.initial_score is None:
                raise ValueError("Initial score (initial_score) must be specified in config.ini")
            if config.scoring_slope is None:
                raise ValueError("Scoring slope (scoring_slope) must be specified in config.ini")
            
            scoring_params = [config.initial_score, config.scoring_slope]
            log_file = config.log_file or "evaluation_logs.txt"
            
            print(f"  - Ground truth file: {config.gt_file}")
            print(f"  - Generated file: {config.gen_file}")
            print(f"  - Output directory: {config.output_dir}")
            print(f"  - Scoring parameters: {scoring_params}")
            print(f"  - Log file: {log_file}")
            
            main(config.gt_file, config.gen_file, config.output_dir, scoring_params, log_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Configuration error: {e}")
            print("Please check your config.ini file or use command line arguments.")
            sys.exit(1)

if __name__ == "__main__":
    main_entry()
