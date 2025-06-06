#!/usr/bin/env python3
"""
Entry point for the grading module when run as a package.
Usage: python -m scripts.grading [arguments]
"""

import sys
from .main import main_cli, main, load_evaluation_config

def main_entry():
    """Main entry point for the grading package."""
    # Use CLI if command line arguments provided, otherwise use config defaults for backward compatibility
    if len(sys.argv) > 1:
        main_cli()
    else:
        # Legacy execution using config defaults for backward compatibility
        config = load_evaluation_config()
        print("Running with config default parameters...")
        
        # Use config defaults or fallback to original hardcoded values if config is missing
        gt_file = config.gt_file or "./solutions/claude-sonnet-4-0514.json"
        gen_file = config.gen_file or "./god_answer.json"
        output_dir = config.output_dir or "./data_0531.json"
        scoring_params = [config.initial_score, config.scoring_slope]
        log_file = config.log_file or "logging.txt"
        
        print(f"  - Ground truth file: {gt_file}")
        print(f"  - Generated file: {gen_file}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Scoring parameters: {scoring_params}")
        print(f"  - Log file: {log_file}")
        
        main(gt_file, gen_file, output_dir, scoring_params, log_file)

if __name__ == "__main__":
    main_entry()
