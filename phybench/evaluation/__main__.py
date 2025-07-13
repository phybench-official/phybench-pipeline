#!/usr/bin/env python3
"""
Entry point for the evaluation module when run as a package.
Usage: python -m phybench.evaluation [arguments]
"""

from .main import main_cli

if __name__ == "__main__":
    main_cli()
