#!/usr/bin/env python3
"""
Entry point for the api_caller module when run as a package.
Usage: python -m phybench.api_caller [arguments]

Supports all CLI arguments that match the config.ini structure.
"""

from .main import main

if __name__ == "__main__":
    main()
