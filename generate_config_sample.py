#!/usr/bin/env python3
"""
Generate config.ini.sample from config.ini by removing sensitive information.
"""

from pathlib import Path


def generate_config_sample() -> bool:
    """Generate config.ini.sample from config.ini with API key removed."""
    config_path = Path("config.ini")
    sample_path = Path("config.ini.sample")

    if not config_path.exists():
        print(f"Error: {config_path} not found")
        return False

    # Read the original file line by line to preserve comments
    with config_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Write sample config with header and preserved content
    with sample_path.open("w", encoding="utf-8") as f:
        # Add header comment
        f.write("; Sample configuration file\n")
        f.write("; Copy this to config.ini and update with your actual values\n")
        f.write("\n")

        # Process each line to replace only the API key value
        for line in lines:
            if line.strip().startswith("api_key ="):
                f.write("api_key = your_api_key_here\n")
            else:
                f.write(line)

    print(f"Generated {sample_path}")
    return True


if __name__ == "__main__":
    generate_config_sample()
