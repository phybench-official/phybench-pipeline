from pathlib import Path

import toml

from phybench.settings import AppSettings


def get_settings(config_file: str | Path) -> AppSettings:
    """
    Loads application settings from a specified TOML file.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config_data = toml.load(config_path)

    return AppSettings.model_validate(config_data)
