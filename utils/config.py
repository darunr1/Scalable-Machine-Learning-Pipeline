"""
Configuration loader utility.
Loads YAML config files and provides easy access to settings.
"""

import os
import yaml
from typing import Any, Optional


_config_cache: dict = {}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to config file. Defaults to configs/default.yaml.

    Returns:
        Dictionary of configuration values.
    """
    global _config_cache

    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    config_path = os.path.abspath(config_path)

    if config_path in _config_cache:
        return _config_cache[config_path]

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _config_cache[config_path] = config
    return config


def get_config_value(key_path: str, default: Any = None, config: Optional[dict] = None) -> Any:
    """
    Get a nested config value using dot notation.

    Args:
        key_path: Dot-separated path, e.g. 'training.test_size'.
        default: Default value if key not found.
        config: Config dict. Loads default if not provided.

    Returns:
        The config value, or default.
    """
    if config is None:
        config = load_config()

    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def resolve_path(relative_path: str) -> str:
    """Resolve a relative path against the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)
