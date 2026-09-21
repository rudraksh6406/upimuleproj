"""YAML Configuration Loader for MuleGuard."""
import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns root directory of the repository."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """Loads a YAML configuration file from the config/ directory.
    
    Args:
        config_name: Filename (e.g. 'model_config.yaml' or 'model_config')
    
    Returns:
        Dict representation of the YAML config.
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name = f"{config_name}.yaml"
        
    config_path = get_project_root() / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    """Loads all four standard YAML configs."""
    return {
        "model": load_config("model_config.yaml"),
        "data": load_config("data_config.yaml"),
        "guardian": load_config("guardian_config.yaml"),
        "dashboard": load_config("dashboard_config.yaml"),
    }
