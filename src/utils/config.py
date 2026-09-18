import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration file into a Python dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """
    Saves configuration snapshot for reproducibility.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
