"""
Configuration Manager for Tiny-Seq2Seq.
Loads YAML configurations with fallback defaults and environment overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


def load_config(config_path: str = "configs/base_config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file and returns a nested dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
