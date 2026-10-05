"""
Helper Utilities for Tiny-Seq2Seq.
Provides seeding, hardware profiling, parameter counting, and general formatting tools.
"""

import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def get_gpu_memory_mb() -> Dict[str, float]:
    """Returns allocated, reserved, and peak GPU memory in Megabytes."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "peak_mb": round(peak, 2),
    }


def reset_peak_gpu_memory() -> None:
    """Resets the peak GPU memory statistics tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total and trainable parameters in a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def format_seconds(seconds: float) -> str:
    """Formats duration in seconds to a human-readable hh:mm:ss string."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
