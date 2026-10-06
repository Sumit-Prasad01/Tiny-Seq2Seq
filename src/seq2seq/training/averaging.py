"""
Checkpoint Averaging Utility for Tiny-Seq2Seq.
Implements Polyak parameter averaging across multiple epoch checkpoints
to mimic the paper's 5-model ensemble gains at zero extra inference cost.
"""

from typing import Dict, List, Optional
import copy
import torch
import torch.nn as nn

from utils.logger import logger


def average_checkpoints(checkpoint_paths: List[str], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Averages model weights across multiple checkpoints:
    theta_avg = (1 / N) * sum(theta_i)
    """
    if not checkpoint_paths:
        raise ValueError("Cannot average an empty list of checkpoints.")

    logger.info(f"Averaging {len(checkpoint_paths)} checkpoints...")
    state_dicts = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device or torch.device("cpu"))
        if "model_state_dict" in ckpt:
            state_dicts.append(ckpt["model_state_dict"])
        else:
            state_dicts.append(ckpt)

    avg_state_dict: Dict[str, torch.Tensor] = {}
    ref_keys = state_dicts[0].keys()

    for key in ref_keys:
        # Check if parameter is floating point
        is_float = state_dicts[0][key].is_floating_point()
        if is_float:
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            avg_tensor = torch.mean(stacked, dim=0)
            avg_state_dict[key] = avg_tensor.to(state_dicts[0][key].dtype)
        else:
            # For integer or non-float buffers, take first
            avg_state_dict[key] = copy.deepcopy(state_dicts[0][key])

    logger.info("Checkpoint averaging completed successfully.")
    return avg_state_dict


def load_averaged_model(model: nn.Module, checkpoint_paths: List[str]) -> nn.Module:
    """Loads averaged weights directly into a model instance."""
    avg_weights = average_checkpoints(checkpoint_paths, device=next(model.parameters()).device)
    model.load_state_dict(avg_weights)
    return model
