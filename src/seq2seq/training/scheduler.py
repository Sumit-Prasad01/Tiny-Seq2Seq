"""
Learning Rate Scheduler for Tiny-Seq2Seq.
Implements linear warmup followed by cosine annealing decay to min_lr_ratio.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """
    Returns a learning rate scheduler with linear warmup followed by cosine decay:
    - Step 0 -> warmup_steps: Linear ramp from 0 to 1.0 * peak_lr
    - Step warmup_steps -> total_steps: Cosine decay from 1.0 to min_lr_ratio
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio
        return decayed

    return LambdaLR(optimizer, lr_lambda)
