"""
Metrics Utilities for Tiny-Seq2Seq.
Provides perplexity computation, token throughput tracking, and running statistics.
"""

import math
import time
from typing import Dict, Optional


def compute_perplexity(loss: float, max_clip: float = 100.0) -> float:
    """
    Computes perplexity = exp(loss).
    Guards against overflow by clipping the input loss.
    """
    if math.isnan(loss):
        return float("nan")
    clipped_loss = min(loss, max_clip)
    try:
        return math.exp(clipped_loss)
    except OverflowError:
        return float("inf")


class ThroughputTracker:
    """Tracks training and inference throughput in tokens per second and batches per second."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.start_time = time.perf_counter()
        self.total_tokens = 0
        self.total_batches = 0
        self.step_times = []
        self.step_tokens = []

    def update(self, num_tokens: int, num_batches: int = 1, elapsed_seconds: Optional[float] = None) -> Dict[str, float]:
        """Records a completed step and returns instant and running throughput metrics."""
        self.total_tokens += num_tokens
        self.total_batches += num_batches

        if elapsed_seconds is not None and elapsed_seconds > 0:
            self.step_times.append(elapsed_seconds)
            self.step_tokens.append(num_tokens)
            if len(self.step_times) > self.window_size:
                self.step_times.pop(0)
                self.step_tokens.pop(0)

        # Windowed calculation
        if self.step_times:
            window_time = sum(self.step_times)
            window_tokens = sum(self.step_tokens)
            instant_tokens_per_sec = window_tokens / window_time if window_time > 0 else 0.0
        else:
            instant_tokens_per_sec = 0.0

        # Cumulative calculation
        total_elapsed = time.perf_counter() - self.start_time
        avg_tokens_per_sec = self.total_tokens / total_elapsed if total_elapsed > 0 else 0.0

        return {
            "tokens_per_second": round(instant_tokens_per_sec, 2),
            "avg_tokens_per_second": round(avg_tokens_per_sec, 2),
            "total_tokens": self.total_tokens,
            "total_batches": self.total_batches,
        }
