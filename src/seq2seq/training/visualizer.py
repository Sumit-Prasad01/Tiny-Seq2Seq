"""
Visualization Engine for Tiny-Seq2Seq.
Generates publication-quality plots using matplotlib/seaborn:
1. Training and Validation Loss / Perplexity Curves
2. Token Throughput and Peak VRAM Memory Profiling
3. Learning Rate Warmup & Cosine Schedule Dynamics
4. Gradient Norm Distributions and Clipping Thresholds
Uploads figures directly to MLflow and persists to disk.
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


class TrainingVisualizer:
    """Creates and logs training diagnostics and performance plots."""

    def __init__(self, output_dir: str = "artifacts/plots") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss_and_perplexity(
        self,
        train_losses: List[float],
        dev_losses: List[float],
        train_ppls: List[float],
        dev_ppls: List[float],
        save_name: str = "loss_perplexity_curve.png",
    ) -> str:
        """Dual-panel plot for loss and perplexity across training epochs."""
        epochs = list(range(1, len(train_losses) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Loss
        ax1.plot(epochs, train_losses, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs, dev_losses, "s--", label="Dev Loss", color="#ff7f0e", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss (Per-Token Cross Entropy)", fontsize=12)
        ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend(frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Panel 2: Perplexity
        ax2.plot(epochs, train_ppls, "o-", label="Train PPL", color="#2ca02c", linewidth=2)
        ax2.plot(epochs, dev_ppls, "s--", label="Dev PPL", color="#d62728", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Perplexity (exp(loss))", fontsize=12)
        ax2.set_title("Training & Validation Perplexity", fontsize=14, fontweight="bold")
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_hardware_and_throughput(
        self,
        steps: List[int],
        tokens_per_sec: List[float],
        vram_mb: List[float],
        save_name: str = "hardware_throughput_profile.png",
    ) -> str:
        """Dual-panel plot for tokens/sec and VRAM allocation over steps."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Throughput
        ax1.plot(steps, tokens_per_sec, color="#9467bd", linewidth=1.5)
        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Tokens / Second", fontsize=12)
        ax1.set_title("Training Throughput (Tokens/s)", fontsize=14, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Panel 2: VRAM Allocation
        ax2.plot(steps, vram_mb, color="#e377c2", linewidth=1.5)
        ax2.axhline(y=4096, color="r", linestyle=":", label="4 GB Hardware Limit")
        ax2.axhline(y=3300, color="orange", linestyle="--", label="3.3 GB Safe Ceiling")
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_ylabel("VRAM Allocated (MB)", fontsize=12)
        ax2.set_title("GPU VRAM Utilization", fontsize=14, fontweight="bold")
        ax2.legend(frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_lr_schedule(
        self,
        steps: List[int],
        lrs: List[float],
        save_name: str = "learning_rate_schedule.png",
    ) -> str:
        """Plots learning rate warmup and cosine annealing decay."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, lrs, color="#17becf", linewidth=2)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule (Warmup + Cosine Decay)", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_gradient_norms(
        self,
        steps: List[int],
        grad_norms: List[float],
        clip_norm: float = 5.0,
        save_name: str = "gradient_norm_distribution.png",
    ) -> str:
        """Plots gradient norm trajectory with clipping threshold line."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(steps, grad_norms, alpha=0.7, color="#8c564b", linewidth=1.2, label="Grad Norm")
        ax.axhline(y=clip_norm, color="r", linestyle="--", linewidth=1.5, label=f"Clip Threshold ({clip_norm})")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Gradient Norm", fontsize=12)
        ax.set_title("Global Gradient Norm & Clipping Threshold", fontsize=14, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
