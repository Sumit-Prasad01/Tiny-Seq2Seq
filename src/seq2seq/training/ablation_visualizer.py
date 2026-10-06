"""
Ablation Study Visualizer for Tiny-Seq2Seq.
Generates comparative figures across the 8 paper replication studies:
- Source Reversal impact
- Beam Width scaling
- Depth scaling
- Optimizer convergence
- Dropout regularisation
- Embedding Weight Tying
- Checkpoint Averaging vs Single Best
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


class AblationVisualizer:
    """Creates comparative charts for paper replication ablations."""

    def __init__(self, output_dir: str = "artifacts/plots/ablations") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_reversal_comparison(
        self,
        reversed_ppl: float,
        non_reversed_ppl: float,
        reversed_bleu: float,
        non_reversed_bleu: float,
        save_name: str = "ablation_reversal.png",
    ) -> str:
        """Plots perplexity and BLEU comparison for source reversal."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        labels = ["Non-Reversed", "Reversed (Paper)"]
        ppls = [non_reversed_ppl, reversed_ppl]
        bleus = [non_reversed_bleu, reversed_bleu]

        # Panel 1: Perplexity (Lower is better)
        bars1 = ax1.bar(labels, ppls, color=["#d95f02", "#1b9e77"], width=0.5)
        ax1.set_ylabel("Dev Perplexity (Lower is Better)", fontsize=11)
        ax1.set_title("Source Reversal Effect on Perplexity", fontsize=12, fontweight="bold")
        for bar in bars1:
            ax1.annotate(f"{bar.get_height():.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

        # Panel 2: BLEU (Higher is better)
        bars2 = ax2.bar(labels, bleus, color=["#d95f02", "#1b9e77"], width=0.5)
        ax2.set_ylabel("Dev SacreBLEU (Higher is Better)", fontsize=11)
        ax2.set_title("Source Reversal Effect on BLEU", fontsize=12, fontweight="bold")
        for bar in bars2:
            ax2.annotate(f"{bar.get_height():.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_beam_width_scaling(
        self,
        beam_sizes: List[int],
        bleu_scores: List[float],
        save_name: str = "ablation_beam_width.png",
    ) -> str:
        """Plots BLEU scaling across beam sizes K in {1, 2, 5, 12}."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(beam_sizes, bleu_scores, "o-", color="#7570b3", linewidth=2.2, markersize=8)

        for x, y in zip(beam_sizes, bleu_scores):
            ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontweight="bold")

        ax.set_xlabel("Beam Size (K)", fontsize=12)
        ax.set_ylabel("SacreBLEU Score", fontsize=12)
        ax.set_title("Beam Search Scaling (Paper Finding: K=2 yields ~80% gains)", fontsize=12, fontweight="bold")
        ax.set_xticks(beam_sizes)
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_depth_comparison(
        self,
        layers: List[int],
        ppl_scores: List[float],
        bleu_scores: List[float],
        save_name: str = "ablation_depth.png",
    ) -> str:
        """Plots perplexity and BLEU across layer counts (2, 3, 4 layers)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        labels = [f"{L} Layers" for L in layers]

        bars1 = ax1.bar(labels, ppl_scores, color="#386cb0", width=0.5)
        ax1.set_ylabel("Dev Perplexity (Lower is Better)", fontsize=11)
        ax1.set_title("LSTM Depth vs Perplexity", fontsize=12, fontweight="bold")
        for bar in bars1:
            ax1.annotate(f"{bar.get_height():.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

        bars2 = ax2.bar(labels, bleu_scores, color="#7fc97f", width=0.5)
        ax2.set_ylabel("Dev BLEU (Higher is Better)", fontsize=11)
        ax2.set_title("LSTM Depth vs BLEU Score", fontsize=12, fontweight="bold")
        for bar in bars2:
            ax2.annotate(f"{bar.get_height():.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_ensemble_averaging(
        self,
        best_bleu: float,
        avg_bleu: float,
        save_name: str = "ablation_checkpoint_averaging.png",
    ) -> str:
        """Plots Single Best Checkpoint vs Polyak Weight Averaging."""
        fig, ax = plt.subplots(figsize=(6, 4.5))
        labels = ["Single Best Checkpoint", "Polyak Averaged (Last 3)"]
        scores = [best_bleu, avg_bleu]
        bars = ax.bar(labels, scores, color=["#666666", "#e7298a"], width=0.45)

        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

        ax.set_ylabel("SacreBLEU Score", fontsize=12)
        ax.set_title("Checkpoint Averaging (Cheap Stand-In for 5-Model Ensemble)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
