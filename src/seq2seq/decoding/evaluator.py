"""
Evaluation Engine for Tiny-Seq2Seq.
Calculates SacreBLEU on detokenized text, evaluates sentence length degradation
(Short < 15, Medium 15-30, Long > 30 tokens), and generates comparative translation tables.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sacrebleu

from src.seq2seq.data.batcher import Batch, BucketBatcher
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.decoding.beam_search import BeamSearchDecoder
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


class Seq2SeqEvaluator:
    """Evaluates Seq2Seq models using SacreBLEU and sentence length bucketing."""

    def __init__(
        self,
        decoder: BeamSearchDecoder,
        tokenizer: TokenizerManager,
        tracker: Optional[MLflowTracker] = None,
        plots_dir: str = "artifacts/plots",
    ) -> None:
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def evaluate_dataset(
        self,
        dataset: ParallelBinaryDataset,
        beam_size: int = 5,
        token_budget: int = 2000,
        max_sentences: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Translates a dataset with beam search and computes:
        - Overall SacreBLEU
        - Bucketed BLEU (Short < 15, Medium 15-30, Long > 30)
        - Qualitative comparison samples
        """
        logger.info(f"Running evaluation with beam_size={beam_size} on {len(dataset)} sentences...")
        batcher_cls = CppBucketBatcher if is_cpp_batcher_available() else BucketBatcher
        batcher = batcher_cls(
            dataset=dataset,
            token_budget=token_budget,
            shuffle=False,  # Keep deterministic order
        )

        sources: List[str] = []
        references: List[str] = []
        hypotheses: List[str] = []
        source_lens: List[int] = []

        batches = batcher.plan_batches(seed=42)
        total_eval = 0

        for b_indices in batches:
            if max_sentences and total_eval >= max_sentences:
                break

            batch = batcher.collate_batch(b_indices).to(self.decoder.model.fc_out.weight.device)
            decoded_ids = self.decoder.decode_batch(
                batch.src_ids, batch.src_lens, beam_size=beam_size
            )

            for i in range(len(b_indices)):
                idx = b_indices[i]
                s_raw, t_raw = dataset[idx]
                s_text = self.tokenizer.decode(s_raw.tolist())
                t_text = self.tokenizer.decode(t_raw.tolist())
                pred_text = self.tokenizer.decode(decoded_ids[i])

                sources.append(s_text)
                references.append(t_text)
                hypotheses.append(pred_text)
                source_lens.append(len(s_raw))
                total_eval += 1

                if max_sentences and total_eval >= max_sentences:
                    break

        # Compute overall SacreBLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        overall_bleu = bleu.score

        # Sentence Length Bucketing (Paper's Figure 3):
        # Short (< 15 tokens), Medium (15-30 tokens), Long (> 30 tokens)
        buckets = {
            "short (<15)": {"refs": [], "hyps": []},
            "medium (15-30)": {"refs": [], "hyps": []},
            "long (>30)": {"refs": [], "hyps": []},
        }

        for s_len, ref, hyp in zip(source_lens, references, hypotheses):
            if s_len < 15:
                b_name = "short (<15)"
            elif 15 <= s_len <= 30:
                b_name = "medium (15-30)"
            else:
                b_name = "long (>30)"

            buckets[b_name]["refs"].append(ref)
            buckets[b_name]["hyps"].append(hyp)

        bucket_scores = {}
        for b_name, b_data in buckets.items():
            if b_data["refs"]:
                b_bleu = sacrebleu.corpus_bleu(b_data["hyps"], [b_data["refs"]]).score
                bucket_scores[b_name] = round(b_bleu, 2)
            else:
                bucket_scores[b_name] = 0.0

        logger.info(f"Evaluation Complete | Overall BLEU: {overall_bleu:.2f}")
        for b_name, score in bucket_scores.items():
            logger.info(f"  Bucket {b_name}: BLEU {score:.2f} ({len(buckets[b_name]['refs'])} sentences)")

        # Generate plot
        plot_path = self.plot_length_vs_bleu(bucket_scores, beam_size=beam_size)

        # Build qualitative samples
        qualitative = []
        for i in range(min(10, len(sources))):
            qualitative.append({
                "source": sources[i],
                "target": references[i],
                "prediction": hypotheses[i],
                "beam_size": beam_size,
            })

        results = {
            "overall_bleu": round(overall_bleu, 2),
            "bucket_bleu": bucket_scores,
            "total_sentences": len(sources),
            "beam_size": beam_size,
            "plot_path": plot_path,
            "samples": qualitative,
        }

        # Log to MLflow
        if self.tracker is not None:
            self.tracker.log_epoch_metrics(
                epoch=beam_size,
                metrics={
                    f"bleu/overall_beam_{beam_size}": overall_bleu,
                    f"bleu/short_beam_{beam_size}": bucket_scores.get("short (<15)", 0.0),
                    f"bleu/medium_beam_{beam_size}": bucket_scores.get("medium (15-30)", 0.0),
                    f"bleu/long_beam_{beam_size}": bucket_scores.get("long (>30)", 0.0),
                },
            )
            if plot_path and os.path.exists(plot_path):
                self.tracker.log_artifact(plot_path, artifact_path="evaluation")
            self.tracker.log_translation_samples(qualitative, epoch=beam_size, artifact_path="evaluation")

        return results

    def plot_length_vs_bleu(
        self,
        bucket_scores: Dict[str, float],
        beam_size: int = 5,
        save_name: Optional[str] = None,
    ) -> str:
        """Plots BLEU score as a function of source sentence length bins."""
        if save_name is None:
            save_name = f"bleu_vs_sentence_length_beam_{beam_size}.png"
        save_path = os.path.join(self.plots_dir, save_name)

        categories = list(bucket_scores.keys())
        scores = list(bucket_scores.values())

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(categories, scores, color=["#4c72b0", "#55a868", "#c44e52"], width=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("SacreBLEU Score", fontsize=12)
        ax.set_xlabel("Sentence Length Category", fontsize=12)
        ax.set_title(
            f"BLEU Score by Sentence Length (Beam Size {beam_size})\nTesting Fixed Bottleneck Degradation",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
