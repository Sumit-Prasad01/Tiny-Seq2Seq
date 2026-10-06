"""
Evaluation CLI for Tiny-Seq2Seq.
Runs vectorized beam search decoding across beam sizes (1, 2, 5, 12),
computes overall SacreBLEU, evaluates sentence length degradation,
and records telemetry and comparative tables to MLflow.
"""

import argparse
import os
import sys
from typing import List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tabulate import tabulate

from src.seq2seq.config import load_config
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.decoding.beam_search import BeamSearchDecoder
from src.seq2seq.decoding.evaluator import Seq2SeqEvaluator
from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.tracker import MLflowTracker
from utils.helpers import get_device, set_seed
from utils.logger import logger


def evaluate(
    checkpoint_path: str = "checkpoints/checkpoint_best.pt",
    config_path: str = "configs/base_config.yaml",
    split: str = "dev",
    beam_sizes: List[int] = None,
    max_sentences: int = None,
) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    dec_cfg = config.get("decoding", {})
    mlflow_cfg = config["mlflow"]

    if beam_sizes is None:
        beam_sizes = dec_cfg.get("beam_sizes", [1, 2, 5, 12])

    device = get_device()
    logger.info(f"Initiating evaluation on device: {device}")

    # Paths
    processed_dir = data_cfg["processed_dir"]
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]

    src_bin = os.path.join(processed_dir, f"{split}.{src_lang}.bin")
    src_idx = os.path.join(processed_dir, f"{split}.{src_lang}.idx")
    tgt_bin = os.path.join(processed_dir, f"{split}.{tgt_lang}.bin")
    tgt_idx = os.path.join(processed_dir, f"{split}.{tgt_lang}.idx")

    if not os.path.exists(src_bin):
        raise FileNotFoundError(f"Split binary file missing: {src_bin}")

    dataset = ParallelBinaryDataset(src_bin, src_idx, tgt_bin, tgt_idx)
    logger.info(f"Loaded {len(dataset)} sentence pairs from split '{split}'.")

    # Load tokenizer
    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".model")]
    if not tok_files:
        raise FileNotFoundError(f"Tokenizer model not found in {tokenizer_dir}")
    tokenizer = TokenizerManager(os.path.join(tokenizer_dir, tok_files[0]))

    # Initialize model
    model = Seq2SeqModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        tie_weights=model_cfg["tie_weights"],
    ).to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Initialize decoder & evaluator
    decoder = BeamSearchDecoder(
        model=model,
        tokenizer=tokenizer,
        length_penalty_alpha=dec_cfg.get("length_penalty_alpha", 0.6),
        max_decoding_len_ratio=dec_cfg.get("max_decoding_len_ratio", 1.5),
        max_decoding_len_const=dec_cfg.get("max_decoding_len_const", 10),
    )

    tracker = MLflowTracker(
        experiment_name="TinySeq2Seq-Evaluation",
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    summary_rows = []

    with tracker.start_run(run_name=f"evaluation_{split}_beams_{beam_sizes}") as run:
        tracker.log_params({
            "split": split,
            "checkpoint": checkpoint_path,
            "beam_sizes": str(beam_sizes),
            "max_sentences": max_sentences,
            "total_sentences": len(dataset),
        })

        evaluator = Seq2SeqEvaluator(
            decoder=decoder,
            tokenizer=tokenizer,
            tracker=tracker,
            plots_dir="artifacts/plots",
        )

        for b_size in beam_sizes:
            results = evaluator.evaluate_dataset(
                dataset=dataset,
                beam_size=b_size,
                max_sentences=max_sentences,
            )
            b_scores = results["bucket_bleu"]
            summary_rows.append([
                b_size,
                results["overall_bleu"],
                b_scores.get("short (<15)", 0.0),
                b_scores.get("medium (15-30)", 0.0),
                b_scores.get("long (>30)", 0.0),
            ])

        # Print summary table
        headers = ["Beam Size (K)", "Overall BLEU", "Short (<15)", "Medium (15-30)", "Long (>30)"]
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS ({split.upper()} SET)")
        print("=" * 60)
        print(tabulate(summary_rows, headers=headers, tablefmt="github"))
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tiny-Seq2Seq Model with SacreBLEU")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=[1, 2, 5, 12])
    parser.add_argument("--max-sentences", type=int, default=None)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        beam_sizes=args.beam_sizes,
        max_sentences=args.max_sentences,
    )
