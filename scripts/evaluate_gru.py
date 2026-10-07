"""
Evaluation CLI for 4-Layer GRU Seq2Seq Model with Byte-Level BPE.
Runs vectorized GPU beam search decoding across beam sizes (1, 2, 5, 12),
computes SacreBLEU, evaluates length degradation buckets (<15, 15-30, >30),
and logs metrics and qualitative tables to MLflow.
"""

import argparse
import os
import sys
import time
from typing import List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tabulate import tabulate

from src.seq2seq.config import load_config
from src.seq2seq.data.bbpe_tokenizer import BBPETokenizer
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.decoding.evaluator import Seq2SeqEvaluator
from src.seq2seq.decoding.gru_beam_search import GRUBeamSearchDecoder
from src.seq2seq.gru_model import GRUSeq2SeqModel
from src.seq2seq.training.tracker import MLflowTracker
from utils.helpers import get_device, set_seed
from utils.logger import logger


def evaluate_gru(
    checkpoint_path: str = "checkpoints/gru_bbpe/checkpoint_best.pt",
    config_path: str = "configs/config_gru_bbpe.yaml",
    split: str = "test",
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
    logger.info(f"Initiating 4-layer GRU evaluation on device: {device}")

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
        raise FileNotFoundError(
            f"Split binary file missing: {src_bin}. Run scripts/prepare_bbpe_binary_data.py first."
        )

    dataset = ParallelBinaryDataset(src_bin, src_idx, tgt_bin, tgt_idx)
    logger.info(f"Loaded {len(dataset)} sentence pairs from split '{split}'.")

    # Load BBPE tokenizer
    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".json")]
    if not tok_files:
        raise FileNotFoundError(f"BBPE tokenizer JSON not found in {tokenizer_dir}")
    tokenizer = BBPETokenizer(os.path.join(tokenizer_dir, tok_files[0]))

    # Initialize GRU model
    model = GRUSeq2SeqModel(
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
    decoder = GRUBeamSearchDecoder(
        model=model,
        tokenizer=tokenizer,
        length_penalty_alpha=dec_cfg.get("length_penalty_alpha", 0.6),
        max_decoding_len_ratio=dec_cfg.get("max_decoding_len_ratio", 1.5),
        max_decoding_len_const=dec_cfg.get("max_decoding_len_const", 10),
    )

    tracker = MLflowTracker(
        experiment_name=mlflow_cfg.get("experiment_evaluation", "TinySeq2Seq-GRU-BBPE-Evaluation"),
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    evaluator = Seq2SeqEvaluator(
        decoder=decoder,
        tokenizer=tokenizer,
        tracker=tracker,
        plots_dir="artifacts/plots",
    )

    summary_table = []

    with tracker.start_run(run_name=f"evaluate_gru_{split}"):
        tracker.log_params({
            "model_type": "gru",
            "d_model": model_cfg["d_model"],
            "n_layers": model_cfg["n_layers"],
            "split": split,
            "num_sentences": len(dataset) if max_sentences is None else max_sentences,
            "checkpoint": checkpoint_path,
            "alpha": dec_cfg.get("length_penalty_alpha", 0.6),
        })

        for K in beam_sizes:
            logger.info(f"\n================ Evaluating Beam Width K = {K} ================")
            t0 = time.time()
            res = evaluator.evaluate_dataset(
                dataset=dataset,
                beam_size=K,
                token_budget=2000,
                max_sentences=max_sentences,
            )
            elapsed = time.time() - t0
            bleu = res["bleu"]
            b_scores = res["bucket_scores"]

            summary_table.append([
                f"K={K}",
                f"{bleu:.2f}",
                f"{b_scores.get('short (<15)', 0.0):.2f}",
                f"{b_scores.get('medium (15-30)', 0.0):.2f}",
                f"{b_scores.get('long (>30)', 0.0):.2f}",
                f"{elapsed:.1f}s",
            ])

            # MLflow logging
            tracker.log_metrics(
                {
                    f"bleu_beam_{K}": bleu,
                    f"bleu_short_beam_{K}": b_scores.get("short (<15)", 0.0),
                    f"bleu_medium_beam_{K}": b_scores.get("medium (15-30)", 0.0),
                    f"bleu_long_beam_{K}": b_scores.get("long (>30)", 0.0),
                    f"eval_time_sec_beam_{K}": elapsed,
                }
            )

        # Print summary
        headers = ["Beam Size", "Overall BLEU", "Short (<15)", "Medium (15-30)", "Long (>30)", "Time"]
        print("\n" + "=" * 60)
        print("4-LAYER GRU EVALUATION RESULTS (SacreBLEU)")
        print("=" * 60)
        print(tabulate(summary_table, headers=headers, tablefmt="github"))
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 4-Layer GRU Seq2Seq Model with BBPE")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gru_bbpe/checkpoint_best.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/config_gru_bbpe.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (val or test)")
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=None, help="Beam sizes to evaluate")
    parser.add_argument("--max-sentences", type=int, default=None, help="Cap evaluation sentences for fast testing")
    args = parser.parse_args()

    evaluate_gru(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        beam_sizes=args.beam_sizes,
        max_sentences=args.max_sentences,
    )
