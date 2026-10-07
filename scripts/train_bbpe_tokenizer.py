"""
Byte-Level BPE (BBPE) Tokenizer Training Script for Tiny-Seq2Seq.
Trains a shared 16k BBPE model on cleaned source and target parallel corpora,
validates special token IDs (<pad>=0, <unk>=1, <bos>=2, <eos>=3),
and logs the tokenizer artifact to MLflow.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.seq2seq.config import load_config
from src.seq2seq.data.bbpe_tokenizer import BBPETokenizer
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


def train_bbpe_tokenizer(config_path: str = "configs/config_gru_bbpe.yaml", vocab_size: int = None) -> BBPETokenizer:
    config = load_config(config_path)
    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]
    mlflow_cfg = config["mlflow"]

    cleaned_dir = data_cfg["cleaned_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]

    train_src = os.path.join(cleaned_dir, f"train.{src_lang}")
    train_tgt = os.path.join(cleaned_dir, f"train.{tgt_lang}")

    if not os.path.exists(train_src) or not os.path.exists(train_tgt):
        raise FileNotFoundError(
            f"Cleaned training files not found in {cleaned_dir}. Please run scripts/prepare_data.py first."
        )

    tokenizer_dir = data_cfg["tokenizer_dir"]
    os.makedirs(tokenizer_dir, exist_ok=True)

    actual_vocab_size = vocab_size or tok_cfg.get("vocab_size", 16000)
    output_filename = f"bbpe_{actual_vocab_size // 1000}k.json" if actual_vocab_size >= 1000 else f"bbpe_{actual_vocab_size}.json"
    output_path = os.path.join(tokenizer_dir, output_filename)

    logger.info(f"Training shared Byte-Level BPE tokenizer on {train_src} and {train_tgt}...")
    tokenizer = BBPETokenizer.train(
        input_files=[train_src, train_tgt],
        output_path=output_path,
        vocab_size=actual_vocab_size,
        min_frequency=tok_cfg.get("min_frequency", 2),
    )

    # Log to MLflow
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg.get("experiment_data", "TinySeq2Seq-GRU-BBPE-Data"),
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    with tracker.start_run(run_name="train_bbpe_tokenizer"):
        tracker.log_params({
            "tokenizer.vocab_size": tokenizer.vocab_size,
            "tokenizer.model_type": "bbpe",
            "tokenizer.pad_id": tokenizer.PAD_ID,
            "tokenizer.unk_id": tokenizer.UNK_ID,
            "tokenizer.bos_id": tokenizer.BOS_ID,
            "tokenizer.eos_id": tokenizer.EOS_ID,
        })
        tracker.log_artifact(output_path, artifact_path="tokenizer")

    logger.info(f"BBPE Tokenizer trained and logged to MLflow successfully. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE Tokenizer for Tiny-Seq2Seq")
    parser.add_argument("--config", type=str, default="configs/config_gru_bbpe.yaml", help="Path to config")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocabulary size")
    args = parser.parse_args()

    train_bbpe_tokenizer(config_path=args.config, vocab_size=args.vocab_size)
