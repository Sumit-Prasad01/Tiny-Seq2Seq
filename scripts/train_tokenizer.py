"""
SentencePiece BPE Tokenizer Training Script for Tiny-Seq2Seq.
Trains a shared 16k BPE model on cleaned source and target parallel corpora,
validates special token IDs (<pad>=0, <unk>=1, <bos>=2, <eos>=3),
and logs the tokenizer artifact to MLflow.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.seq2seq.config import load_config

from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


def train_tokenizer(config_path: str = "configs/base_config.yaml", vocab_size: int = None) -> TokenizerManager:
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

    actual_vocab_size = vocab_size or tok_cfg["vocab_size"]
    model_prefix = os.path.join(tokenizer_dir, f"spm_{actual_vocab_size // 1000}k" if actual_vocab_size >= 1000 else f"spm_{actual_vocab_size}")

    logger.info(f"Training shared SentencePiece BPE tokenizer on {train_src} and {train_tgt}...")
    tokenizer = TokenizerManager.train(
        input_files=[train_src, train_tgt],
        model_prefix=model_prefix,
        vocab_size=actual_vocab_size,
        model_type=tok_cfg["model_type"],
        character_coverage=tok_cfg["character_coverage"],
    )

    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    # Log to MLflow
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_data"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    with tracker.start_run(run_name="train_tokenizer") as run:
        tracker.log_params({
            "tokenizer.vocab_size": tokenizer.vocab_size,
            "tokenizer.model_type": tok_cfg["model_type"],
            "tokenizer.pad_id": tokenizer.PAD_ID,
            "tokenizer.unk_id": tokenizer.UNK_ID,
            "tokenizer.bos_id": tokenizer.BOS_ID,
            "tokenizer.eos_id": tokenizer.EOS_ID,
        })
        tracker.log_artifact(model_file, artifact_path="tokenizer")
        tracker.log_artifact(vocab_file, artifact_path="tokenizer")

    logger.info(f"Tokenizer trained and logged to MLflow successfully. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE Tokenizer for Tiny-Seq2Seq")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocabulary size")
    args = parser.parse_args()

    train_tokenizer(config_path=args.config, vocab_size=args.vocab_size)
