"""
Production Training CLI for Tiny-Seq2Seq.
Executes model training with mixed precision, gradient clipping, dynamic scheduling,
resilient checkpointing, early stopping, and visualization generation.
Tracks all artifacts and telemetry in MLflow.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.seq2seq.config import load_config
from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.tracker import MLflowTracker
from src.seq2seq.training.trainer import Seq2SeqTrainer
from utils.helpers import get_device, set_seed
from utils.logger import logger


def train(
    config_path: str = "configs/base_config.yaml",
    resume: bool = False,
    epochs: int = None,
    lr: float = None,
    token_budget: int = None,
    run_name: str = "baseline_3layer_tied_adam",
) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    # CLI overrides
    if epochs is not None:
        train_cfg["epochs"] = epochs
    if lr is not None:
        train_cfg["learning_rate"] = lr
    if token_budget is not None:
        train_cfg["token_budget"] = token_budget

    set_seed(config["project"].get("seed", 42))
    device = get_device()
    logger.info(f"Initiating training run on device: {device}")

    # Paths
    processed_dir = data_cfg["processed_dir"]
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]

    train_src_bin = os.path.join(processed_dir, f"train.{src_lang}.bin")
    train_src_idx = os.path.join(processed_dir, f"train.{src_lang}.idx")
    train_tgt_bin = os.path.join(processed_dir, f"train.{tgt_lang}.bin")
    train_tgt_idx = os.path.join(processed_dir, f"train.{tgt_lang}.idx")

    dev_src_bin = os.path.join(processed_dir, f"dev.{src_lang}.bin")
    dev_src_idx = os.path.join(processed_dir, f"dev.{src_lang}.idx")
    dev_tgt_bin = os.path.join(processed_dir, f"dev.{tgt_lang}.bin")
    dev_tgt_idx = os.path.join(processed_dir, f"dev.{tgt_lang}.idx")

    # Load datasets
    train_dataset = ParallelBinaryDataset(train_src_bin, train_src_idx, train_tgt_bin, train_tgt_idx)
    dev_dataset = ParallelBinaryDataset(dev_src_bin, dev_src_idx, dev_tgt_bin, dev_tgt_idx) if os.path.exists(dev_src_bin) else None

    # Load tokenizer
    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".model")]
    if not tok_files:
        raise FileNotFoundError(f"Tokenizer model not found in {tokenizer_dir}")
    tokenizer_path = os.path.join(tokenizer_dir, tok_files[0])
    tokenizer = TokenizerManager(tokenizer_path)

    # Initialize batchers
    batcher_cls = CppBucketBatcher if is_cpp_batcher_available() else BucketBatcher
    logger.info(f"Using batcher backend: {batcher_cls.__name__}")

    train_batcher = batcher_cls(
        dataset=train_dataset,
        token_budget=train_cfg["token_budget"],
        max_seq_len=data_cfg["max_seq_len"],
        reverse_source=model_cfg.get("reverse_source", True),
        shuffle=True,
        seed=config["project"].get("seed", 42),
    )

    dev_batcher = None
    if dev_dataset is not None:
        dev_batcher = batcher_cls(
            dataset=dev_dataset,
            token_budget=train_cfg["token_budget"],
            max_seq_len=data_cfg["max_seq_len"],
            reverse_source=model_cfg.get("reverse_source", True),
            shuffle=False,
        )

    # Initialize model
    model = Seq2SeqModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        tie_weights=model_cfg["tie_weights"],
        init_uniform_bound=model_cfg.get("init_uniform_bound", 0.08),
    )
    param_info = model.count_parameters()
    logger.info(f"Model parameters: {param_info['trainable_unique_parameters']:,} (Tied: {param_info['tied_weights']})")

    # Initialize tracker
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_training"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    resume_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt") if resume else None

    with tracker.start_run(run_name=run_name) as run:
        # Log all configuration parameters
        tracker.log_params({
            "model": model_cfg,
            "training": train_cfg,
            "data": data_cfg,
            "batcher": batcher_cls.__name__,
            "param_count": param_info["trainable_unique_parameters"],
            "device": str(device),
        })

        trainer = Seq2SeqTrainer(
            model=model,
            train_batcher=train_batcher,
            dev_batcher=dev_batcher,
            tokenizer=tokenizer,
            config=config,
            tracker=tracker,
            checkpoint_dir=checkpoint_dir,
            plots_dir="artifacts/plots",
            device=device,
        )

        results = trainer.train(resume_path=resume_path)
        logger.info(f"Training completed successfully. Best Dev Perplexity: {results['best_dev_ppl']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny-Seq2Seq Model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--token-budget", type=int, default=None, help="Override token budget")
    parser.add_argument("--run-name", type=str, default="baseline_training", help="MLflow run name")
    args = parser.parse_args()

    train(
        config_path=args.config,
        resume=args.resume,
        epochs=args.epochs,
        lr=args.lr,
        token_budget=args.token_budget,
        run_name=args.run_name,
    )
