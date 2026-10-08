"""
Training CLI for 4-Layer Bidirectional LSTM Seq2Seq Model (Tiny-Seq2Seq).
Executes high-throughput training powered by compiled C++ batch builder (seq2seq_c_batcher),
mixed precision AMP, gradient clipping, resilient checkpointing, and MLflow telemetry.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.seq2seq.bilstm_model import BiLSTMSeq2SeqModel
from src.seq2seq.config import load_config
from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.training.bilstm_trainer import BiLSTMTrainer
from src.seq2seq.training.tracker import MLflowTracker
from utils.helpers import get_device, set_seed
from utils.logger import logger


def train_bilstm(
    config_path: str = "configs/config_bilstm.yaml",
    resume: bool = False,
    epochs: int = None,
    lr: float = None,
    token_budget: int = None,
    run_name: str = "bilstm_4layer_training",
) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    if epochs is not None:
        train_cfg["epochs"] = epochs
    if lr is not None:
        train_cfg["learning_rate"] = lr
    if token_budget is not None:
        train_cfg["token_budget"] = token_budget

    set_seed(config["project"].get("seed", 42))
    device = get_device()
    logger.info(f"Initiating 4-layer BiLSTM training run on device: {device}")

    # Paths
    processed_dir = data_cfg["processed_dir"]
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]

    train_src_bin = os.path.join(processed_dir, f"train.{src_lang}.bin")
    train_src_idx = os.path.join(processed_dir, f"train.{src_lang}.idx")
    train_tgt_bin = os.path.join(processed_dir, f"train.{tgt_lang}.bin")
    train_tgt_idx = os.path.join(processed_dir, f"train.{tgt_lang}.idx")

    val_src_bin = os.path.join(processed_dir, f"val.{src_lang}.bin")
    val_src_idx = os.path.join(processed_dir, f"val.{src_lang}.idx")
    val_tgt_bin = os.path.join(processed_dir, f"val.{tgt_lang}.bin")
    val_tgt_idx = os.path.join(processed_dir, f"val.{tgt_lang}.idx")

    # If 'val' does not exist, try 'dev'
    if not os.path.exists(val_src_bin):
        val_src_bin = os.path.join(processed_dir, f"dev.{src_lang}.bin")
        val_src_idx = os.path.join(processed_dir, f"dev.{src_lang}.idx")
        val_tgt_bin = os.path.join(processed_dir, f"dev.{tgt_lang}.bin")
        val_tgt_idx = os.path.join(processed_dir, f"dev.{tgt_lang}.idx")

    # Load datasets
    if not os.path.exists(train_src_bin):
        raise FileNotFoundError(
            f"Binary training data not found in {processed_dir}. Run scripts/prepare_binary_data.py first."
        )

    train_dataset = ParallelBinaryDataset(train_src_bin, train_src_idx, train_tgt_bin, train_tgt_idx)
    dev_dataset = (
        ParallelBinaryDataset(val_src_bin, val_src_idx, val_tgt_bin, val_tgt_idx)
        if os.path.exists(val_src_bin)
        else None
    )

    # Load tokenizer
    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".model")]
    if not tok_files:
        raise FileNotFoundError(f"Tokenizer model not found in {tokenizer_dir}")
    tokenizer_path = os.path.join(tokenizer_dir, tok_files[0])
    tokenizer = TokenizerManager(tokenizer_path)

    # Initialize batchers with C++ acceleration
    use_cpp = is_cpp_batcher_available() and train_cfg.get("use_c_batcher", True)
    batcher_cls = CppBucketBatcher if use_cpp else BucketBatcher
    logger.info(f"Using batcher backend: {batcher_cls.__name__} (C++ accelerated: {use_cpp})")

    train_batcher = batcher_cls(
        dataset=train_dataset,
        token_budget=train_cfg["token_budget"],
        max_seq_len=data_cfg["max_seq_len"],
        reverse_source=model_cfg.get("reverse_source", False),
        shuffle=True,
        seed=config["project"].get("seed", 42),
    )

    dev_batcher = None
    if dev_dataset is not None:
        dev_batcher = batcher_cls(
            dataset=dev_dataset,
            token_budget=train_cfg["token_budget"],
            max_seq_len=data_cfg["max_seq_len"],
            reverse_source=model_cfg.get("reverse_source", False),
            shuffle=False,
        )

    # Initialize BiLSTM model
    model = BiLSTMSeq2SeqModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        tie_weights=model_cfg["tie_weights"],
        init_uniform_bound=model_cfg.get("init_uniform_bound", 0.08),
    )
    param_info = model.count_parameters()
    logger.info(
        f"Initialized 4-layer BiLSTM model: {param_info['trainable_unique_parameters']:,} params "
        f"(Tied: {param_info['tied_weights']}, Vocab: {tokenizer.vocab_size})"
    )

    # Initialize tracker
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_training"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints/bilstm")
    resume_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt") if resume else None

    with tracker.start_run(run_name=run_name):
        tracker.log_params({
            "model_type": "bilstm",
            "d_model": model_cfg["d_model"],
            "n_layers": model_cfg["n_layers"],
            "param_count": param_info["trainable_unique_parameters"],
            "batcher": batcher_cls.__name__,
            "device": str(device),
            "token_budget": train_cfg["token_budget"],
            "learning_rate": train_cfg["learning_rate"],
        })

        trainer = BiLSTMTrainer(
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
        logger.info(f"BiLSTM training complete. Best Dev Perplexity: {results['best_dev_ppl']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-Layer BiLSTM Seq2Seq Model")
    parser.add_argument("--config", type=str, default="configs/config_bilstm.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--token-budget", type=int, default=None, help="Override token budget")
    parser.add_argument("--run-name", type=str, default="bilstm_training", help="MLflow run name")
    args = parser.parse_args()

    train_bilstm(
        config_path=args.config,
        resume=args.resume,
        epochs=args.epochs,
        lr=args.lr,
        token_budget=args.token_budget,
        run_name=args.run_name,
    )
