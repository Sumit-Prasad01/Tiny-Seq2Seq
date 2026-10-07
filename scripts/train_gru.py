"""
Training CLI for 4-Layer GRU Seq2Seq with Byte-Level BPE (Tiny-Seq2Seq).
Executes high-throughput training powered by compiled C++ batch builder (seq2seq_c_batcher),
Rust-backed BBPE tokenizer, mixed precision AMP, gradient clipping, resilient checkpointing,
and MLflow telemetry.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.seq2seq.config import load_config
from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.bbpe_tokenizer import BBPETokenizer
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.gru_model import GRUSeq2SeqModel
from src.seq2seq.training.gru_trainer import GRUTrainer
from src.seq2seq.training.tracker import MLflowTracker
from utils.helpers import get_device, set_seed
from utils.logger import logger


def train_gru(
    config_path: str = "configs/config_gru_bbpe.yaml",
    resume: bool = False,
    epochs: int = None,
    lr: float = None,
    token_budget: int = None,
    run_name: str = "gru_4layer_bbpe_training",
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
    logger.info(f"Initiating 4-layer GRU training run on device: {device}")

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

    # Load datasets
    if not os.path.exists(train_src_bin):
        raise FileNotFoundError(
            f"Binary training data not found in {processed_dir}. Run scripts/prepare_bbpe_binary_data.py first."
        )

    train_dataset = ParallelBinaryDataset(train_src_bin, train_src_idx, train_tgt_bin, train_tgt_idx)
    dev_dataset = ParallelBinaryDataset(val_src_bin, val_src_idx, val_tgt_bin, val_tgt_idx) if os.path.exists(val_src_bin) else None

    # Load BBPE tokenizer
    tok_files = [f for f in os.listdir(tokenizer_dir) if f.endswith(".json")]
    if not tok_files:
        raise FileNotFoundError(f"BBPE tokenizer JSON not found in {tokenizer_dir}")
    tokenizer_path = os.path.join(tokenizer_dir, tok_files[0])
    tokenizer = BBPETokenizer(tokenizer_path)

    # Initialize batchers with C++ acceleration
    use_cpp = is_cpp_batcher_available() and train_cfg.get("use_c_batcher", True)
    batcher_cls = CppBucketBatcher if use_cpp else BucketBatcher
    logger.info(f"Using batcher backend: {batcher_cls.__name__} (C++ accelerated: {use_cpp})")

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

    # Initialize GRU model
    model = GRUSeq2SeqModel(
        vocab_size=tokenizer.vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        tie_weights=model_cfg["tie_weights"],
        init_uniform_bound=model_cfg.get("init_uniform_bound", 0.08),
    )
    param_info = model.count_parameters()
    logger.info(
        f"Initialized 4-layer GRU model: {param_info['trainable_unique_parameters']:,} params "
        f"(Tied: {param_info['tied_weights']}, Vocab: {tokenizer.vocab_size})"
    )

    # Initialize tracker
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_training"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints/gru_bbpe")
    resume_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt") if resume else None

    with tracker.start_run(run_name=run_name):
        tracker.log_params({
            "model_type": "gru",
            "d_model": model_cfg["d_model"],
            "n_layers": model_cfg["n_layers"],
            "param_count": param_info["trainable_unique_parameters"],
            "tokenizer_type": "bbpe",
            "batcher": batcher_cls.__name__,
            "device": str(device),
            "token_budget": train_cfg["token_budget"],
            "learning_rate": train_cfg["learning_rate"],
        })

        trainer = GRUTrainer(
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
        logger.info(f"GRU training complete. Best Dev Perplexity: {results['best_dev_ppl']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-Layer GRU Seq2Seq Model with BBPE")
    parser.add_argument("--config", type=str, default="configs/config_gru_bbpe.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--token-budget", type=int, default=None, help="Override token budget")
    parser.add_argument("--run-name", type=str, default="gru_bbpe_training", help="MLflow run name")
    args = parser.parse_args()

    train_gru(
        config_path=args.config,
        resume=args.resume,
        epochs=args.epochs,
        lr=args.lr,
        token_budget=args.token_budget,
        run_name=args.run_name,
    )
