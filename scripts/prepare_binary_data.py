"""
Binary Serialization Script for Tiny-Seq2Seq.
Encodes cleaned parallel text into flat uint16 binary files (.bin) and int64 index offsets (.idx).
Logs dataset statistics and shapes to MLflow.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.seq2seq.config import load_config

from src.seq2seq.data.binary_serializer import BinaryReader, BinaryWriter
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


def serialize_split(
    src_file: str,
    tgt_file: str,
    src_bin: str,
    src_idx: str,
    tgt_bin: str,
    tgt_idx: str,
    tokenizer: TokenizerManager,
    max_len: int = 64,
) -> dict:
    """Encodes and serializes one dataset split into flat binary arrays."""
    src_writer = BinaryWriter(src_bin, src_idx)
    tgt_writer = BinaryWriter(tgt_bin, tgt_idx)

    num_pairs = 0
    total_src_tokens = 0
    total_tgt_tokens = 0

    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
        for line_src, line_tgt in zip(f_src, f_tgt):
            src_text = line_src.strip()
            tgt_text = line_tgt.strip()
            if not src_text or not tgt_text:
                continue

            src_tokens = tokenizer.encode(src_text)
            tgt_tokens = tokenizer.encode(tgt_text)

            # Cap length to hardware budget
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue

            src_writer.write_sentence(src_tokens)
            tgt_writer.write_sentence(tgt_tokens)

            num_pairs += 1
            total_src_tokens += len(src_tokens)
            total_tgt_tokens += len(tgt_tokens)

    src_writer.close()
    tgt_writer.close()

    avg_src_len = total_src_tokens / num_pairs if num_pairs > 0 else 0
    avg_tgt_len = total_tgt_tokens / num_pairs if num_pairs > 0 else 0

    return {
        "num_pairs": num_pairs,
        "src_tokens": total_src_tokens,
        "tgt_tokens": total_tgt_tokens,
        "avg_src_len": round(avg_src_len, 2),
        "avg_tgt_len": round(avg_tgt_len, 2),
    }


def prepare_binary_data(config_path: str = "configs/base_config.yaml", tokenizer_model_path: str = None) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]
    mlflow_cfg = config["mlflow"]

    cleaned_dir = data_cfg["cleaned_dir"]
    processed_dir = data_cfg["processed_dir"]
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]
    max_len = data_cfg["max_seq_len"]

    os.makedirs(processed_dir, exist_ok=True)

    # Locate tokenizer model
    if tokenizer_model_path is None:
        vocab_size = tok_cfg["vocab_size"]
        prefix = f"spm_{vocab_size // 1000}k" if vocab_size >= 1000 else f"spm_{vocab_size}"
        tokenizer_model_path = os.path.join(tokenizer_dir, f"{prefix}.model")

    logger.info(f"Loading tokenizer from {tokenizer_model_path}...")
    tokenizer = TokenizerManager(tokenizer_model_path)

    stats = {}
    for split in ["train", "dev", "test"]:
        src_file = os.path.join(cleaned_dir, f"{split}.{src_lang}")
        tgt_file = os.path.join(cleaned_dir, f"{split}.{tgt_lang}")

        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            logger.warning(f"Split file missing: {src_file} or {tgt_file}. Skipping {split}.")
            continue

        src_bin = os.path.join(processed_dir, f"{split}.{src_lang}.bin")
        src_idx = os.path.join(processed_dir, f"{split}.{src_lang}.idx")
        tgt_bin = os.path.join(processed_dir, f"{split}.{tgt_lang}.bin")
        tgt_idx = os.path.join(processed_dir, f"{split}.{tgt_lang}.idx")

        logger.info(f"Serializing split '{split}' into binary format...")
        split_stats = serialize_split(
            src_file, tgt_file, src_bin, src_idx, tgt_bin, tgt_idx, tokenizer, max_len=max_len
        )
        stats[split] = split_stats
        logger.info(f"Split '{split}': {split_stats}")

        # Verification: Read back first sentence via memmap
        reader_src = BinaryReader(src_bin, src_idx)
        reader_tgt = BinaryReader(tgt_bin, tgt_idx)
        assert len(reader_src) == len(reader_tgt) == split_stats["num_pairs"]
        if split_stats["num_pairs"] > 0:
            first_src = tokenizer.decode(reader_src[0].tolist())
            first_tgt = tokenizer.decode(reader_tgt[0].tolist())
            logger.info(f"Sample decoded [{split}]:\n  Src: {first_src}\n  Tgt: {first_tgt}")

    # Log to MLflow
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_data"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    with tracker.start_run(run_name="binary_serialization") as run:
        tracker.log_params({
            "max_len": max_len,
            "vocab_size": tokenizer.vocab_size,
            "processed_dir": processed_dir,
        })
        for split, s in stats.items():
            for k, v in s.items():
                tracker.log_step_metrics(step=0, metrics={f"{split}/{k}": float(v)})

    logger.info("Binary serialization completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serialize tokenized datasets to flat binary arrays")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config")
    parser.add_argument("--tokenizer-model", type=str, default=None, help="Path to tokenizer .model file")
    args = parser.parse_args()

    prepare_binary_data(config_path=args.config, tokenizer_model_path=args.tokenizer_model)
