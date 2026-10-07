"""
High-Throughput Binary Serialization Script for Byte-Level BPE (Tiny-Seq2Seq).
Encodes cleaned parallel text into flat uint16 binary files (.bin) and int64 index offsets (.idx)
using native multithreaded batch tokenization and vectorized I/O.
Outputs are directly memory-mappable by the C++ batch builder (seq2seq_c_batcher).
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.seq2seq.config import load_config
from src.seq2seq.data.bbpe_tokenizer import BBPETokenizer
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


def serialize_split_fast(
    src_file: str,
    tgt_file: str,
    src_bin: str,
    src_idx: str,
    tgt_bin: str,
    tgt_idx: str,
    tokenizer: BBPETokenizer,
    max_len: int = 64,
    chunk_size: int = 25000,
) -> Dict[str, float]:
    """
    Serializes a parallel text split into binary format with native multithreaded chunking.
    """
    os.makedirs(os.path.dirname(os.path.abspath(src_bin)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(tgt_bin)), exist_ok=True)

    src_offsets = [0]
    tgt_offsets = [0]
    total_src_tokens = 0
    total_tgt_tokens = 0
    num_pairs = 0

    t0 = time.time()
    with open(src_file, "r", encoding="utf-8") as f_src, \
         open(tgt_file, "r", encoding="utf-8") as f_tgt, \
         open(src_bin, "wb") as out_src_bin, \
         open(tgt_bin, "wb") as out_tgt_bin:

        chunk_src_text: List[str] = []
        chunk_tgt_text: List[str] = []

        def process_chunk(src_lines: List[str], tgt_lines: List[str]):
            nonlocal num_pairs, total_src_tokens, total_tgt_tokens
            if not src_lines:
                return

            # Native multithreaded Rust/C++ batch tokenization (releases GIL)
            src_encs = tokenizer.encode_batch(src_lines)
            tgt_encs = tokenizer.encode_batch(tgt_lines)

            valid_src_tokens: List[int] = []
            valid_tgt_tokens: List[int] = []

            for s_ids, t_ids in zip(src_encs, tgt_encs):
                s_len = len(s_ids)
                t_len = len(t_ids)
                if s_len == 0 or t_len == 0 or s_len > max_len or t_len > max_len:
                    continue

                valid_src_tokens.extend(s_ids)
                total_src_tokens += s_len
                src_offsets.append(total_src_tokens)

                valid_tgt_tokens.extend(t_ids)
                total_tgt_tokens += t_len
                tgt_offsets.append(total_tgt_tokens)

                num_pairs += 1

            if valid_src_tokens:
                src_arr = np.array(valid_src_tokens, dtype=np.uint16)
                out_src_bin.write(src_arr.tobytes())

            if valid_tgt_tokens:
                tgt_arr = np.array(valid_tgt_tokens, dtype=np.uint16)
                out_tgt_bin.write(tgt_arr.tobytes())

        for line_src, line_tgt in zip(f_src, f_tgt):
            s_str = line_src.strip()
            t_str = line_tgt.strip()
            if s_str and t_str:
                chunk_src_text.append(s_str)
                chunk_tgt_text.append(t_str)

            if len(chunk_src_text) >= chunk_size:
                process_chunk(chunk_src_text, chunk_tgt_text)
                chunk_src_text.clear()
                chunk_tgt_text.clear()

        # Flush residual
        if chunk_src_text:
            process_chunk(chunk_src_text, chunk_tgt_text)

    # Save offset indices
    np.array(src_offsets, dtype=np.int64).tofile(src_idx)
    np.array(tgt_offsets, dtype=np.int64).tofile(tgt_idx)

    elapsed = time.time() - t0
    logger.info(
        f"Serialized {num_pairs:,} sentence pairs ({total_src_tokens:,} src, {total_tgt_tokens:,} tgt tokens) "
        f"in {elapsed:.2f}s ({num_pairs / max(elapsed, 0.001):.0f} pairs/sec)"
    )

    avg_src_len = total_src_tokens / num_pairs if num_pairs > 0 else 0
    avg_tgt_len = total_tgt_tokens / num_pairs if num_pairs > 0 else 0

    return {
        "num_pairs": float(num_pairs),
        "src_tokens": float(total_src_tokens),
        "tgt_tokens": float(total_tgt_tokens),
        "avg_src_len": round(avg_src_len, 2),
        "avg_tgt_len": round(avg_tgt_len, 2),
        "serialization_time_sec": round(elapsed, 2),
    }


def prepare_bbpe_binary_data(
    config_path: str = "configs/config_gru_bbpe.yaml",
    tokenizer_path: str = None,
) -> None:
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

    if tokenizer_path is None:
        vocab_size = tok_cfg.get("vocab_size", 16000)
        prefix = f"bbpe_{vocab_size // 1000}k" if vocab_size >= 1000 else f"bbpe_{vocab_size}"
        tokenizer_path = os.path.join(tokenizer_dir, f"{prefix}.json")

    logger.info(f"Loading BBPE tokenizer from {tokenizer_path}...")
    tokenizer = BBPETokenizer(tokenizer_path)

    stats = {}
    splits = ["train", "val", "test"]

    for split in splits:
        src_txt = os.path.join(cleaned_dir, f"{split}.{src_lang}")
        tgt_txt = os.path.join(cleaned_dir, f"{split}.{tgt_lang}")

        src_bin = os.path.join(processed_dir, f"{split}.{src_lang}.bin")
        src_idx = os.path.join(processed_dir, f"{split}.{src_lang}.idx")
        tgt_bin = os.path.join(processed_dir, f"{split}.{tgt_lang}.bin")
        tgt_idx = os.path.join(processed_dir, f"{split}.{tgt_lang}.idx")

        if not os.path.exists(src_txt) or not os.path.exists(tgt_txt):
            logger.warning(f"Split {split} text files not found in {cleaned_dir}. Skipping.")
            continue

        logger.info(f"Serializing {split} split with fast multithreaded BBPE encoder...")
        split_stats = serialize_split_fast(
            src_file=src_txt,
            tgt_file=tgt_txt,
            src_bin=src_bin,
            src_idx=src_idx,
            tgt_bin=tgt_bin,
            tgt_idx=tgt_idx,
            tokenizer=tokenizer,
            max_len=max_len,
        )
        stats[split] = split_stats

    # Log to MLflow
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg.get("experiment_data", "TinySeq2Seq-GRU-BBPE-Data"),
        tracking_uri=mlflow_cfg["tracking_uri"],
    )
    with tracker.start_run(run_name="prepare_bbpe_binary_data"):
        for split, s in stats.items():
            for k, v in s.items():
                tracker.log_metrics({f"{split}_{k}": v})

        tracker.log_params({
            "tokenizer_path": tokenizer_path,
            "max_seq_len": max_len,
            "vocab_size": tokenizer.vocab_size,
        })

    logger.info("BBPE binary data serialization completed and logged to MLflow successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare high-throughput binary data with BBPE")
    parser.add_argument("--config", type=str, default="configs/config_gru_bbpe.yaml", help="Path to config")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Path to BBPE tokenizer JSON")
    args = parser.parse_args()

    prepare_bbpe_binary_data(config_path=args.config, tokenizer_path=args.tokenizer_path)
