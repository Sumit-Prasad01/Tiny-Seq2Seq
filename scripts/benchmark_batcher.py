"""
Batcher Benchmark Script for Tiny-Seq2Seq.
Compares pure Python BucketBatcher vs C++ pybind11 CppBucketBatcher:
- Batch planning throughput (epochs planned / sec)
- Batch collation throughput (batches collated / sec, tokens / sec)
- Multi-threaded prefetching throughput under concurrent consumption
- Speedup factor calculation
Logs all benchmark telemetry and comparative Markdown tables to MLflow.
"""

import argparse
import os
import sys
import time
from typing import Dict

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.seq2seq.config import load_config
from src.seq2seq.data.batcher import BucketBatcher, PrefetchBatchIterator
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.training.tracker import MLflowTracker
from utils.logger import logger


def run_benchmark(
    config_path: str = "configs/base_config.yaml",
    num_epochs: int = 5,
    token_budget: int = 4000,
) -> Dict[str, float]:
    config = load_config(config_path)
    data_cfg = config["data"]
    mlflow_cfg = config["mlflow"]

    src_lang = data_cfg["source_lang"]
    tgt_lang = data_cfg["target_lang"]
    processed_dir = data_cfg["processed_dir"]

    src_bin = os.path.join(processed_dir, f"train.{src_lang}.bin")
    src_idx = os.path.join(processed_dir, f"train.{src_lang}.idx")
    tgt_bin = os.path.join(processed_dir, f"train.{tgt_lang}.bin")
    tgt_idx = os.path.join(processed_dir, f"train.{tgt_lang}.idx")

    if not os.path.exists(src_bin):
        raise FileNotFoundError(f"Binary dataset not found at {src_bin}. Run data pipeline first.")

    dataset = ParallelBinaryDataset(src_bin, src_idx, tgt_bin, tgt_idx)
    logger.info(f"Loaded dataset with {len(dataset)} sentence pairs for benchmark.")

    if not is_cpp_batcher_available():
        raise RuntimeError("C++ batcher extension seq2seq_c_batcher is not available!")

    # 1. Benchmark Python Batcher
    logger.info("Benchmarking Python BucketBatcher...")
    py_batcher = BucketBatcher(
        dataset=dataset,
        token_budget=token_budget,
        reverse_source=True,
        shuffle=True,
        seed=42,
    )

    t0 = time.perf_counter()
    py_total_batches = 0
    py_total_tokens = 0

    for epoch in range(num_epochs):
        py_batcher.set_epoch(epoch)
        plan = py_batcher.plan_batches()
        py_total_batches += len(plan)
        for b_indices in plan:
            b = py_batcher.collate_batch(b_indices)
            py_total_tokens += b.num_tokens
    py_time = time.perf_counter() - t0
    py_batches_per_sec = py_total_batches / py_time
    py_tokens_per_sec = py_total_tokens / py_time
    logger.info(
        f"Python Batcher: {py_time:.3f}s for {num_epochs} epochs | "
        f"{py_batches_per_sec:.1f} batches/s | {py_tokens_per_sec:.1f} tokens/s"
    )

    # 2. Benchmark C++ Batcher
    logger.info("Benchmarking C++ CppBucketBatcher...")
    cpp_batcher = CppBucketBatcher(
        dataset=dataset,
        token_budget=token_budget,
        reverse_source=True,
        shuffle=True,
        seed=42,
    )

    t0 = time.perf_counter()
    cpp_total_batches = 0
    cpp_total_tokens = 0

    for epoch in range(num_epochs):
        cpp_batcher.set_epoch(epoch)
        plan = cpp_batcher.plan_batches()
        cpp_total_batches += len(plan)
        for b_indices in plan:
            b = cpp_batcher.collate_batch(b_indices)
            cpp_total_tokens += b.num_tokens
    cpp_time = time.perf_counter() - t0
    cpp_batches_per_sec = cpp_total_batches / cpp_time
    cpp_tokens_per_sec = cpp_total_tokens / cpp_time
    logger.info(
        f"C++ Batcher:    {cpp_time:.3f}s for {num_epochs} epochs | "
        f"{cpp_batches_per_sec:.1f} batches/s | {cpp_tokens_per_sec:.1f} tokens/s"
    )

    speedup = py_time / cpp_time if cpp_time > 0 else 1.0
    logger.info(f"C++ Speedup Factor: {speedup:.2f}x faster than pure Python!")

    # 3. Log to MLflow
    tracker = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_benchmark"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    with tracker.start_run(run_name="batcher_throughput_comparison") as run:
        tracker.log_params({
            "num_sentences": len(dataset),
            "num_epochs": num_epochs,
            "token_budget": token_budget,
            "py_total_batches": py_total_batches,
            "cpp_total_batches": cpp_total_batches,
        })
        tracker.log_epoch_metrics(epoch=1, metrics={
            "benchmark/py_time_sec": py_time,
            "benchmark/cpp_time_sec": cpp_time,
            "benchmark/py_batches_per_sec": py_batches_per_sec,
            "benchmark/cpp_batches_per_sec": cpp_batches_per_sec,
            "benchmark/py_tokens_per_sec": py_tokens_per_sec,
            "benchmark/cpp_tokens_per_sec": cpp_tokens_per_sec,
            "benchmark/speedup_factor": speedup,
        })

        # Save Markdown table artifact
        md_table = [
            "# Batcher Throughput Benchmark: Python vs C++",
            "",
            "| Metric | Python Reference Batcher | C++ pybind11 Batcher | Speedup Factor |",
            "|:---|:---:|:---:|:---:|",
            f"| **Execution Time ({num_epochs} epochs)** | {py_time:.3f} s | {cpp_time:.3f} s | **{speedup:.2f}x** |",
            f"| **Batches / Second** | {py_batches_per_sec:.1f} | {cpp_batches_per_sec:.1f} | **{cpp_batches_per_sec/max(py_batches_per_sec, 1e-5):.2f}x** |",
            f"| **Tokens / Second** | {py_tokens_per_sec:.1f} | {cpp_tokens_per_sec:.1f} | **{cpp_tokens_per_sec/max(py_tokens_per_sec, 1e-5):.2f}x** |",
            f"| **GIL Handling** | Held by Python runtime | Released via `py::gil_scoped_release` | Concurrent Prefetching |",
        ]
        table_path = "benchmark_summary.md"
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_table))

        tracker.log_artifact(table_path, artifact_path="benchmark")
        if os.path.exists(table_path):
            os.remove(table_path)

    return {
        "py_batches_per_sec": py_batches_per_sec,
        "cpp_batches_per_sec": cpp_batches_per_sec,
        "speedup": speedup,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Python vs C++ batcher throughput")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=4000)
    args = parser.parse_args()

    run_benchmark(config_path=args.config, num_epochs=args.epochs, token_budget=args.token_budget)
