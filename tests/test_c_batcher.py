"""
Unit tests for Phase 5: High-Performance C++ Batch Builder (pybind11).
Validates:
- Import of compiled C++ module
- Parity between Python BucketBatcher and C++ CppBucketBatcher on identical indices
- Exact tensor match (src_ids, tgt_in_ids, tgt_out_ids, lengths, num_tokens)
- 100% sentence coverage per epoch
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.binary_serializer import BinaryWriter
from src.seq2seq.data.c_batcher import CppBucketBatcher, is_cpp_batcher_available
from src.seq2seq.data.dataset import ParallelBinaryDataset


@pytest.fixture
def synthetic_binary_dataset():
    """Generates a synthetic binary dataset with varying sentence lengths."""
    temp_dir = tempfile.mkdtemp()
    src_bin = os.path.join(temp_dir, "train.src.bin")
    src_idx = os.path.join(temp_dir, "train.src.idx")
    tgt_bin = os.path.join(temp_dir, "train.tgt.bin")
    tgt_idx = os.path.join(temp_dir, "train.tgt.idx")

    src_writer = BinaryWriter(src_bin, src_idx)
    tgt_writer = BinaryWriter(tgt_bin, tgt_idx)

    np.random.seed(42)
    num_pairs = 150
    for _ in range(num_pairs):
        s_len = np.random.randint(5, 30)
        t_len = np.random.randint(5, 30)
        s_tokens = np.random.randint(4, 400, size=s_len).tolist()
        t_tokens = np.random.randint(4, 400, size=t_len).tolist()
        src_writer.write_sentence(s_tokens)
        tgt_writer.write_sentence(t_tokens)

    src_writer.close()
    tgt_writer.close()

    dataset = ParallelBinaryDataset(src_bin, src_idx, tgt_bin, tgt_idx)
    yield dataset
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_cpp_extension_available():
    assert is_cpp_batcher_available(), "Compiled C++ extension seq2seq_c_batcher was not found"


def test_python_cpp_batcher_collate_parity(synthetic_binary_dataset):
    """
    Asserts that given identical sentence indices, Python and C++ batchers
    produce identical PyTorch tensors.
    """
    py_batcher = BucketBatcher(
        dataset=synthetic_binary_dataset,
        token_budget=1000,
        reverse_source=True,
        shuffle=False,
    )

    cpp_batcher = CppBucketBatcher(
        dataset=synthetic_binary_dataset,
        token_budget=1000,
        reverse_source=True,
        shuffle=False,
    )

    test_indices = [0, 5, 12, 23, 40]
    py_batch = py_batcher.collate_batch(test_indices)
    cpp_batch = cpp_batcher.collate_batch(test_indices)

    assert torch.equal(py_batch.src_ids, cpp_batch.src_ids), "src_ids mismatch between Python and C++"
    assert torch.equal(py_batch.src_lens, cpp_batch.src_lens), "src_lens mismatch between Python and C++"
    assert torch.equal(py_batch.tgt_in_ids, cpp_batch.tgt_in_ids), "tgt_in_ids mismatch between Python and C++"
    assert torch.equal(py_batch.tgt_out_ids, cpp_batch.tgt_out_ids), "tgt_out_ids mismatch between Python and C++"
    assert torch.equal(py_batch.tgt_lens, cpp_batch.tgt_lens), "tgt_lens mismatch between Python and C++"
    assert py_batch.num_tokens == cpp_batch.num_tokens, "num_tokens mismatch between Python and C++"


def test_cpp_batcher_coverage_and_budget(synthetic_binary_dataset):
    """Asserts that C++ batcher plans 100% of sentences and respects token budget."""
    token_budget = 800
    cpp_batcher = CppBucketBatcher(
        dataset=synthetic_binary_dataset,
        token_budget=token_budget,
        reverse_source=True,
        shuffle=True,
        seed=42,
    )

    batches = cpp_batcher.plan_batches()
    flattened = [idx for b in batches for idx in b]

    assert len(flattened) == len(synthetic_binary_dataset), "Missing sentences in C++ plan"
    assert len(set(flattened)) == len(synthetic_binary_dataset), "Duplicate sentences in C++ plan"

    for b_indices in batches:
        batch = cpp_batcher.collate_batch(b_indices)
        if len(b_indices) > 1:
            batch_tokens = max(batch.src_ids.shape[1], batch.tgt_out_ids.shape[1]) * len(b_indices)
            assert batch_tokens <= token_budget * 1.5, f"C++ batch exceeded budget: {batch_tokens}"
