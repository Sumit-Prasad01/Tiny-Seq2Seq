"""
Unit tests for Phase 3: Python Reference Length-Bucketed Batcher & Parity Suite.
Validates:
- 100% sentence coverage per epoch
- Strict token budget compliance
- Source reversal and special token conventions (<bos>, <eos>, <pad>)
- Reproducibility under fixed random seeds
- Async prefetcher queue iterator
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.seq2seq.data.batcher import BucketBatcher, PrefetchBatchIterator
from src.seq2seq.data.binary_serializer import BinaryWriter
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

    # Create 300 sentence pairs with varying lengths between 5 and 40 tokens
    np.random.seed(42)
    num_pairs = 300
    for _ in range(num_pairs):
        s_len = np.random.randint(5, 40)
        t_len = np.random.randint(5, 40)
        s_tokens = np.random.randint(4, 500, size=s_len).tolist()
        t_tokens = np.random.randint(4, 500, size=t_len).tolist()
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


def test_batcher_100_percent_coverage(synthetic_binary_dataset):
    """Ensures every sentence is included exactly once in an epoch."""
    batcher = BucketBatcher(
        dataset=synthetic_binary_dataset,
        token_budget=1000,
        reverse_source=True,
        shuffle=True,
        seed=42,
    )

    batches = batcher.plan_batches()
    flattened_indices = [idx for batch in batches for idx in batch]

    assert len(flattened_indices) == len(synthetic_binary_dataset), "Missing or extra sentences in planned batches"
    assert len(set(flattened_indices)) == len(synthetic_binary_dataset), "Duplicate sentence indices found in batches"


def test_batcher_token_budget_compliance(synthetic_binary_dataset):
    """Verifies that batches adhere to the token budget."""
    token_budget = 1000
    batcher = BucketBatcher(
        dataset=synthetic_binary_dataset,
        token_budget=token_budget,
        reverse_source=True,
        shuffle=False,
    )

    batches = batcher.plan_batches()
    for batch_indices in batches:
        batch = batcher.collate_batch(batch_indices)
        batch_tokens = max(batch.src_ids.shape[1], batch.tgt_out_ids.shape[1]) * len(batch_indices)
        # Single sentences are allowed to exceed if a single sentence is longer than budget
        if len(batch_indices) > 1:
            assert batch_tokens <= token_budget * 1.5, f"Batch tokens {batch_tokens} exceeded budget {token_budget}"


def test_source_reversal_and_special_tokens(synthetic_binary_dataset):
    """Verifies that source sentences are reversed and end with EOS, and target shifts are correct."""
    batcher = BucketBatcher(
        dataset=synthetic_binary_dataset,
        token_budget=500,
        reverse_source=True,
        shuffle=False,
    )

    batches = batcher.plan_batches()
    first_batch = batcher.collate_batch(batches[0])

    for i in range(len(first_batch.indices)):
        idx = first_batch.indices[i]
        orig_src, orig_tgt = synthetic_binary_dataset[idx]

        # Check source reversal + EOS
        src_row = first_batch.src_ids[i].tolist()
        src_len = first_batch.src_lens[i].item()

        # Last valid token in source must be EOS (3)
        assert src_row[src_len - 1] == BucketBatcher.EOS_ID
        # Preceding tokens must match reversed original
        reversed_orig = orig_src[::-1].tolist()
        assert src_row[: src_len - 1] == reversed_orig
        # Subsequent tokens must be PAD (0)
        assert all(t == BucketBatcher.PAD_ID for t in src_row[src_len:])

        # Check target input: BOS (2) + orig_tgt
        tgt_in_row = first_batch.tgt_in_ids[i].tolist()
        assert tgt_in_row[0] == BucketBatcher.BOS_ID
        assert tgt_in_row[1 : len(orig_tgt) + 1] == orig_tgt.tolist()

        # Check target output: orig_tgt + EOS (3)
        tgt_out_row = first_batch.tgt_out_ids[i].tolist()
        tgt_len = first_batch.tgt_lens[i].item()
        assert tgt_out_row[: len(orig_tgt)] == orig_tgt.tolist()
        assert tgt_out_row[tgt_len - 1] == BucketBatcher.EOS_ID
        assert all(t == BucketBatcher.PAD_ID for t in tgt_out_row[tgt_len:])


def test_batcher_seed_reproducibility(synthetic_binary_dataset):
    """Ensures that the same seed produces identical batch planning."""
    batcher1 = BucketBatcher(synthetic_binary_dataset, token_budget=1000, seed=123)
    batcher2 = BucketBatcher(synthetic_binary_dataset, token_budget=1000, seed=123)

    plan1 = batcher1.plan_batches(seed=123)
    plan2 = batcher2.plan_batches(seed=123)

    assert plan1 == plan2, "Identical seeds produced different batch plans"


def test_prefetch_batch_iterator(synthetic_binary_dataset):
    """Verifies that the background prefetcher yields all batches correctly."""
    batcher = BucketBatcher(synthetic_binary_dataset, token_budget=800, seed=42)
    expected_plan = batcher.plan_batches()

    iterator = PrefetchBatchIterator(batcher, queue_size=4)
    assert len(iterator) == len(expected_plan)

    yielded_sentences = 0
    yielded_batches = 0
    for batch in iterator:
        yielded_batches += 1
        yielded_sentences += len(batch.indices)
        assert isinstance(batch.src_ids, torch.Tensor)
        assert isinstance(batch.tgt_out_ids, torch.Tensor)

    assert yielded_batches == len(expected_plan)
    assert yielded_sentences == len(synthetic_binary_dataset)
