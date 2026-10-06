"""
Length-Bucketed Batch Builder and Prefetcher for Tiny-Seq2Seq.
Implements the paper's batching strategy:
- Groups sentences of similar length to minimize padding waste (2x speedup)
- Enforces dynamic token budget (e.g. 4,000 tokens/batch)
- Implements exact source reversal convention (source reversed + <eos>)
- Collate tensors with <bos>, <eos>, and <pad> IDs
- Multi-threaded producer-consumer prefetch queue for zero GPU starvation
"""

import queue
import random
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
import numpy as np
import torch

from src.seq2seq.data.dataset import ParallelBinaryDataset
from utils.logger import logger


@dataclass
class Batch:
    """Represents a collated batch ready for training or evaluation."""
    src_ids: torch.Tensor       # [B, max_src_len] (reversed + EOS + PAD)
    src_lens: torch.Tensor      # [B] (actual length including EOS)
    tgt_in_ids: torch.Tensor    # [B, max_tgt_len] (BOS + target tokens + PAD)
    tgt_out_ids: torch.Tensor   # [B, max_tgt_len] (target tokens + EOS + PAD)
    tgt_lens: torch.Tensor      # [B] (actual target length including EOS)
    num_tokens: int             # Total non-pad target tokens (for loss normalization)
    indices: List[int]          # Original sentence indices in the dataset

    def to(self, device: torch.device) -> "Batch":
        """Transfers batch tensors to target device (e.g. CUDA)."""
        return Batch(
            src_ids=self.src_ids.to(device, non_blocking=True),
            src_lens=self.src_lens.to(device, non_blocking=True),
            tgt_in_ids=self.tgt_in_ids.to(device, non_blocking=True),
            tgt_out_ids=self.tgt_out_ids.to(device, non_blocking=True),
            tgt_lens=self.tgt_lens.to(device, non_blocking=True),
            num_tokens=self.num_tokens,
            indices=self.indices,
        )


class BucketBatcher:
    """
    Plans and builds length-bucketed batches adhering to a strict token budget.
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(
        self,
        dataset: ParallelBinaryDataset,
        token_budget: int = 4000,
        max_seq_len: int = 64,
        reverse_source: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.token_budget = token_budget
        self.max_seq_len = max_seq_len
        self.reverse_source = reverse_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Precompute sentence lengths
        self.src_lens, self.tgt_lens = dataset.get_all_lengths()

    def plan_batches(self, seed: Optional[int] = None) -> List[List[int]]:
        """
        Plans batches for one epoch:
        1. Filters and sorts sentence indices by sequence length (with randomized tie-breaking)
        2. Groups sentences into batches until the token budget is reached
        3. Shuffles the execution order of batches
        """
        current_seed = (self.seed + self.epoch) if seed is None else seed
        rng = np.random.RandomState(current_seed)

        n = len(self.dataset)
        indices = np.arange(n)

        # Max length for each pair including special tokens (+1 for EOS/BOS)
        pair_max_lens = np.maximum(self.src_lens + 1, self.tgt_lens + 1)

        if self.shuffle:
            # Sort with random perturbation to break ties randomly
            noise = rng.uniform(0, 0.999, size=n)
            sort_keys = pair_max_lens + noise
            sorted_indices = indices[np.argsort(sort_keys)]
        else:
            sorted_indices = indices[np.argsort(pair_max_lens)]

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_src = 0
        current_max_tgt = 0

        for idx in sorted_indices:
            s_len = int(self.src_lens[idx]) + 1  # +1 for EOS
            t_len = int(self.tgt_lens[idx]) + 1  # +1 for EOS/BOS

            # Potential batch dimensions with this sentence included
            new_max_src = max(current_max_src, s_len)
            new_max_tgt = max(current_max_tgt, t_len)
            new_batch_size = len(current_batch) + 1
            new_tokens = max(new_max_src, new_max_tgt) * new_batch_size

            if current_batch and new_tokens > self.token_budget:
                # Close current batch
                batches.append(current_batch)
                current_batch = [int(idx)]
                current_max_src = s_len
                current_max_tgt = t_len
            else:
                current_batch.append(int(idx))
                current_max_src = new_max_src
                current_max_tgt = new_max_tgt

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            # Shuffle batch execution order
            rng.shuffle(batches)

        return batches

    def collate_batch(self, batch_indices: List[int]) -> Batch:
        """
        Fetches and collates sentences for a list of indices into padded PyTorch tensors.
        Implements exact source reversal and BOS/EOS wrapping.
        """
        batch_size = len(batch_indices)
        src_raw = []
        tgt_raw = []

        max_src_len = 0
        max_tgt_len = 0

        for idx in batch_indices:
            s, t = self.dataset[idx]
            src_raw.append(s)
            tgt_raw.append(t)
            max_src_len = max(max_src_len, len(s) + 1)  # +1 for EOS
            max_tgt_len = max(max_tgt_len, len(t) + 1)  # +1 for EOS/BOS

        src_ids = torch.zeros((batch_size, max_src_len), dtype=torch.long)
        tgt_in_ids = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)
        tgt_out_ids = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)
        src_lens = torch.zeros(batch_size, dtype=torch.long)
        tgt_lens = torch.zeros(batch_size, dtype=torch.long)

        num_tokens = 0

        for i in range(batch_size):
            s = src_raw[i]
            t = tgt_raw[i]

            s_len = len(s)
            t_len = len(t)

            # Source Reversal Convention:
            # Reversed tokens, followed by EOS
            if self.reverse_source:
                s_tokens = s[::-1].tolist()
            else:
                s_tokens = s.tolist()
            s_tokens.append(self.EOS_ID)

            src_ids[i, : len(s_tokens)] = torch.tensor(s_tokens, dtype=torch.long)
            src_lens[i] = len(s_tokens)

            # Target:
            # Decoder Input: BOS + target tokens
            t_in = [self.BOS_ID] + t.tolist()
            tgt_in_ids[i, : len(t_in)] = torch.tensor(t_in, dtype=torch.long)

            # Decoder Output (Ground truth): target tokens + EOS
            t_out = t.tolist() + [self.EOS_ID]
            tgt_out_ids[i, : len(t_out)] = torch.tensor(t_out, dtype=torch.long)
            tgt_lens[i] = len(t_out)
            num_tokens += len(t_out)

        return Batch(
            src_ids=src_ids,
            src_lens=src_lens,
            tgt_in_ids=tgt_in_ids,
            tgt_out_ids=tgt_out_ids,
            tgt_lens=tgt_lens,
            num_tokens=num_tokens,
            indices=batch_indices,
        )

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch index for deterministic shuffling progression."""
        self.epoch = epoch


class PrefetchBatchIterator:
    """
    Producer-consumer prefetcher running on a background thread.
    Keeps a queue of prepared batches so the GPU training loop is never stalled.
    """

    def __init__(
        self,
        batcher: BucketBatcher,
        queue_size: int = 8,
        device: Optional[torch.device] = None,
    ) -> None:
        self.batcher = batcher
        self.queue_size = queue_size
        self.device = device
        self.batches_plan = self.batcher.plan_batches()
        self.num_batches = len(self.batches_plan)

        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _producer(self) -> None:
        """Background worker thread populating the queue."""
        for batch_indices in self.batches_plan:
            if self._stop_event.is_set():
                break
            batch = self.batcher.collate_batch(batch_indices)
            if self.device is not None:
                batch = batch.to(self.device)
            # Blocks if queue is full until consumer dequeues
            while not self._stop_event.is_set():
                try:
                    self._queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

        # Signal completion
        self._queue.put(None)

    def __iter__(self) -> Iterator[Batch]:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                yield item
        finally:
            self._stop_event.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)

    def __len__(self) -> int:
        return self.num_batches
