"""
C++ Batch Builder Wrapper for Tiny-Seq2Seq.
Provides a drop-in replacement for BucketBatcher powered by the compiled
seq2seq_c_batcher pybind11 module with automatic pure-Python fallback.
"""

from typing import List, Optional
import numpy as np
import torch

from src.seq2seq.data.batcher import Batch, BucketBatcher
from src.seq2seq.data.dataset import ParallelBinaryDataset
from utils.logger import logger

_CPP_AVAILABLE = False
try:
    import seq2seq_c_batcher
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


def is_cpp_batcher_available() -> bool:
    """Returns True if the compiled C++ batcher module is accessible."""
    return _CPP_AVAILABLE


class CppBucketBatcher:
    """
    Drop-in replacement for BucketBatcher powered by compiled C++ (pybind11).
    Releases the GIL during sorting, bucketing, and tensor generation.
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

        if not _CPP_AVAILABLE:
            logger.warning(
                "C++ batch builder 'seq2seq_c_batcher' is not compiled or available. "
                "Falling back to pure-Python BucketBatcher."
            )
            self._python_fallback = BucketBatcher(
                dataset=dataset,
                token_budget=token_budget,
                max_seq_len=max_seq_len,
                reverse_source=reverse_source,
                shuffle=shuffle,
                seed=seed,
            )
            self._cpp_builder = None
        else:
            # Memory-mapped NumPy arrays to pass to C++
            src_arr = np.asarray(dataset.src_reader.data)
            src_off = np.asarray(dataset.src_reader.offsets)
            tgt_arr = np.asarray(dataset.tgt_reader.data)
            tgt_off = np.asarray(dataset.tgt_reader.offsets)

            self._cpp_builder = seq2seq_c_batcher.CppBatchBuilder(
                src_arr, src_off, tgt_arr, tgt_off
            )
            self._python_fallback = None

    def plan_batches(self, seed: Optional[int] = None) -> List[List[int]]:
        """Plans batches across the dataset, releasing GIL in C++."""
        if self._cpp_builder is None:
            return self._python_fallback.plan_batches(seed)

        current_seed = (self.seed + self.epoch) if seed is None else seed
        return self._cpp_builder.plan_batches(
            token_budget=self.token_budget,
            shuffle=self.shuffle,
            seed=current_seed,
        )

    def collate_batch(self, batch_indices: List[int]) -> Batch:
        """Assembles a batch in C++, returning PyTorch tensors."""
        if self._cpp_builder is None:
            return self._python_fallback.collate_batch(batch_indices)

        batch_dict = self._cpp_builder.build_batch(
            indices=batch_indices,
            reverse_source=self.reverse_source,
            pad_id=self.PAD_ID,
            bos_id=self.BOS_ID,
            eos_id=self.EOS_ID,
        )

        return Batch(
            src_ids=torch.from_numpy(batch_dict["src_ids"]),
            src_lens=torch.from_numpy(batch_dict["src_lens"]),
            tgt_in_ids=torch.from_numpy(batch_dict["tgt_in_ids"]),
            tgt_out_ids=torch.from_numpy(batch_dict["tgt_out_ids"]),
            tgt_lens=torch.from_numpy(batch_dict["tgt_lens"]),
            num_tokens=int(batch_dict["num_tokens"]),
            indices=batch_indices,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self._python_fallback is not None:
            self._python_fallback.set_epoch(epoch)
