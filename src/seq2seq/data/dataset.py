"""
Parallel Binary Dataset for Tiny-Seq2Seq.
Memory-maps source and target binary token arrays (.bin) and index offsets (.idx)
for zero-copy random access and fast length querying.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.seq2seq.data.binary_serializer import BinaryReader
from utils.custom_exception import DataAlignmentError
from utils.logger import logger


class ParallelBinaryDataset:
    """
    Pairs source and target memory-mapped binary files.
    Allows O(1) retrieval of sentence lengths and token sequences.
    """

    def __init__(
        self,
        src_bin: str,
        src_idx: str,
        tgt_bin: str,
        tgt_idx: str,
        dtype: np.dtype = np.uint16,
    ) -> None:
        self.src_reader = BinaryReader(src_bin, src_idx, dtype=dtype)
        self.tgt_reader = BinaryReader(tgt_bin, tgt_idx, dtype=dtype)

        if len(self.src_reader) != len(self.tgt_reader):
            raise DataAlignmentError(
                f"Source ({len(self.src_reader)}) and target ({len(self.tgt_reader)}) "
                f"sentence counts mismatch."
            )

        self.num_sentences = len(self.src_reader)
        logger.info(f"Loaded ParallelBinaryDataset with {self.num_sentences} sentence pairs.")

    def __len__(self) -> int:
        return self.num_sentences

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.src_reader[idx], self.tgt_reader[idx]

    def get_lengths(self, idx: int) -> Tuple[int, int]:
        """Returns (src_len, tgt_len) without reading or copying token arrays."""
        return self.src_reader.get_sentence_length(idx), self.tgt_reader.get_sentence_length(idx)

    def get_all_lengths(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns full arrays of lengths for fast length bucketing."""
        src_offsets = self.src_reader.offsets
        tgt_offsets = self.tgt_reader.offsets
        src_lens = np.diff(src_offsets)
        tgt_lens = np.diff(tgt_offsets)
        return src_lens, tgt_lens
