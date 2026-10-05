"""
Binary Serializer for Tiny-Seq2Seq.
Packs tokenized corpora into flat 1D uint16 binary files (.bin) and int64 index offsets (.idx).
Provides fast, zero-copy memory-mapped access for high-throughput batching.
"""

import os
from typing import Iterator, List, Tuple
import numpy as np

from utils.custom_exception import DataAlignmentError
from utils.logger import logger


class BinaryWriter:
    """Writes token lists into flat .bin and .idx files incrementally."""

    def __init__(self, bin_path: str, idx_path: str, dtype: np.dtype = np.uint16) -> None:
        self.bin_path = bin_path
        self.idx_path = idx_path
        self.dtype = dtype

        os.makedirs(os.path.dirname(os.path.abspath(bin_path)), exist_ok=True)
        self.bin_file = open(bin_path, "wb")
        self.offsets = [0]
        self.total_tokens = 0
        self.num_sentences = 0

    def write_sentence(self, token_ids: List[int]) -> None:
        """Appends one sentence's token IDs to the binary file and records offset."""
        arr = np.array(token_ids, dtype=self.dtype)
        self.bin_file.write(arr.tobytes())
        self.total_tokens += len(token_ids)
        self.offsets.append(self.total_tokens)
        self.num_sentences += 1

    def close(self) -> Tuple[int, int]:
        """Flushes binary file and writes the offset index array to disk."""
        self.bin_file.close()
        idx_arr = np.array(self.offsets, dtype=np.int64)
        with open(self.idx_path, "wb") as f:
            f.write(idx_arr.tobytes())

        logger.info(
            f"Saved binary dataset to {self.bin_path} ({self.num_sentences} sentences, "
            f"{self.total_tokens} tokens, size: {os.path.getsize(self.bin_path) / (1024**2):.2f} MB)"
        )
        return self.num_sentences, self.total_tokens


class BinaryReader:
    """Reads flat binary datasets using numpy.memmap for zero-copy access."""

    def __init__(self, bin_path: str, idx_path: str, dtype: np.dtype = np.uint16) -> None:
        if not os.path.exists(bin_path) or not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing binary files: {bin_path} or {idx_path}")

        self.bin_path = bin_path
        self.idx_path = idx_path
        self.dtype = dtype

        # Load index offsets
        self.offsets = np.fromfile(idx_path, dtype=np.int64)
        self.num_sentences = len(self.offsets) - 1

        # Memory-map the flat binary token array
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")

    def __len__(self) -> int:
        return self.num_sentences

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.num_sentences:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_sentences} sentences")
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        return self.data[start:end]

    def get_sentence_length(self, idx: int) -> int:
        """Returns length of sentence without copying tokens."""
        return int(self.offsets[idx + 1] - self.offsets[idx])
