"""
Byte-Level BPE (BBPE) Tokenizer for Tiny-Seq2Seq.
Wraps Hugging Face's Rust-backed ByteLevelBPETokenizer for high-throughput multi-threaded tokenization
with guaranteed special token preservation (<pad>=0, <unk>=1, <bos>=2, <eos>=3) and 0% out-of-vocabulary rate.
"""

import os
from typing import List, Optional, Union
from tokenizers import ByteLevelBPETokenizer, Tokenizer

from utils.custom_exception import VocabMismatchError
from utils.logger import logger


class BBPETokenizer:
    """
    Byte-Level BPE Tokenizer Manager.
    Uses native Rust/C-ABI multithreaded engine to encode and decode text with 100% byte coverage.
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self.tokenizer: Optional[Tokenizer] = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Loads a trained Byte-Level BPE tokenizer JSON configuration and validates special tokens."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Byte-Level BPE model file not found: {model_path}")
        self.tokenizer = Tokenizer.from_file(model_path)
        self.model_path = model_path
        self._validate_special_tokens()
        logger.info(
            f"Loaded Byte-Level BPE tokenizer from {model_path} (Vocab size: {self.vocab_size})"
        )

    def _validate_special_tokens(self) -> None:
        """Ensures special token IDs match the fixed contract required by the model and batcher."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        pad_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        unk_id = self.tokenizer.token_to_id(self.UNK_TOKEN)
        bos_id = self.tokenizer.token_to_id(self.BOS_TOKEN)
        eos_id = self.tokenizer.token_to_id(self.EOS_TOKEN)

        if pad_id != self.PAD_ID:
            raise VocabMismatchError(f"Expected pad_id={self.PAD_ID}, got {pad_id}")
        if unk_id != self.UNK_ID:
            raise VocabMismatchError(f"Expected unk_id={self.UNK_ID}, got {unk_id}")
        if bos_id != self.BOS_ID:
            raise VocabMismatchError(f"Expected bos_id={self.BOS_ID}, got {bos_id}")
        if eos_id != self.EOS_ID:
            raise VocabMismatchError(f"Expected eos_id={self.EOS_ID}, got {eos_id}")

    @property
    def vocab_size(self) -> int:
        """Returns total vocabulary size including special tokens and byte tokens."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        input_files: Union[str, List[str]],
        output_path: str,
        vocab_size: int = 16000,
        min_frequency: int = 2,
    ) -> "BBPETokenizer":
        """
        Trains a ByteLevelBPETokenizer on input files and saves the unified JSON tokenizer.
        Guarantees special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3.
        """
        if isinstance(input_files, str):
            input_files = [input_files]

        for f in input_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Input file for training tokenizer does not exist: {f}")

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"Training Byte-Level BPE tokenizer on {len(input_files)} file(s) with vocab_size={vocab_size}..."
        )
        trainer_tok = ByteLevelBPETokenizer()
        trainer_tok.train(
            files=input_files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
        )

        trainer_tok.save(output_path)
        logger.info(f"Saved trained BBPE tokenizer to {output_path}")

        instance = cls(output_path)
        return instance

    def encode(self, text: str) -> List[int]:
        """Encodes raw text into a list of token IDs without special tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized or loaded.")
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return encoding.ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a list of raw texts into lists of token IDs in parallel using native multithreading.
        Releases the GIL for massive throughput on multi-core systems.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized or loaded.")
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        return [e.ids for e in encodings]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs into a text string, stripping special tokens if requested."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized or loaded.")
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID)]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decodes a batch of token ID lists in parallel."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized or loaded.")
        if skip_special_tokens:
            cleaned_seqs = [
                [i for i in seq if i not in (self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID)]
                for seq in sequences
            ]
        else:
            cleaned_seqs = sequences
        return self.tokenizer.decode_batch(cleaned_seqs, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """Returns the ID for a given token string."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized or loaded.")
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Returns the string representation for a given token ID."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been initialized or loaded.")
        return self.tokenizer.id_to_token(token_id)
