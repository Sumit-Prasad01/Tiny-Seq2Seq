"""
Tokenizer Manager for Tiny-Seq2Seq.
Wraps SentencePiece BPE model training, serialization, encoding, and detokenization
with guaranteed special token preservation (<pad>=0, <unk>=1, <bos>=2, <eos>=3).
"""

import os
from typing import List, Optional, Union
import sentencepiece as spm

from utils.custom_exception import VocabMismatchError
from utils.logger import logger


class TokenizerManager:
    """Manages SentencePiece BPE model lifecycle and token conversions."""

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Loads a trained SentencePiece model and validates special token IDs."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.sp.load(model_path)
        self.model_path = model_path
        self._validate_special_tokens()
        logger.info(f"Loaded SentencePiece model from {model_path} (Vocab size: {self.vocab_size})")

    def _validate_special_tokens(self) -> None:
        """Ensures special token IDs match the fixed contract required by the model and batcher."""
        if self.sp.pad_id() != self.PAD_ID:
            raise VocabMismatchError(f"Expected pad_id={self.PAD_ID}, got {self.sp.pad_id()}")
        if self.sp.unk_id() != self.UNK_ID:
            raise VocabMismatchError(f"Expected unk_id={self.UNK_ID}, got {self.sp.unk_id()}")
        if self.sp.bos_id() != self.BOS_ID:
            raise VocabMismatchError(f"Expected bos_id={self.BOS_ID}, got {self.sp.bos_id()}")
        if self.sp.eos_id() != self.EOS_ID:
            raise VocabMismatchError(f"Expected eos_id={self.EOS_ID}, got {self.sp.eos_id()}")

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @classmethod
    def train(
        cls,
        input_files: Union[str, List[str]],
        model_prefix: str,
        vocab_size: int = 16000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        max_sentence_length: int = 4192,
    ) -> "TokenizerManager":
        """
        Trains a SentencePiece model with fixed special tokens:
        <pad>=0, <unk>=1, <bos>=2, <eos>=3.
        """
        if isinstance(input_files, list):
            input_arg = ",".join(input_files)
        else:
            input_arg = input_files

        os.makedirs(os.path.dirname(os.path.abspath(model_prefix)), exist_ok=True)

        train_cmd = (
            f"--input={input_arg} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type={model_type} "
            f"--character_coverage={character_coverage} "
            f"--pad_id={cls.PAD_ID} "
            f"--unk_id={cls.UNK_ID} "
            f"--bos_id={cls.BOS_ID} "
            f"--eos_id={cls.EOS_ID} "
            f"--pad_piece={cls.PAD_TOKEN} "
            f"--unk_piece={cls.UNK_TOKEN} "
            f"--bos_piece={cls.BOS_TOKEN} "
            f"--eos_piece={cls.EOS_TOKEN} "
            f"--max_sentence_length={max_sentence_length}"
        )

        logger.info(f"Training SentencePiece BPE model with vocab_size={vocab_size}...")
        spm.SentencePieceTrainer.train(train_cmd)

        model_file = f"{model_prefix}.model"
        instance = cls(model_file)
        return instance

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        reverse: bool = False,
    ) -> List[int]:
        """Encodes text to a list of token IDs with optional BOS/EOS and reversal."""
        ids = self.sp.encode(text, out_type=int)
        if reverse:
            ids = ids[::-1]
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Decodes a list of token IDs back into a detokenized text string."""
        if strip_special:
            special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}
            filtered = [t for t in ids if t not in special_ids]
        else:
            filtered = ids
        return self.sp.decode(filtered)
