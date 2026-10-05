"""
Custom Exceptions for Tiny-Seq2Seq.
Defines domain-specific error types for dataset validation, memory guards, and model stability.
"""


class TinySeq2SeqException(Exception):
    """Base exception for all Tiny-Seq2Seq errors."""
    pass


class DataAlignmentError(TinySeq2SeqException):
    """Raised when source and target sentence counts or offsets are misaligned."""
    pass


class TokenBudgetExceededError(TinySeq2SeqException):
    """Raised when a single sentence or batch exceeds the hardware token budget."""
    pass


class VRAMExceededError(TinySeq2SeqException):
    """Raised when GPU memory allocation exceeds the safety threshold for the 4 GB budget."""
    pass


class ModelDivergenceError(TinySeq2SeqException):
    """Raised when training loss produces NaN, Inf, or diverges unexpectedly."""
    pass


class VocabMismatchError(TinySeq2SeqException):
    """Raised when tokenizer special token IDs or sizes mismatch model expectations."""
    pass


class BatcherParityError(TinySeq2SeqException):
    """Raised when C++ and Python batcher implementations produce mismatched tensors."""
    pass
