"""
Utils package for Tiny-Seq2Seq.
"""

from utils.custom_exception import (
    BatcherParityError,
    DataAlignmentError,
    ModelDivergenceError,
    TinySeq2SeqException,
    TokenBudgetExceededError,
    VocabMismatchError,
    VRAMExceededError,
)
from utils.helpers import (
    count_parameters,
    format_seconds,
    get_device,
    get_gpu_memory_mb,
    reset_peak_gpu_memory,
    set_seed,
)
from utils.logger import logger, setup_logger
from utils.metrics import ThroughputTracker, compute_perplexity

__all__ = [
    "logger",
    "setup_logger",
    "TinySeq2SeqException",
    "DataAlignmentError",
    "TokenBudgetExceededError",
    "VRAMExceededError",
    "ModelDivergenceError",
    "VocabMismatchError",
    "BatcherParityError",
    "set_seed",
    "get_gpu_memory_mb",
    "reset_peak_gpu_memory",
    "count_parameters",
    "format_seconds",
    "get_device",
    "compute_perplexity",
    "ThroughputTracker",
]
