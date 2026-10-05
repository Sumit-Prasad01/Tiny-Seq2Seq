"""
Text Preprocessing and Filtering Utilities for Tiny-Seq2Seq.
Implements the data hygiene rules specified in the implementation plan:
- Casing is preserved (cased BLEU evaluation)
- Unicode whitespace normalization
- Duplicate pair removal
- Identical source-target removal (untranslated copied lines)
- Length ratio and absolute length boundary enforcement
"""

import unicodedata
from typing import Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalizes Unicode form (NFC) and standardizes whitespace while preserving casing."""
    if not text:
        return ""
    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)
    # Normalize all whitespace (including non-breaking spaces) to standard space
    text = " ".join(text.strip().split())
    return text


def is_valid_pair(
    src: str,
    tgt: str,
    min_tokens: int = 1,
    max_tokens: int = 64,
    max_length_ratio: float = 2.5,
    filter_identical: bool = True,
) -> bool:
    """
    Validates a parallel sentence pair against hygiene criteria.
    Tokens are estimated by whitespace split prior to BPE tokenization.
    """
    if not src or not tgt:
        return False

    # Check identical copies (often noise or non-translations in web crawls)
    if filter_identical and src.strip().lower() == tgt.strip().lower():
        return False

    src_words = src.split()
    tgt_words = tgt.split()

    len_src = len(src_words)
    len_tgt = len(tgt_words)

    # Length bounds
    if len_src < min_tokens or len_tgt < min_tokens:
        return False
    if len_src > max_tokens or len_tgt > max_tokens:
        return False

    # Ratio bound (checks for severe misalignments)
    ratio = max(len_src, len_tgt) / max(min(len_src, len_tgt), 1)
    if ratio > max_length_ratio:
        return False

    return True


def clean_sentence_pair(
    src: str,
    tgt: str,
    min_tokens: int = 1,
    max_tokens: int = 64,
    max_length_ratio: float = 2.5,
    filter_identical: bool = True,
) -> Optional[Tuple[str, str]]:
    """Normalizes and validates a sentence pair. Returns None if pair fails validation."""
    clean_src = normalize_text(src)
    clean_tgt = normalize_text(tgt)

    if not is_valid_pair(
        clean_src,
        clean_tgt,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        max_length_ratio=max_length_ratio,
        filter_identical=filter_identical,
    ):
        return None

    return clean_src, clean_tgt
