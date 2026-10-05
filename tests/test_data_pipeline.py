"""
Unit tests for Phase 2: Data Pipeline, Tokenization & Binary Serialization.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.seq2seq.data.preprocessor import normalize_text, is_valid_pair, clean_sentence_pair
from src.seq2seq.data.tokenizer_manager import TokenizerManager
from src.seq2seq.data.binary_serializer import BinaryWriter, BinaryReader


@pytest.fixture
def temp_data_env():
    """Provides an isolated temporary directory for data pipeline tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_text_normalization():
    raw = "  Hello \t  world \u00a0 from  Seq2Seq! \n\n"
    normalized = normalize_text(raw)
    assert normalized == "Hello world from Seq2Seq!"


def test_pair_validation_filtering():
    # Valid pair
    assert is_valid_pair("Hello world", "Bonjour le monde", max_length_ratio=2.5)

    # Identical pair (should be filtered out)
    assert not is_valid_pair("Same text", "Same text", filter_identical=True)

    # Extreme length ratio (misaligned pair)
    assert not is_valid_pair("Short", "This is a ridiculously long target sentence with way too many words", max_length_ratio=2.5)

    # Empty pair
    assert not is_valid_pair("", "Non empty")
    assert not is_valid_pair("   ", "   ")


def test_tokenizer_training_and_special_tokens(temp_data_env):
    corpus_file = os.path.join(temp_data_env, "sample_corpus.txt")
    sample_lines = [
        "The quick brown fox jumps over the lazy dog.",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Sequence to sequence learning is an elegant machine translation approach.",
        "L'apprentissage séquence à séquence est une approche élégante de traduction.",
        "Recurrent neural networks and long short term memory architectures.",
        "Réseaux de neurones récurrents et architectures à mémoire à long et court terme.",
    ] * 20

    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_lines))

    model_prefix = os.path.join(temp_data_env, "test_spm")
    tokenizer = TokenizerManager.train(
        input_files=corpus_file,
        model_prefix=model_prefix,
        vocab_size=200,
        model_type="bpe",
    )

    # Verify special token IDs
    assert tokenizer.sp.pad_id() == 0
    assert tokenizer.sp.unk_id() == 1
    assert tokenizer.sp.bos_id() == 2
    assert tokenizer.sp.eos_id() == 3

    # Test encoding and decoding
    text = "The quick brown fox jumps"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded.strip() == text

    # Test special token wrapping
    wrapped = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert wrapped[0] == TokenizerManager.BOS_ID
    assert wrapped[-1] == TokenizerManager.EOS_ID

    # Test source reversal
    reversed_tokens = tokenizer.encode(text, reverse=True)
    assert reversed_tokens == tokens[::-1]


def test_binary_serializer_roundtrip(temp_data_env):
    bin_file = os.path.join(temp_data_env, "dataset.bin")
    idx_file = os.path.join(temp_data_env, "dataset.idx")

    sentences = [
        [10, 20, 30, 40],
        [50, 60],
        [70, 80, 90, 100, 110],
    ]

    # Write
    writer = BinaryWriter(bin_file, idx_file, dtype=np.uint16)
    for sent in sentences:
        writer.write_sentence(sent)
    num_sentences, total_tokens = writer.close()

    assert num_sentences == 3
    assert total_tokens == 11

    # Read back via memmap
    reader = BinaryReader(bin_file, idx_file, dtype=np.uint16)
    assert len(reader) == 3
    assert reader.get_sentence_length(0) == 4
    assert reader.get_sentence_length(1) == 2
    assert reader.get_sentence_length(2) == 5

    assert list(reader[0]) == sentences[0]
    assert list(reader[1]) == sentences[1]
    assert list(reader[2]) == sentences[2]
