"""
Unit tests for 4-Layer GRU Seq2Seq model, Byte-Level BPE tokenizer, and GRU Beam Search decoder.
"""

import os
import shutil
import sys
import tempfile

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn

from src.seq2seq.data.bbpe_tokenizer import BBPETokenizer
from src.seq2seq.decoding.gru_beam_search import GRUBeamSearchDecoder
from src.seq2seq.gru_model import GRUDecoder, GRUEncoder, GRUSeq2SeqModel


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_bbpe_tokenizer_training_and_special_tokens(temp_dir):
    corpus_path = os.path.join(temp_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("Hello world!\nThis is a Byte-Level BPE test.\nBonjour le monde.\n")

    model_path = os.path.join(temp_dir, "bbpe.json")
    tok = BBPETokenizer.train(corpus_path, model_path, vocab_size=300)

    # Special token contract
    assert tok.token_to_id("<pad>") == 0
    assert tok.token_to_id("<unk>") == 1
    assert tok.token_to_id("<bos>") == 2
    assert tok.token_to_id("<eos>") == 3
    assert tok.vocab_size >= 260  # At least 256 bytes + 4 specials

    # Encoding & Decoding round-trip
    text = "Hello world!"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert decoded == text

    # Batch methods
    batch_texts = ["Hello world!", "Bonjour le monde."]
    batch_ids = tok.encode_batch(batch_texts)
    assert len(batch_ids) == 2
    batch_decoded = tok.decode_batch(batch_ids)
    assert batch_decoded == batch_texts

    # Save and reload
    reloaded = BBPETokenizer(model_path)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.decode(ids) == text


def test_gru_encoder_shapes():
    vocab_size = 500
    d_model = 64
    n_layers = 4
    batch_size = 4
    max_src_len = 12

    encoder = GRUEncoder(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)

    src_ids = torch.randint(4, vocab_size, (batch_size, max_src_len))
    src_lens = torch.tensor([12, 10, 8, 5])

    final_h = encoder(src_ids, src_lens)

    # Output shape: [n_layers, batch_size, d_model]
    assert final_h.shape == (n_layers, batch_size, d_model)


def test_gru_decoder_shapes():
    vocab_size = 500
    d_model = 64
    n_layers = 4
    batch_size = 4
    max_tgt_len = 10

    decoder = GRUDecoder(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    encoder_state = torch.randn(n_layers, batch_size, d_model)

    tgt_in_ids = torch.randint(4, vocab_size, (batch_size, max_tgt_len))
    tgt_in_ids[:, 0] = 2  # BOS

    outputs, final_h = decoder(tgt_in_ids, encoder_state)
    assert outputs.shape == (batch_size, max_tgt_len, d_model)
    assert final_h.shape == (n_layers, batch_size, d_model)

    # Test single step autoregressive decoding
    curr_token = torch.tensor([[2], [2], [2], [2]])
    step_out, step_h = decoder.step(curr_token, encoder_state)
    assert step_out.shape == (batch_size, 1, d_model)
    assert step_h.shape == (n_layers, batch_size, d_model)


def test_gru_seq2seq_forward_and_backward():
    vocab_size = 500
    d_model = 64
    n_layers = 4
    batch_size = 4
    max_src_len = 10
    max_tgt_len = 8

    model = GRUSeq2SeqModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        tie_weights=True,
    )

    # Tied weights check
    assert model.fc_out.weight is model.decoder.embedding.weight

    param_info = model.count_parameters()
    assert param_info["model_type"] == "gru"
    assert param_info["trainable_unique_parameters"] > 0

    src_ids = torch.randint(4, vocab_size, (batch_size, max_src_len))
    src_lens = torch.tensor([10, 8, 7, 5])
    tgt_in_ids = torch.randint(4, vocab_size, (batch_size, max_tgt_len))
    tgt_in_ids[:, 0] = 2  # BOS

    logits = model(src_ids, src_lens, tgt_in_ids)
    assert logits.shape == (batch_size, max_tgt_len, vocab_size)

    # Backward pass and gradient check
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"


def test_gru_beam_search_greedy_and_beam():
    vocab_size = 200
    d_model = 64
    n_layers = 4
    batch_size = 2

    model = GRUSeq2SeqModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    decoder = GRUBeamSearchDecoder(model=model, beam_size=3)

    src_ids = torch.tensor([[10, 20, 30, 3, 0], [15, 25, 35, 45, 3]], dtype=torch.long)
    src_lens = torch.tensor([4, 5], dtype=torch.long)

    # Test Greedy (K=1)
    greedy_results = decoder.decode_batch(src_ids, src_lens, beam_size=1, max_len=10)
    assert len(greedy_results) == batch_size
    for hyp in greedy_results:
        assert isinstance(hyp, list)
        assert 0 not in hyp  # No PAD
        assert 2 not in hyp  # No BOS

    # Test Beam Search (K=3)
    beam_results = decoder.decode_batch(src_ids, src_lens, beam_size=3, max_len=10)
    assert len(beam_results) == batch_size
    for hyp in beam_results:
        assert isinstance(hyp, list)
        assert 0 not in hyp
        assert 2 not in hyp


def test_gru_beam_search_length_penalty():
    model = GRUSeq2SeqModel(vocab_size=100, d_model=32, n_layers=2)
    decoder = GRUBeamSearchDecoder(model=model, length_penalty_alpha=0.6)

    lengths = torch.tensor([2, 5, 10, 20, 40])
    lps = decoder.compute_length_penalty(lengths)

    # Must increase monotonically
    for i in range(len(lps) - 1):
        assert lps[i] < lps[i + 1]
