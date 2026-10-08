"""
Unit tests for 4-Layer Bidirectional LSTM Seq2Seq model and BiLSTM Beam Search decoder.
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

from src.seq2seq.bilstm_model import BiLSTMDecoder, BiLSTMEncoder, BiLSTMSeq2SeqModel
from src.seq2seq.decoding.bilstm_beam_search import BiLSTMBeamSearchDecoder


def test_bilstm_encoder_shapes():
    vocab_size = 500
    d_model = 64
    n_layers = 4
    batch_size = 4
    max_src_len = 12

    encoder = BiLSTMEncoder(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)

    src_ids = torch.randint(4, vocab_size, (batch_size, max_src_len))
    src_lens = torch.tensor([12, 10, 8, 5])

    final_h, final_c = encoder(src_ids, src_lens)

    # Output shape after layer-wise bridge must match unidirectional decoder expectations:
    # [n_layers, batch_size, d_model]
    assert final_h.shape == (n_layers, batch_size, d_model)
    assert final_c.shape == (n_layers, batch_size, d_model)


def test_bilstm_decoder_shapes_and_step():
    vocab_size = 500
    d_model = 64
    n_layers = 4
    batch_size = 4
    max_tgt_len = 10

    decoder = BiLSTMDecoder(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    encoder_h = torch.randn(n_layers, batch_size, d_model)
    encoder_c = torch.randn(n_layers, batch_size, d_model)

    tgt_in_ids = torch.randint(4, vocab_size, (batch_size, max_tgt_len))
    tgt_in_ids[:, 0] = 2  # BOS

    outputs, (final_h, final_c) = decoder(tgt_in_ids, (encoder_h, encoder_c))
    assert outputs.shape == (batch_size, max_tgt_len, d_model)
    assert final_h.shape == (n_layers, batch_size, d_model)
    assert final_c.shape == (n_layers, batch_size, d_model)

    # Autoregressive single step test
    curr_token = torch.tensor([[2], [2], [2], [2]])
    step_out, (step_h, step_c) = decoder.step(curr_token, (encoder_h, encoder_c))
    assert step_out.shape == (batch_size, 1, d_model)
    assert step_h.shape == (n_layers, batch_size, d_model)
    assert step_c.shape == (n_layers, batch_size, d_model)


def test_bilstm_seq2seq_forward_backward_and_tied_weights():
    vocab_size = 500
    d_model = 64
    n_layers = 4
    batch_size = 4
    max_src_len = 10
    max_tgt_len = 8

    model = BiLSTMSeq2SeqModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        tie_weights=True,
    )

    # Tied weights check
    assert model.fc_out.weight is model.decoder.embedding.weight

    param_info = model.count_parameters()
    assert param_info["model_type"] == "bilstm"
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


def test_bilstm_beam_search_greedy_and_beam():
    vocab_size = 200
    d_model = 64
    n_layers = 4
    batch_size = 2

    model = BiLSTMSeq2SeqModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    decoder = BiLSTMBeamSearchDecoder(model=model, beam_size=3)

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


def test_bilstm_beam_search_length_penalty():
    model = BiLSTMSeq2SeqModel(vocab_size=100, d_model=32, n_layers=2)
    decoder = BiLSTMBeamSearchDecoder(model=model, length_penalty_alpha=0.6)

    lengths = torch.tensor([2, 5, 10, 20, 40])
    lps = decoder.compute_length_penalty(lengths)

    for i in range(len(lps) - 1):
        assert lps[i] < lps[i + 1]
