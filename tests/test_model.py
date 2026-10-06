"""
Unit tests for Phase 4: Seq2Seq Architecture & Parameter Counts.
Validates:
- Exact parameter counts (~29.0M tied, ~37.2M untied)
- Uniform weight initialization in [-0.08, 0.08]
- Padding invariance of encoder final states (packed sequences)
- Forward and backward pass execution on GPU with mixed precision
"""

import pytest
import torch

from src.seq2seq.model import Seq2SeqModel


def test_parameter_counts_tied_and_untied():
    vocab_size = 16000
    d_model = 512
    n_layers = 3

    # 1. Tied weights model
    model_tied = Seq2SeqModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        tie_weights=True,
    )
    counts_tied = model_tied.count_parameters()
    # Expected: 2 * (3 * 2.101M) + 2 * (16k * 512) + 16k bias = 29,007,488
    assert counts_tied["trainable_unique_parameters"] == 29007488, (
        f"Expected 29,007,488 params tied, got {counts_tied['trainable_unique_parameters']}"
    )

    # 2. Untied weights model
    model_untied = Seq2SeqModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        tie_weights=False,
    )
    counts_untied = model_untied.count_parameters()
    # Expected: 29,007,488 + (16k * 512) = 37,199,488
    assert counts_untied["trainable_unique_parameters"] == 37199488, (
        f"Expected 37,199,488 params untied, got {counts_untied['trainable_unique_parameters']}"
    )


def test_uniform_initialization_bounds():
    bound = 0.08
    model = Seq2SeqModel(vocab_size=500, d_model=128, n_layers=2, init_uniform_bound=bound)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.min().item() >= -bound - 1e-5, f"{name} min value violated"
            assert param.max().item() <= bound + 1e-5, f"{name} max value violated"

    # Padding index row must be zeros
    assert torch.all(model.encoder.embedding.weight[0] == 0)
    assert torch.all(model.decoder.embedding.weight[0] == 0)


def test_packed_sequence_padding_invariance():
    """
    Validates that padding tokens do NOT alter the final encoder hidden state.
    Sentence [10, 20, 3] alone vs padded with [0, 0, 0] must produce identical states.
    """
    model = Seq2SeqModel(vocab_size=100, d_model=64, n_layers=2)
    model.eval()

    sent = torch.tensor([[10, 20, 3]], dtype=torch.long)
    sent_len = torch.tensor([3], dtype=torch.long)

    # Padded batch: sentence 1 has 3 tokens, sentence 2 has 5 tokens
    padded_batch = torch.tensor([
        [10, 20, 3, 0, 0],
        [30, 40, 50, 60, 3],
    ], dtype=torch.long)
    batch_lens = torch.tensor([3, 5], dtype=torch.long)

    with torch.no_grad():
        h_single, c_single = model.encoder(sent, sent_len)
        h_batch, c_batch = model.encoder(padded_batch, batch_lens)

    # State of sentence 0 in batch must match single sentence state
    assert torch.allclose(h_single[:, 0, :], h_batch[:, 0, :], atol=1e-5)
    assert torch.allclose(c_single[:, 0, :], c_batch[:, 0, :], atol=1e-5)


def test_forward_backward_gpu_amp():
    """Validates full forward/backward pass with AMP FP16 on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqModel(vocab_size=500, d_model=128, n_layers=2, tie_weights=True).to(device)

    batch_size = 4
    max_src = 15
    max_tgt = 12

    src_ids = torch.randint(1, 400, (batch_size, max_src), device=device)
    src_lens = torch.tensor([15, 12, 10, 8], device=device)
    tgt_in_ids = torch.randint(1, 400, (batch_size, max_tgt), device=device)
    tgt_out_ids = torch.randint(1, 400, (batch_size, max_tgt), device=device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        logits = model(src_ids, src_lens, tgt_in_ids)
        assert logits.shape == (batch_size, max_tgt, 500)
        loss = criterion(logits.view(-1, 500), tgt_out_ids.view(-1))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not torch.isnan(loss)
    assert loss.item() > 0
