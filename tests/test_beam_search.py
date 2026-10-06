"""
Unit tests for Phase 7: Vectorized GPU Beam Search & Multi-Metric Evaluation.
"""

import os
import shutil
import tempfile
import pytest
import torch

from src.seq2seq.decoding.beam_search import BeamSearchDecoder
from src.seq2seq.decoding.evaluator import Seq2SeqEvaluator
from src.seq2seq.model import Seq2SeqModel


@pytest.fixture
def temp_eval_env():
    """Provides temporary environment for evaluation tests."""
    temp_dir = tempfile.mkdtemp()
    plots_dir = os.path.join(temp_dir, "plots")
    yield plots_dir, temp_dir
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_length_penalty_monotonicity():
    model = Seq2SeqModel(vocab_size=50, d_model=32, n_layers=1)
    decoder = BeamSearchDecoder(model=model, length_penalty_alpha=0.6)

    lens = torch.tensor([5, 10, 20, 40])
    lps = decoder.compute_length_penalty(lens)

    # Length penalty must increase monotonically with length
    for i in range(len(lps) - 1):
        assert lps[i] < lps[i + 1]


def test_beam_search_decoding_shapes():
    vocab_size = 50
    d_model = 32
    n_layers = 1
    model = Seq2SeqModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)

    decoder = BeamSearchDecoder(model=model, beam_size=3)

    B = 2
    max_src = 6
    src_ids = torch.tensor([[10, 20, 30, 3, 0, 0], [15, 25, 35, 45, 3, 0]], dtype=torch.long)
    src_lens = torch.tensor([4, 5], dtype=torch.long)

    # Test Greedy (K = 1)
    hyp_greedy = decoder.decode_batch(src_ids, src_lens, beam_size=1, max_len=10)
    assert len(hyp_greedy) == B
    for hyp in hyp_greedy:
        assert isinstance(hyp, list)
        # Must not contain special tokens
        assert 0 not in hyp
        assert 2 not in hyp
        assert 3 not in hyp

    # Test Beam Search (K = 3)
    hyp_beam = decoder.decode_batch(src_ids, src_lens, beam_size=3, max_len=10)
    assert len(hyp_beam) == B
    for hyp in hyp_beam:
        assert isinstance(hyp, list)
        assert 0 not in hyp
        assert 2 not in hyp
        assert 3 not in hyp


def test_evaluator_plot_generation(temp_eval_env):
    plots_dir, _ = temp_eval_env
    model = Seq2SeqModel(vocab_size=50, d_model=32, n_layers=1)
    decoder = BeamSearchDecoder(model=model)
    evaluator = Seq2SeqEvaluator(decoder=decoder, tokenizer=None, plots_dir=plots_dir)

    bucket_scores = {
        "short (<15)": 15.4,
        "medium (15-30)": 12.8,
        "long (>30)": 8.2,
    }

    plot_path = evaluator.plot_length_vs_bleu(bucket_scores, beam_size=5)
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 1000
