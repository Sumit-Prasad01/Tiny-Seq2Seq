"""
Unit tests for Phase 8: Paper Replication Ablation Suite.
Verifies Polyak checkpoint averaging, ablation plots, and comparative utilities.
"""

import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn

from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.averaging import average_checkpoints, load_averaged_model
from src.seq2seq.training.ablation_visualizer import AblationVisualizer


@pytest.fixture
def temp_ablation_env():
    """Provides temporary environment for ablation tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_average_checkpoints_math(temp_ablation_env):
    """Verifies that Polyak checkpoint averaging calculates exact mean tensors."""
    ckpt1_path = os.path.join(temp_ablation_env, "ckpt1.pt")
    ckpt2_path = os.path.join(temp_ablation_env, "ckpt2.pt")

    t1 = torch.tensor([1.0, 3.0, 5.0])
    t2 = torch.tensor([3.0, 5.0, 7.0])
    buf = torch.tensor([1, 2, 3], dtype=torch.long)

    torch.save({"model_state_dict": {"weight": t1, "step_count": buf}}, ckpt1_path)
    torch.save({"model_state_dict": {"weight": t2, "step_count": buf}}, ckpt2_path)

    avg_state = average_checkpoints([ckpt1_path, ckpt2_path])
    expected_weight = torch.tensor([2.0, 4.0, 6.0])

    assert torch.allclose(avg_state["weight"], expected_weight)
    assert avg_state["step_count"].dtype == torch.long
    assert torch.equal(avg_state["step_count"], buf)


def test_average_checkpoints_empty_error():
    """Verifies that attempting to average an empty list raises ValueError."""
    with pytest.raises(ValueError):
        average_checkpoints([])


def test_load_averaged_model(temp_ablation_env):
    """Verifies that averaged parameters can be loaded seamlessly into Seq2SeqModel."""
    model1 = Seq2SeqModel(vocab_size=60, d_model=32, n_layers=1)
    model2 = Seq2SeqModel(vocab_size=60, d_model=32, n_layers=1)

    # Force known weights
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.fill_(1.0)
            p2.fill_(3.0)

    ckpt1 = os.path.join(temp_ablation_env, "m1.pt")
    ckpt2 = os.path.join(temp_ablation_env, "m2.pt")
    torch.save({"model_state_dict": model1.state_dict()}, ckpt1)
    torch.save({"model_state_dict": model2.state_dict()}, ckpt2)

    target_model = Seq2SeqModel(vocab_size=60, d_model=32, n_layers=1)
    load_averaged_model(target_model, [ckpt1, ckpt2])

    for p in target_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.0))


def test_ablation_visualizer_plots(temp_ablation_env):
    """Verifies that all 4 ablation charts are generated cleanly."""
    viz = AblationVisualizer(output_dir=temp_ablation_env)

    # 1. Reversal plot
    p1 = viz.plot_reversal_comparison(
        reversed_ppl=3.2,
        non_reversed_ppl=4.5,
        reversed_bleu=26.4,
        non_reversed_bleu=21.1,
    )
    assert os.path.exists(p1)
    assert os.path.getsize(p1) > 1000

    # 2. Beam width scaling plot
    p2 = viz.plot_beam_width_scaling(
        beam_sizes=[1, 2, 5, 12],
        bleu_scores=[20.1, 25.2, 26.5, 26.8],
    )
    assert os.path.exists(p2)
    assert os.path.getsize(p2) > 1000

    # 3. Depth comparison plot
    p3 = viz.plot_depth_comparison(
        layers=[2, 3, 4],
        ppl_scores=[4.1, 3.2, 3.1],
        bleu_scores=[22.5, 26.4, 26.9],
    )
    assert os.path.exists(p3)
    assert os.path.getsize(p3) > 1000

    # 4. Checkpoint averaging plot
    p4 = viz.plot_ensemble_averaging(
        best_bleu=26.4,
        avg_bleu=27.2,
    )
    assert os.path.exists(p4)
    assert os.path.getsize(p4) > 1000
