"""
Unit tests for Phase 6: Production Training Loop, Checkpointing, and Visualizations.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.seq2seq.data.batcher import BucketBatcher
from src.seq2seq.data.binary_serializer import BinaryWriter
from src.seq2seq.data.dataset import ParallelBinaryDataset
from src.seq2seq.model import Seq2SeqModel
from src.seq2seq.training.scheduler import get_warmup_cosine_scheduler
from src.seq2seq.training.trainer import Seq2SeqTrainer
from src.seq2seq.training.visualizer import TrainingVisualizer


@pytest.fixture
def temp_training_env():
    """Provides a temporary environment with synthetic data and dirs."""
    temp_dir = tempfile.mkdtemp()
    ckpt_dir = os.path.join(temp_dir, "checkpoints")
    plots_dir = os.path.join(temp_dir, "plots")

    src_bin = os.path.join(temp_dir, "train.src.bin")
    src_idx = os.path.join(temp_dir, "train.src.idx")
    tgt_bin = os.path.join(temp_dir, "train.tgt.bin")
    tgt_idx = os.path.join(temp_dir, "train.tgt.idx")

    src_w = BinaryWriter(src_bin, src_idx)
    tgt_w = BinaryWriter(tgt_bin, tgt_idx)

    for i in range(50):
        src_w.write_sentence([10, 20, 30, 40])
        tgt_w.write_sentence([50, 60, 70])

    src_w.close()
    tgt_w.close()

    dataset = ParallelBinaryDataset(src_bin, src_idx, tgt_bin, tgt_idx)
    yield dataset, ckpt_dir, plots_dir, temp_dir
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_warmup_cosine_scheduler():
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=0.001)
    warmup_steps = 100
    total_steps = 1000
    min_lr_ratio = 0.05

    scheduler = get_warmup_cosine_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio
    )

    # Step 0: Initial LR
    assert round(optimizer.param_groups[0]["lr"], 6) == 0.0

    # Step at warmup midpoint
    for _ in range(50):
        scheduler.step()
    assert 0.0004 <= optimizer.param_groups[0]["lr"] <= 0.0006

    # Step to warmup peak
    for _ in range(50):
        scheduler.step()
    assert round(optimizer.param_groups[0]["lr"], 4) == 0.001

    # Step to completion
    for _ in range(900):
        scheduler.step()
    expected_final = 0.001 * min_lr_ratio
    assert round(optimizer.param_groups[0]["lr"], 6) == round(expected_final, 6)


def test_visualizer_plot_generation(temp_training_env):
    _, _, plots_dir, _ = temp_training_env
    visualizer = TrainingVisualizer(output_dir=plots_dir)

    p1 = visualizer.plot_loss_and_perplexity(
        train_losses=[3.5, 2.1, 1.2],
        dev_losses=[3.8, 2.4, 1.5],
        train_ppls=[33.1, 8.2, 3.3],
        dev_ppls=[44.7, 11.0, 4.5],
    )
    assert os.path.exists(p1) and os.path.getsize(p1) > 1000

    p2 = visualizer.plot_hardware_and_throughput(
        steps=[1, 2, 3, 4],
        tokens_per_sec=[5000, 7500, 8000, 8100],
        vram_mb=[500, 650, 700, 700],
    )
    assert os.path.exists(p2) and os.path.getsize(p2) > 1000

    p3 = visualizer.plot_lr_schedule(
        steps=[1, 2, 3, 4],
        lrs=[0.0001, 0.0005, 0.001, 0.0009],
    )
    assert os.path.exists(p3) and os.path.getsize(p3) > 1000

    p4 = visualizer.plot_gradient_norms(
        steps=[1, 2, 3, 4],
        grad_norms=[2.1, 4.8, 5.2, 3.1],
        clip_norm=5.0,
    )
    assert os.path.exists(p4) and os.path.getsize(p4) > 1000


def test_checkpoint_save_and_reload(temp_training_env):
    dataset, ckpt_dir, plots_dir, _ = temp_training_env
    batcher = BucketBatcher(dataset, token_budget=500, shuffle=False)
    model = Seq2SeqModel(vocab_size=100, d_model=64, n_layers=2)

    config = {
        "training": {
            "learning_rate": 1e-3,
            "epochs": 2,
            "warmup_steps": 10,
            "checkpoint_dir": ckpt_dir,
            "amp_enabled": False,
        }
    }

    trainer = Seq2SeqTrainer(
        model=model,
        train_batcher=batcher,
        dev_batcher=batcher,
        config=config,
        checkpoint_dir=ckpt_dir,
        plots_dir=plots_dir,
        device=torch.device("cpu"),
    )

    ckpt_file = os.path.join(ckpt_dir, "test_ckpt.pt")
    trainer.global_step = 42
    trainer.best_dev_ppl = 2.718
    trainer.save_checkpoint(ckpt_file, is_best=False)

    # Instantiate fresh model & trainer
    new_model = Seq2SeqModel(vocab_size=100, d_model=64, n_layers=2)
    new_trainer = Seq2SeqTrainer(
        model=new_model,
        train_batcher=batcher,
        dev_batcher=batcher,
        config=config,
        checkpoint_dir=ckpt_dir,
        plots_dir=plots_dir,
        device=torch.device("cpu"),
    )

    new_trainer.load_checkpoint(ckpt_file)
    assert new_trainer.global_step == 42
    assert new_trainer.best_dev_ppl == 2.718
    # Assert model weights match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
