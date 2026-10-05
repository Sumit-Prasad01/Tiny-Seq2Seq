"""
Unit tests for Phase 0: Environment, Toolchain, and Core Infrastructure.
"""

import os
import torch
import torch.nn as nn
import pytest

from utils import (
    set_seed,
    get_gpu_memory_mb,
    count_parameters,
    compute_perplexity,
    ThroughputTracker,
    logger,
)


def test_seed_reproducibility():
    set_seed(42)
    a = torch.randn(10, 10)
    set_seed(42)
    b = torch.randn(10, 10)
    assert torch.equal(a, b), "Deterministic seed failed to reproduce tensor"


def test_parameter_counting():
    linear = nn.Linear(512, 512, bias=True)
    counts = count_parameters(linear)
    expected = 512 * 512 + 512
    assert counts["total_parameters"] == expected
    assert counts["trainable_parameters"] == expected


def test_perplexity_computation():
    assert round(compute_perplexity(0.0), 4) == 1.0
    assert round(compute_perplexity(1.0), 4) == round(2.718281828, 4)
    # Test clipping overflow protection
    huge_loss = 500.0
    ppl = compute_perplexity(huge_loss)
    import math
    assert ppl > 0 and not math.isnan(ppl) and (not math.isinf(ppl) or ppl == float("inf"))


def test_throughput_tracker():
    tracker = ThroughputTracker(window_size=5)
    stats = tracker.update(num_tokens=1000, num_batches=1, elapsed_seconds=0.1)
    assert stats["total_tokens"] == 1000
    assert stats["total_batches"] == 1
    assert stats["tokens_per_second"] == 10000.0


def test_cuda_gemm_fp16():
    """Validates GPU availability and FP16 GEMM execution."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU GEMM test")

    device = torch.device("cuda")
    mem_before = get_gpu_memory_mb()
    assert mem_before is not None

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        x = torch.randn(1024, 512, device=device, dtype=torch.float16)
        w = torch.randn(512, 512, device=device, dtype=torch.float16)
        out = torch.matmul(x, w)

    assert out.shape == (1024, 512)
    assert out.device.type == "cuda"
    mem_after = get_gpu_memory_mb()
    assert mem_after["allocated_mb"] > 0
    logger.info(f"Phase 0 GPU GEMM Success. VRAM Allocated: {mem_after['allocated_mb']} MB")
