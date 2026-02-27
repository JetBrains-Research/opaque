"""Shared test fixtures for kernel tests.

Provides pytest fixtures for:
- mellum_config: Mellum-4b model dimensions for realistic testing
- assert_precision: Assert precision using torch.testing.assert_close (atol+rtol)
- measure_time_and_memory: Benchmark execution time and peak CUDA memory
- assert_perf_benefit: Assert performance improvement
"""

import gc
import time
import pytest
import torch


# ============================================================================
# Mellum-4b model configuration
# ============================================================================

MELLUM_CONFIG = {
    "batch_size": 8,
    "seq_len": 512,
    "hidden_dim": 3072,
    "intermediate_dim": 8256,  # FFN intermediate size
    "n_heads": 24,
    "head_dim": 128,
    "vocab_size": 98304,
    "rank": 64,  # LoRA rank
    "vmap_batch": 4,
}


# ============================================================================
# Utility functions
# ============================================================================

def _assert_precision(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    label: str = "",
) -> dict:
    """Assert precision using combined atol+rtol formula with diagnostic reporting.

    Uses torch.testing.assert_close: |actual - expected| <= atol + rtol * |expected|
    This gracefully handles near-zero values (falls back to atol) and large values
    (uses rtol), avoiding the near-zero blowup of pure max relative error.

    Returns dict with abs_err and rel_err for diagnostic purposes.
    """
    diff = (actual.float() - expected.float()).abs()
    abs_err = diff.max().item()

    # Compute max relative error for diagnostics (not used for assertion)
    expected_abs = expected.float().abs()
    mask = expected_abs >= 1e-6
    if mask.sum() > 0:
        rel_err = (diff[mask] / expected_abs[mask]).max().item()
    else:
        rel_err = float("nan")

    prefix = f"  {label}: " if label else "  "
    print(f"{prefix}abs={abs_err:.2e}, rel={rel_err:.2e} (rtol={rtol:.0e}, atol={atol:.0e})")

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    return {"abs_err": abs_err, "rel_err": rel_err}


def _measure_time_and_memory(fn, *args, warmup=3, runs=10):
    """Measure execution time and peak CUDA memory.

    Runs warmup iterations, then timed iterations. Each iteration clones
    inputs and runs forward + backward (if output requires grad).
    """
    for _ in range(warmup):
        inputs = [a.detach().clone().requires_grad_(a.requires_grad) for a in args]
        out = fn(*inputs)
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward()
        torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(runs):
        inputs = [a.detach().clone().requires_grad_(a.requires_grad) for a in args]
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*inputs)
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "time_ms": sum(times) / len(times) * 1000,
        "memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def _assert_perf_benefit(pt_stats, op_stats, label="", max_perf_overhead=0.20):
    """Assert that opaque kernel shows benefit in speed or memory."""
    speedup = pt_stats["time_ms"] / op_stats["time_ms"]
    mem_reduction = pt_stats["memory_mb"] / op_stats["memory_mb"]

    print(f"\n{label} performance:")
    print(f"  PyTorch: {pt_stats['time_ms']:.2f}ms, {pt_stats['memory_mb']:.1f}MB")
    print(f"  Opaque:  {op_stats['time_ms']:.2f}ms, {op_stats['memory_mb']:.1f}MB")
    print(f"  Speedup: {speedup:.2f}x, Memory reduction: {mem_reduction:.2f}x")

    has_benefit = speedup > (1.0 - max_perf_overhead) or mem_reduction > 1.0
    assert has_benefit, (
        f"No benefit for {label}: speedup={speedup:.2f}x, "
        f"mem_reduction={mem_reduction:.2f}x"
    )


# ============================================================================
# Pytest fixtures
# ============================================================================

@pytest.fixture(scope="session")
def mellum_config():
    """Mellum-4b model configuration for realistic testing."""
    return MELLUM_CONFIG


@pytest.fixture(scope="session")
def assert_precision():
    """Assert precision using torch.testing.assert_close with diagnostics."""
    return _assert_precision


@pytest.fixture(scope="session")
def measure_time_and_memory():
    """Benchmark execution time and peak CUDA memory."""
    return _measure_time_and_memory


@pytest.fixture(scope="session")
def assert_perf_benefit():
    """Assert performance improvement over PyTorch baseline."""
    return _assert_perf_benefit
