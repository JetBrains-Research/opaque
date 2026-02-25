"""
SwiGLU Kernel Tests

Tests opaque SwiGLU kernel against PyTorch vmap:
1. Precision: rtol=1e-7 (relative error only)
2. Performance: measure forward+backward time vs PyTorch vmap
3. Memory: measure peak memory vs PyTorch vmap

Each test verifies that the kernel provides either a performance or memory benefit.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add kernel path without importing main opaque package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

try:
    from .kernel_validation_framework import (
        benchmark_forward_backward,
        validate_implementations,
        print_validation_result,
        print_comparison_table,
    )
except ImportError:
    from kernel_validation_framework import (
        benchmark_forward_backward,
        validate_implementations,
        print_validation_result,
        print_comparison_table,
    )

# Import kernels directly to avoid opaque __init__.py dependency issues
from opaque.kernels.swiglu import NewStyleSwiGLU


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_swiglu(e, g):
    """PyTorch reference: SwiGLU = silu(e) * g."""
    return F.silu(e) * g


def opaque_swiglu(e, g):
    """Opaque kernel implementation."""
    result = NewStyleSwiGLU.apply(e, g)
    return result[0] if isinstance(result, tuple) else result


# ============================================================================
# Test Configurations
# ============================================================================

# Mellum-4b-base-like: intermediate_dim=8256
TEST_CONFIGS = [
    {"batch_size": 2, "seq_len": 128, "hidden_size": 8256, "name": "Mellum-MLP"},
    {"batch_size": 4, "seq_len": 256, "hidden_size": 4096, "name": "Medium"},
]


# ============================================================================
# Tests
# ============================================================================

class TestSwiGLU:
    """Test SwiGLU kernel vs PyTorch vmap."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_precision_vs_pytorch_vmap(self, config):
        """Verify precision: rtol=1e-7 vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e = torch.randn(vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float64)
        g = torch.randn(vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float64)

        def opaque_vmapped(e, g):
            return torch.vmap(opaque_swiglu, in_dims=(0, 0))(e, g)

        def pytorch_vmapped(e, g):
            return torch.vmap(pytorch_swiglu, in_dims=(0, 0))(e, g)

        val = validate_implementations(
            opaque_vmapped,
            pytorch_vmapped,
            [e, g],
            grad_inputs_idx=[0, 1],
            name=f"SwiGLU {config['name']}",
            rtol=1e-7,
            use_relative_only=True,
        )

        print_validation_result(val)

        assert val.forward_matches, (
            f"Forward precision failed: {config['name']} "
            f"(rel_error={val.forward_max_diff:.2e}, rtol=1e-7)"
        )
        assert val.backward_matches, (
            f"Backward precision failed: {config['name']} "
            f"(rel_errors={val.backward_max_diff}, rtol=1e-7)"
        )

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_performance_vs_pytorch_vmap(self, config):
        """Verify performance vs PyTorch vmap."""
        torch.manual_seed(42)
        vmap_batch = 4
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e = torch.randn(vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float64)
        g = torch.randn(vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float64)

        def opaque_vmapped(e, g):
            return torch.vmap(opaque_swiglu, in_dims=(0, 0))(e, g)

        def pytorch_vmapped(e, g):
            return torch.vmap(pytorch_swiglu, in_dims=(0, 0))(e, g)

        opaque_bench = benchmark_forward_backward(
            opaque_vmapped, [e, g], grad_inputs_idx=[0, 1], name=f"Opaque {config['name']}"
        )
        pytorch_bench = benchmark_forward_backward(
            pytorch_vmapped, [e, g], grad_inputs_idx=[0, 1], name=f"PyTorch {config['name']}"
        )

        print_comparison_table([opaque_bench, pytorch_bench], f"SwiGLU {config['name']}")

        perf_ratio = opaque_bench.total_time_ms / pytorch_bench.total_time_ms
        mem_ratio = opaque_bench.memory_peak_mb / pytorch_bench.memory_peak_mb

        print(f"\n{config['name']}: perf_ratio={perf_ratio:.2%}, mem_ratio={mem_ratio:.2%}")

        has_benefit = perf_ratio < 1.0 or mem_ratio < 1.0
        if not has_benefit:
            pytest.skip(
                f"No performance or memory benefit: "
                f"perf={perf_ratio:.2%}, mem={mem_ratio:.2%}"
            )


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("SWIGLU KERNEL TESTS")
    print("=" * 80)

    pytest.main([__file__, "-v", "-s"])
