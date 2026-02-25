"""
LayerNorm Kernel Validation

3-phase validation:
1. Opaque kernel vs PyTorch F.layer_norm (baseline)
2. Opaque kernel backward vs PyTorch backward
3. Opaque vmap vs torch.vmap(PyTorch)
"""

import pytest
import torch
import torch.nn.functional as F
import sys

try:
    from .kernel_validation_framework import (
        benchmark_forward_backward,
        print_comparison_table,
    )
except ImportError:
    from kernel_validation_framework import (
        benchmark_forward_backward,
        print_comparison_table,
    )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel tests"
)


# ============================================================================
# Test Configurations
# ============================================================================

TEST_CONFIGS = [
    {"batch_size": 4, "seq_len": 128, "hidden_size": 256, "name": "Small"},
    {"batch_size": 2, "seq_len": 256, "hidden_size": 1024, "name": "Medium"},
    {"batch_size": 1, "seq_len": 512, "hidden_size": 4096, "name": "Large"},
]


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_layernorm(x, weight, bias, eps=1e-5):
    """PyTorch reference implementation."""
    hidden_size = x.shape[-1]
    return F.layer_norm(x, (hidden_size,), weight, bias, eps)


def opaque_layernorm(x, weight, bias, eps=1e-5):
    """Opaque kernel implementation."""
    from opaque.kernels import NewStyleLayerNorm
    # NewStyleLayerNorm returns (Y, r, mu, BLOCK_SIZE, num_warps)
    result = NewStyleLayerNorm.apply(x, weight, bias, eps)
    return result[0]  # Just the output Y


# ============================================================================
# Phase 1: Forward Pass Validation
# ============================================================================

class TestLayerNormForward:
    """Test forward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_forward_matches_pytorch(self, config):
        """Verify forward pass matches PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_layernorm(x, weight, bias)
        pytorch_out = pytorch_layernorm(x, weight, bias)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        print(f"\n{config['name']}: forward diff = {max_diff:.2e}")

        assert torch.allclose(opaque_out, pytorch_out, rtol=1e-4, atol=1e-4), (
            f"Forward mismatch for {config['name']}: diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 2: Backward Pass Validation
# ============================================================================

class TestLayerNormBackward:
    """Test backward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_backward_matches_pytorch(self, config):
        """Verify gradients match PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        # Opaque kernel
        x_opaque = torch.randn(
            batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_layernorm(x_opaque, weight, bias)
        opaque_out.sum().backward()
        opaque_grad = x_opaque.grad.clone()

        # PyTorch reference
        x_pytorch = x_opaque.detach().clone().requires_grad_(True)
        pytorch_out = pytorch_layernorm(x_pytorch, weight, bias)
        pytorch_out.sum().backward()
        pytorch_grad = x_pytorch.grad

        max_diff = (opaque_grad - pytorch_grad).abs().max().item()
        print(f"\n{config['name']}: backward max diff = {max_diff:.2e}")

        assert torch.allclose(opaque_grad, pytorch_grad, rtol=1e-3, atol=1e-3), (
            f"Backward mismatch for {config['name']}: max_diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 3: Vmap Validation
# ============================================================================

class TestLayerNormVmap:
    """Test vmap support for DP-SGD."""

    def test_vmap_forward_matches_loop(self):
        """Verify vmap produces same results as loop."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, hidden = 4, 2, 64, 256

        x = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
        )
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        # vmap over opaque kernel
        vmapped_fn = torch.vmap(
            lambda inp: opaque_layernorm(inp, weight, bias),
            in_dims=0
        )
        vmap_results = vmapped_fn(x)

        # Loop reference
        loop_results = torch.stack([
            opaque_layernorm(x[i], weight, bias)
            for i in range(vmap_batch)
        ])

        max_diff = (vmap_results - loop_results).abs().max().item()
        print(f"\nvmap vs loop: max diff = {max_diff:.2e}")

        assert torch.allclose(vmap_results, loop_results, rtol=1e-5, atol=1e-5), (
            f"vmap mismatch: max_diff={max_diff:.2e}"
        )

    def test_vmap_matches_pytorch_vmap(self):
        """Verify Opaque vmap matches torch.vmap(PyTorch)."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, hidden = 4, 2, 64, 256

        x = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
        )
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        # Opaque vmap
        opaque_vmapped = torch.vmap(
            lambda inp: opaque_layernorm(inp, weight, bias), in_dims=0
        )
        opaque_results = opaque_vmapped(x)

        # PyTorch vmap
        pytorch_vmapped = torch.vmap(
            lambda inp: pytorch_layernorm(inp, weight, bias), in_dims=0
        )
        pytorch_results = pytorch_vmapped(x)

        max_diff = (opaque_results - pytorch_results).abs().max().item()
        print(f"\nOpaque vmap vs PyTorch vmap: max diff = {max_diff:.2e}")

        assert torch.allclose(opaque_results, pytorch_results, rtol=1e-4, atol=1e-4), (
            f"vmap mismatch: max_diff={max_diff:.2e}"
        )

    def test_vmap_backward(self):
        """Verify vmap backward pass works correctly."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, hidden = 3, 2, 32, 128

        x = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        # vmap forward + backward
        vmapped_fn = torch.vmap(
            lambda inp: opaque_layernorm(inp, weight, bias), in_dims=0
        )
        vmap_results = vmapped_fn(x)
        vmap_results.sum().backward()

        assert x.grad is not None, "No gradient computed"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        print(f"\nvmap backward: grad shape = {x.grad.shape}")


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestLayerNormPerformance:
    """Performance benchmarks."""

    def test_benchmark_forward_backward(self):
        """Benchmark forward and backward pass."""
        torch.manual_seed(42)
        batch, seq_len, hidden = 8, 256, 4096

        x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        results = []

        # PyTorch
        bench_pytorch = benchmark_forward_backward(
            lambda inp: pytorch_layernorm(inp, weight, bias),
            [x], [0], name="PyTorch F.layer_norm"
        )
        results.append(bench_pytorch)

        # Opaque
        bench_opaque = benchmark_forward_backward(
            lambda inp: opaque_layernorm(inp, weight, bias),
            [x], [0], name="Opaque kernel"
        )
        results.append(bench_opaque)

        print_comparison_table(results, "LayerNorm Performance (batch=8, seq=256, hidden=4096)")


# ============================================================================
# Full Validation Suite
# ============================================================================

def run_full_validation():
    """Run complete validation suite with detailed output."""
    print("\n" + "="*80)
    print("LAYERNORM KERNEL VALIDATION")
    print("="*80)

    all_passed = True

    # Phase 1: Forward validation
    print("\n--- Phase 1: Forward Pass Validation ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_layernorm(x, weight, bias)
        pytorch_out = pytorch_layernorm(x, weight, bias)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        passed = max_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 2: Backward validation
    print("\n--- Phase 2: Backward Pass Validation ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        x_opaque = torch.randn(
            batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_layernorm(x_opaque, weight, bias)
        opaque_out.sum().backward()
        opaque_grad = x_opaque.grad.clone()

        x_pytorch = x_opaque.detach().clone().requires_grad_(True)
        pytorch_out = pytorch_layernorm(x_pytorch, weight, bias)
        pytorch_out.sum().backward()
        pytorch_grad = x_pytorch.grad

        max_diff = (opaque_grad - pytorch_grad).abs().max().item()
        passed = max_diff < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (max_diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 3: Vmap validation
    print("\n--- Phase 3: Vmap Validation ---")
    torch.manual_seed(42)
    vmap_batch, batch, seq_len, hidden = 4, 2, 64, 256

    x = torch.randn(
        vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
    )
    weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
    bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

    opaque_vmapped = torch.vmap(
        lambda inp: opaque_layernorm(inp, weight, bias), in_dims=0
    )
    pytorch_vmapped = torch.vmap(
        lambda inp: pytorch_layernorm(inp, weight, bias), in_dims=0
    )

    opaque_results = opaque_vmapped(x)
    pytorch_results = pytorch_vmapped(x)

    max_diff = (opaque_results - pytorch_results).abs().max().item()
    passed = max_diff < 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  Opaque vmap vs PyTorch vmap: {status} (max_diff={max_diff:.2e})")

    if not passed:
        all_passed = False

    # Performance comparison
    print("\n--- Performance Comparison ---")
    batch, seq_len, hidden = 8, 256, 4096
    x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
    weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
    bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

    bench_pytorch = benchmark_forward_backward(
        lambda inp: pytorch_layernorm(inp, weight, bias),
        [x], [0], name="PyTorch"
    )
    bench_opaque = benchmark_forward_backward(
        lambda inp: opaque_layernorm(inp, weight, bias),
        [x], [0], name="Opaque"
    )

    print_comparison_table([bench_pytorch, bench_opaque], "Forward+Backward Performance")

    # Summary
    print("\n" + "="*80)
    print(f"VALIDATION RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)
