"""
GeGLU Kernel Validation

3-phase validation:
1. Opaque kernel vs PyTorch gelu(e) * g (baseline)
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

def pytorch_geglu_exact(e, g):
    """PyTorch reference: GeGLU (exact) = gelu(e) * g."""
    return F.gelu(e, approximate="none") * g


def pytorch_geglu_approx(e, g):
    """PyTorch reference: GeGLU (approx tanh) = gelu_tanh(e) * g."""
    return F.gelu(e, approximate="tanh") * g


def opaque_geglu_exact(e, g):
    """Opaque kernel implementation (exact)."""
    from opaque.kernels import NewStyleGeGLUExact
    result = NewStyleGeGLUExact.apply(e, g)
    return result[0] if isinstance(result, tuple) else result


def opaque_geglu_approx(e, g):
    """Opaque kernel implementation (approx)."""
    from opaque.kernels import NewStyleGeGLUApprox
    result = NewStyleGeGLUApprox.apply(e, g)
    return result[0] if isinstance(result, tuple) else result


# ============================================================================
# Phase 1: Forward Pass Validation
# ============================================================================

class TestGeGLUForward:
    """Test forward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_exact_forward_matches_pytorch(self, config):
        """Verify exact GeGLU forward pass matches PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        g = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_geglu_exact(e, g)
        pytorch_out = pytorch_geglu_exact(e, g)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        print(f"\n{config['name']} (exact): forward diff = {max_diff:.2e}")

        assert torch.allclose(opaque_out, pytorch_out, rtol=1e-4, atol=1e-4), (
            f"Forward mismatch for {config['name']}: diff={max_diff:.2e}"
        )

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_approx_forward_matches_pytorch(self, config):
        """Verify approx GeGLU forward pass matches PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        g = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_geglu_approx(e, g)
        pytorch_out = pytorch_geglu_approx(e, g)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        print(f"\n{config['name']} (approx): forward diff = {max_diff:.2e}")

        # Approx may have slightly larger tolerance due to tanh approximation
        assert torch.allclose(opaque_out, pytorch_out, rtol=1e-3, atol=1e-3), (
            f"Forward mismatch for {config['name']}: diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 2: Backward Pass Validation
# ============================================================================

class TestGeGLUBackward:
    """Test backward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_exact_backward_matches_pytorch(self, config):
        """Verify exact GeGLU gradients match PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        # Opaque kernel
        e_opaque = torch.randn(
            batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        g_opaque = torch.randn(
            batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )

        opaque_out = opaque_geglu_exact(e_opaque, g_opaque)
        opaque_out.sum().backward()
        opaque_e_grad = e_opaque.grad.clone()
        opaque_g_grad = g_opaque.grad.clone()

        # PyTorch reference
        e_pytorch = e_opaque.detach().clone().requires_grad_(True)
        g_pytorch = g_opaque.detach().clone().requires_grad_(True)
        pytorch_out = pytorch_geglu_exact(e_pytorch, g_pytorch)
        pytorch_out.sum().backward()

        e_diff = (opaque_e_grad - e_pytorch.grad).abs().max().item()
        g_diff = (opaque_g_grad - g_pytorch.grad).abs().max().item()
        print(f"\n{config['name']} (exact): e_grad diff = {e_diff:.2e}, g_grad diff = {g_diff:.2e}")

        assert torch.allclose(opaque_e_grad, e_pytorch.grad, rtol=1e-3, atol=1e-3), (
            f"Backward e mismatch: max_diff={e_diff:.2e}"
        )
        assert torch.allclose(opaque_g_grad, g_pytorch.grad, rtol=1e-3, atol=1e-3), (
            f"Backward g mismatch: max_diff={g_diff:.2e}"
        )


# ============================================================================
# Phase 3: Vmap Validation
# ============================================================================

class TestGeGLUVmap:
    """Test vmap support for DP-SGD."""

    def test_exact_vmap_matches_pytorch_vmap(self):
        """Verify Opaque exact vmap matches torch.vmap(PyTorch)."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, hidden = 4, 2, 64, 256

        e = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
        )
        g = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
        )

        # Opaque vmap
        opaque_vmapped = torch.vmap(opaque_geglu_exact, in_dims=(0, 0))
        opaque_results = opaque_vmapped(e, g)

        # PyTorch vmap
        pytorch_vmapped = torch.vmap(pytorch_geglu_exact, in_dims=(0, 0))
        pytorch_results = pytorch_vmapped(e, g)

        max_diff = (opaque_results - pytorch_results).abs().max().item()
        print(f"\nExact vmap: Opaque vs PyTorch max diff = {max_diff:.2e}")

        assert torch.allclose(opaque_results, pytorch_results, rtol=1e-4, atol=1e-4), (
            f"vmap mismatch: max_diff={max_diff:.2e}"
        )

    def test_vmap_backward(self):
        """Verify vmap backward pass works correctly."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, hidden = 3, 2, 32, 128

        e = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        g = torch.randn(
            vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )

        # vmap forward + backward
        vmapped_fn = torch.vmap(opaque_geglu_exact, in_dims=(0, 0))
        vmap_results = vmapped_fn(e, g)
        vmap_results.sum().backward()

        assert e.grad is not None, "No gradient computed for e"
        assert g.grad is not None, "No gradient computed for g"
        print(f"\nvmap backward: e grad shape = {e.grad.shape}, g grad shape = {g.grad.shape}")


# ============================================================================
# Full Validation Suite
# ============================================================================

def run_full_validation():
    """Run complete validation suite with detailed output."""
    print("\n" + "="*80)
    print("GEGLU KERNEL VALIDATION")
    print("="*80)

    all_passed = True

    # Phase 1: Forward validation (exact)
    print("\n--- Phase 1: Forward Pass Validation (Exact) ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        g = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_geglu_exact(e, g)
        pytorch_out = pytorch_geglu_exact(e, g)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        passed = max_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 1: Forward validation (approx)
    print("\n--- Phase 1: Forward Pass Validation (Approx) ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
        g = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)

        opaque_out = opaque_geglu_approx(e, g)
        pytorch_out = pytorch_geglu_approx(e, g)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        passed = max_diff < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 2: Backward validation (exact)
    print("\n--- Phase 2: Backward Pass Validation (Exact) ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, hidden = config["batch_size"], config["seq_len"], config["hidden_size"]

        e_opaque = torch.randn(
            batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        g_opaque = torch.randn(
            batch, seq_len, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )

        opaque_out = opaque_geglu_exact(e_opaque, g_opaque)
        opaque_out.sum().backward()

        e_pytorch = e_opaque.detach().clone().requires_grad_(True)
        g_pytorch = g_opaque.detach().clone().requires_grad_(True)
        pytorch_out = pytorch_geglu_exact(e_pytorch, g_pytorch)
        pytorch_out.sum().backward()

        e_diff = (e_opaque.grad - e_pytorch.grad).abs().max().item()
        g_diff = (g_opaque.grad - g_pytorch.grad).abs().max().item()
        passed = e_diff < 1e-3 and g_diff < 1e-3
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (e_diff={e_diff:.2e}, g_diff={g_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 3: Vmap validation
    print("\n--- Phase 3: Vmap Validation ---")
    torch.manual_seed(42)
    vmap_batch, batch, seq_len, hidden = 4, 2, 64, 256

    e = torch.randn(
        vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
    )
    g = torch.randn(
        vmap_batch, batch, seq_len, hidden, device="cuda", dtype=torch.float32
    )

    opaque_vmapped = torch.vmap(opaque_geglu_exact, in_dims=(0, 0))
    pytorch_vmapped = torch.vmap(pytorch_geglu_exact, in_dims=(0, 0))

    opaque_results = opaque_vmapped(e, g)
    pytorch_results = pytorch_vmapped(e, g)

    max_diff = (opaque_results - pytorch_results).abs().max().item()
    passed = max_diff < 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  Opaque vmap vs PyTorch vmap: {status} (max_diff={max_diff:.2e})")

    if not passed:
        all_passed = False

    # Performance comparison
    print("\n--- Performance Comparison ---")
    batch, seq_len, hidden = 8, 256, 4096
    e = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)
    g = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float32)

    bench_pytorch = benchmark_forward_backward(
        lambda e_in, g_in: pytorch_geglu_exact(e_in, g_in),
        [e, g], [0, 1], name="PyTorch"
    )
    bench_opaque = benchmark_forward_backward(
        lambda e_in, g_in: opaque_geglu_exact(e_in, g_in),
        [e, g], [0, 1], name="Opaque"
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
