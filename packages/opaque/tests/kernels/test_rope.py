"""
RoPE (Rotary Position Embedding) Kernel Validation

3-phase validation:
1. Opaque kernel vs PyTorch reference (baseline)
2. Opaque kernel backward vs PyTorch backward
3. Opaque vmap vs torch.vmap(PyTorch)
"""

import pytest
import torch
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
# Reference Implementations
# ============================================================================

def rotate_half(x):
    """Standard rotate_half for RoPE."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def pytorch_rope(Q, cos, sin):
    """PyTorch reference RoPE implementation.

    Args:
        Q: (batch, seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim/2)
        sin: (seq_len, head_dim/2)
    """
    batch, seq_len, n_heads, head_dim = Q.shape

    # Expand cos/sin for broadcasting
    cos_expanded = cos[None, :, None, :].expand(batch, seq_len, n_heads, -1)
    sin_expanded = sin[None, :, None, :].expand(batch, seq_len, n_heads, -1)

    # Duplicate for full head_dim
    cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
    sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)

    # Apply rotation
    return Q * cos_full + rotate_half(Q) * sin_full


def opaque_rope(Q, cos, sin):
    """Opaque kernel implementation."""
    from opaque.kernels import NewStyleRoPEEmbedding
    result = NewStyleRoPEEmbedding.apply(Q, cos, sin)
    return result[0]  # Just the rotated Q


def generate_cos_sin(seq_len, head_dim, device="cuda", dtype=torch.float32):
    """Generate cos/sin for RoPE."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device)
    freqs = torch.outer(positions, freqs)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


# ============================================================================
# Test Configurations
# ============================================================================

TEST_CONFIGS = [
    {"batch_size": 2, "seq_len": 64, "n_heads": 8, "head_dim": 64, "name": "Small"},
    {"batch_size": 1, "seq_len": 128, "n_heads": 32, "head_dim": 128, "name": "Medium"},
]


# ============================================================================
# Phase 1: Forward Pass Validation
# ============================================================================

class TestRoPEForward:
    """Test forward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_forward_matches_pytorch(self, config):
        """Verify forward pass matches PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, n_heads, head_dim = (
            config["batch_size"], config["seq_len"], config["n_heads"], config["head_dim"]
        )

        Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        cos, sin = generate_cos_sin(seq_len, head_dim)

        opaque_out = opaque_rope(Q, cos, sin)
        pytorch_out = pytorch_rope(Q, cos, sin)

        max_diff = (opaque_out - pytorch_out).abs().max().item()
        print(f"\n{config['name']}: forward diff = {max_diff:.2e}")

        assert torch.allclose(opaque_out, pytorch_out, rtol=1e-4, atol=1e-4), (
            f"Forward mismatch for {config['name']}: diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 2: Backward Pass Validation
# ============================================================================

class TestRoPEBackward:
    """Test backward pass matches PyTorch."""

    @pytest.mark.parametrize("config", TEST_CONFIGS)
    def test_backward_matches_pytorch(self, config):
        """Verify gradients match PyTorch."""
        torch.manual_seed(42)
        batch, seq_len, n_heads, head_dim = (
            config["batch_size"], config["seq_len"], config["n_heads"], config["head_dim"]
        )

        # Opaque kernel
        Q_opaque = torch.randn(
            batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        cos, sin = generate_cos_sin(seq_len, head_dim)

        opaque_out = opaque_rope(Q_opaque, cos, sin)
        opaque_out.sum().backward()
        opaque_grad = Q_opaque.grad.clone()

        # PyTorch reference
        Q_pytorch = Q_opaque.detach().clone().requires_grad_(True)
        pytorch_out = pytorch_rope(Q_pytorch, cos, sin)
        pytorch_out.sum().backward()
        pytorch_grad = Q_pytorch.grad

        max_diff = (opaque_grad - pytorch_grad).abs().max().item()
        print(f"\n{config['name']}: backward max diff = {max_diff:.2e}")

        assert torch.allclose(opaque_grad, pytorch_grad, rtol=1e-4, atol=1e-4), (
            f"Backward mismatch for {config['name']}: max_diff={max_diff:.2e}"
        )


# ============================================================================
# Phase 3: Vmap Validation
# ============================================================================

class TestRoPEVmap:
    """Test vmap support for DP-SGD."""

    def test_vmap_forward_matches_loop(self):
        """Verify vmap produces same results as loop."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, n_heads, head_dim = 4, 2, 32, 4, 32

        Q = torch.randn(
            vmap_batch, batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32
        )
        cos, sin = generate_cos_sin(seq_len, head_dim)

        # vmap over opaque kernel
        vmapped_fn = torch.vmap(lambda q: opaque_rope(q, cos, sin), in_dims=0)
        vmap_results = vmapped_fn(Q)

        # Loop reference
        loop_results = torch.stack([
            opaque_rope(Q[i], cos, sin) for i in range(vmap_batch)
        ])

        max_diff = (vmap_results - loop_results).abs().max().item()
        print(f"\nvmap vs loop: max diff = {max_diff:.2e}")

        assert torch.allclose(vmap_results, loop_results, rtol=1e-5, atol=1e-5), (
            f"vmap mismatch: max_diff={max_diff:.2e}"
        )

    def test_vmap_backward(self):
        """Verify vmap backward pass works correctly."""
        torch.manual_seed(42)
        vmap_batch, batch, seq_len, n_heads, head_dim = 3, 2, 32, 4, 32

        Q = torch.randn(
            vmap_batch, batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        cos, sin = generate_cos_sin(seq_len, head_dim)

        # vmap forward + backward
        vmapped_fn = torch.vmap(lambda q: opaque_rope(q, cos, sin), in_dims=0)
        vmap_results = vmapped_fn(Q)
        vmap_results.sum().backward()

        assert Q.grad is not None, "No gradient computed"
        assert Q.grad.shape == Q.shape, "Gradient shape mismatch"
        print(f"\nvmap backward: grad shape = {Q.grad.shape}")


# ============================================================================
# Full Validation Suite
# ============================================================================

def run_full_validation():
    """Run complete validation suite with detailed output."""
    print("\n" + "="*80)
    print("ROPE KERNEL VALIDATION")
    print("="*80)

    all_passed = True

    # Phase 1: Forward validation
    print("\n--- Phase 1: Forward Pass Validation ---")
    for config in TEST_CONFIGS:
        torch.manual_seed(42)
        batch, seq_len, n_heads, head_dim = (
            config["batch_size"], config["seq_len"], config["n_heads"], config["head_dim"]
        )

        Q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32)
        cos, sin = generate_cos_sin(seq_len, head_dim)

        opaque_out = opaque_rope(Q, cos, sin)
        pytorch_out = pytorch_rope(Q, cos, sin)

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
        batch, seq_len, n_heads, head_dim = (
            config["batch_size"], config["seq_len"], config["n_heads"], config["head_dim"]
        )

        Q_opaque = torch.randn(
            batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
        )
        cos, sin = generate_cos_sin(seq_len, head_dim)

        opaque_out = opaque_rope(Q_opaque, cos, sin)
        opaque_out.sum().backward()

        Q_pytorch = Q_opaque.detach().clone().requires_grad_(True)
        pytorch_out = pytorch_rope(Q_pytorch, cos, sin)
        pytorch_out.sum().backward()

        max_diff = (Q_opaque.grad - Q_pytorch.grad).abs().max().item()
        passed = max_diff < 1e-4
        status = "PASS" if passed else "FAIL"
        print(f"  {config['name']}: {status} (max_diff={max_diff:.2e})")

        if not passed:
            all_passed = False

    # Phase 3: Vmap validation
    print("\n--- Phase 3: Vmap Validation ---")
    torch.manual_seed(42)
    vmap_batch, batch, seq_len, n_heads, head_dim = 4, 2, 32, 4, 32

    Q = torch.randn(
        vmap_batch, batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float32
    )
    cos, sin = generate_cos_sin(seq_len, head_dim)

    vmapped_fn = torch.vmap(lambda q: opaque_rope(q, cos, sin), in_dims=0)
    vmap_results = vmapped_fn(Q)

    loop_results = torch.stack([opaque_rope(Q[i], cos, sin) for i in range(vmap_batch)])

    max_diff = (vmap_results - loop_results).abs().max().item()
    passed = max_diff < 1e-5
    status = "PASS" if passed else "FAIL"
    print(f"  vmap vs loop: {status} (max_diff={max_diff:.2e})")

    if not passed:
        all_passed = False

    # Summary
    print("\n" + "="*80)
    print(f"VALIDATION RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = run_full_validation()
    sys.exit(0 if success else 1)
