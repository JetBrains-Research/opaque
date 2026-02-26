"""
LayerNorm Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Forward and backward performance benchmarks

Config: Mellum-4b scale (hidden_dim=3072)
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import vmap, grad

from opaque.kernels.layernorm import Opaque_LayerNorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

RTOL_FORWARD = 2e-4
RTOL_BACKWARD = 1e-2


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_layernorm(x, weight, bias, eps=1e-5):
    """PyTorch reference implementation."""
    hidden_size = x.shape[-1]
    return F.layer_norm(x, (hidden_size,), weight, bias, eps)


def opaque_layernorm(x, weight, bias, eps=1e-5):
    """Opaque Triton kernel."""
    # Opaque_LayerNorm returns (Y, r, mu)
    result = Opaque_LayerNorm.apply(x, weight, bias, eps)
    return result[0]


# ============================================================================
# Forward Pass Tests
# ============================================================================

class TestLayerNormForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, mellum_config, precision_error):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        out_pytorch = pytorch_layernorm(x, weight, bias)
        out_opaque = opaque_layernorm(x, weight, bias)

        err = precision_error(out_opaque, out_pytorch, threshold=1e-4)
        print(f"\nForward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_FORWARD, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"


# ============================================================================
# Backward Pass Tests
# ============================================================================

class TestLayerNormBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, mellum_config, precision_error):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        # PyTorch reference
        x_pt = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        out_pt = pytorch_layernorm(x_pt, weight, bias)
        # Use random gradient (sum() produces near-zero grads for LayerNorm)
        grad = torch.randn_like(out_pt)
        out_pt.backward(grad)

        # Opaque kernel
        x_op = x_pt.detach().clone().requires_grad_(True)
        out_op = opaque_layernorm(x_op, weight, bias)
        out_op.backward(grad)

        x_err = precision_error(x_op.grad, x_pt.grad, threshold=1e-4)

        print(f"\nBackward:")
        print(f"  x.grad: abs={x_err['abs_err']:.2e}, rel={x_err['rel_err']:.2e}, {x_err['pct_large']:.1f}% > thresh (target: rel<{RTOL_BACKWARD:.0e})")

        assert x_err["rel_err"] < RTOL_BACKWARD, f"x.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"


# ============================================================================
# Vmap Tests
# ============================================================================

class TestLayerNormVmapForward:
    """Test vmap forward: Triton vmap vs PyTorch vmap."""

    def test_vmap_forward_precision(self, mellum_config, precision_error):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)

        out_pt = vmap(lambda inp: pytorch_layernorm(inp, weight, bias))(x)
        out_op = vmap(lambda inp: opaque_layernorm(inp, weight, bias))(x)

        err = precision_error(out_op, out_pt, threshold=1e-4)
        print(f"\nvmap forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_FORWARD, f"vmap forward rel_err {err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"

    def test_vmap_forward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Triton vmap forward must be faster or use less memory."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)

        pt_stats = measure_time_and_memory(lambda inp: vmap(lambda i: pytorch_layernorm(i, weight, bias))(inp), x)
        op_stats = measure_time_and_memory(lambda inp: vmap(lambda i: opaque_layernorm(i, weight, bias))(inp), x)

        assert_perf_benefit(pt_stats, op_stats, label="LayerNorm vmap forward")


class TestLayerNormVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, mellum_config, precision_error):
        """Per-example gradients: opaque Triton vs PyTorch reference.

        Uses a random target weighting because .sum() produces near-zero grads
        for LayerNorm (output is mean-centered, so sum ≈ 0).
        """
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)
        # Fixed random target provides non-trivial gradient signal
        target = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32)

        def f_pt(xi):
            return (pytorch_layernorm(xi, weight, bias) * target).sum()

        def f_op(xi):
            return (opaque_layernorm(xi, weight, bias) * target).sum()

        grads_pt = vmap(grad(f_pt))(x)
        grads_op = vmap(grad(f_op))(x)

        err = precision_error(grads_op, grads_pt, threshold=1e-4)
        print(f"\nvmap(grad): abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert err["rel_err"] < RTOL_BACKWARD, f"vmap(grad) rel_err {err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"

    def test_vmap_grad_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Triton vmap(grad) must be faster or use less memory."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32)
        target = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32)

        def make_pt_fn():
            def f(xi):
                return (pytorch_layernorm(xi, weight, bias) * target).sum()
            return vmap(grad(f))

        def make_op_fn():
            def f(xi):
                return (opaque_layernorm(xi, weight, bias) * target).sum()
            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), x)
        op_stats = measure_time_and_memory(make_op_fn(), x)

        assert_perf_benefit(pt_stats, op_stats, label="LayerNorm vmap(grad)")


# ============================================================================
# Performance Tests
# ============================================================================

class TestLayerNormPerformance:
    """Benchmark forward and backward performance."""

    def test_forward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Forward-only: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        pt_stats = measure_time_and_memory(pytorch_layernorm, x, weight, bias)
        op_stats = measure_time_and_memory(opaque_layernorm, x, weight, bias)

        assert_perf_benefit(pt_stats, op_stats, label="forward", max_perf_overhead=0.60)

    def test_backward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Forward+backward: opaque vs pytorch performance."""
        torch.manual_seed(42)
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32, requires_grad=True)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        pt_stats = measure_time_and_memory(pytorch_layernorm, x, weight, bias)
        op_stats = measure_time_and_memory(opaque_layernorm, x, weight, bias)

        assert_perf_benefit(pt_stats, op_stats, label="backward", max_perf_overhead=0.60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
