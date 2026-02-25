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

from opaque.kernels.layernorm import NewStyleLayerNorm

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
    # NewStyleLayerNorm returns (Y, r, mu)
    result = NewStyleLayerNorm.apply(x, weight, bias, eps)
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

class TestLayerNormVmap:
    """Test vmap (per-sample gradient) precision and performance."""

    def test_vmap_precision(self, mellum_config, precision_error):
        """vmap: opaque vs pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32, requires_grad=True)

        # PyTorch vmap
        x_pt = x.detach().clone().requires_grad_(True)
        out_pt = torch.vmap(lambda inp: pytorch_layernorm(inp, weight, bias))(x_pt)
        # Use random gradient (sum() produces near-zero grads for LayerNorm)
        grad = torch.randn_like(out_pt)
        out_pt.backward(grad)

        # Opaque vmap
        x_op = x.detach().clone().requires_grad_(True)
        out_op = torch.vmap(lambda inp: opaque_layernorm(inp, weight, bias))(x_op)
        out_op.backward(grad)

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        x_err = precision_error(x_op.grad, x_pt.grad, threshold=1e-4)

        print(f"\nvmap precision:")
        print(f"  forward: abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")
        print(f"  x.grad:  abs={x_err['abs_err']:.2e}, rel={x_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_FORWARD, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"
        assert x_err["rel_err"] < RTOL_BACKWARD, f"vmap x.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"

    def test_vmap_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """vmap: opaque should be faster or use less memory than pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, hidden = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["hidden_dim"]

        weight = torch.ones(hidden, device="cuda", dtype=torch.float32)
        bias = torch.zeros(hidden, device="cuda", dtype=torch.float32)

        x = torch.randn(vmap_batch, batch, seq, hidden, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_vmap_fn(inp):
            return torch.vmap(lambda i: pytorch_layernorm(i, weight, bias))(inp)

        def opaque_vmap_fn(inp):
            return torch.vmap(lambda i: opaque_layernorm(i, weight, bias))(inp)

        pt_stats = measure_time_and_memory(pytorch_vmap_fn, x)
        op_stats = measure_time_and_memory(opaque_vmap_fn, x)

        assert_perf_benefit(pt_stats, op_stats, label="vmap", max_perf_overhead=0.50)


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
