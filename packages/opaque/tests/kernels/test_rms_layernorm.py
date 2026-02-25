"""
RMS LayerNorm Kernel Tests

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Performance benchmarks (forward-only and forward+backward)
5. Edge cases (zero input, large values)

Config: Mellum-4b scale (hidden_dim=3072)
"""

import pytest
import torch
from opaque.kernels.rms_layernorm import rms_layernorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

RTOL_FORWARD = 1e-5
RTOL_BACKWARD = 1e-3


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def pytorch_rms_layernorm(x, weight, eps=1e-6):
    """PyTorch reference implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def opaque_rms_layernorm(x, weight, eps=1e-6):
    """Opaque Triton kernel wrapper."""
    return rms_layernorm(x, weight, eps=eps)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestRMSLayerNormForward:
    """Test forward pass precision."""

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch at Mellum scale."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

        out_pytorch = pytorch_rms_layernorm(x, weight)
        out_opaque = opaque_rms_layernorm(x, weight)

        err = precision_error(out_opaque, out_pytorch, threshold=1e-4)
        print(
            f"\nRMSNorm Forward: abs={err['abs_err']:.2e}, "
            f"rel={err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})"
        )

        assert err["rel_err"] < RTOL_FORWARD, (
            f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"
        )


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

class TestRMSLayerNormBackward:
    """Test backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch (x.grad and weight.grad)."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        # PyTorch reference
        x_pt = torch.randn(
            batch, seq, hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        w_pt = torch.randn(
            hidden, device="cuda", dtype=torch.float32, requires_grad=True
        )
        out_pt = pytorch_rms_layernorm(x_pt, w_pt)
        out_pt.sum().backward()

        # Opaque kernel
        x_op = x_pt.detach().clone().requires_grad_(True)
        w_op = w_pt.detach().clone().requires_grad_(True)
        out_op = opaque_rms_layernorm(x_op, w_op)
        out_op.sum().backward()

        x_err = precision_error(x_op.grad, x_pt.grad, threshold=1e-4)
        w_err = precision_error(w_op.grad, w_pt.grad, threshold=1e-4)

        print(f"\nRMSNorm Backward:")
        print(
            f"  x.grad:      abs={x_err['abs_err']:.2e}, "
            f"rel={x_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})"
        )
        print(
            f"  weight.grad: abs={w_err['abs_err']:.2e}, "
            f"rel={w_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})"
        )

        assert x_err["rel_err"] < RTOL_BACKWARD, (
            f"x.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        )
        assert w_err["rel_err"] < RTOL_BACKWARD, (
            f"weight.grad rel_err {w_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        )


# ---------------------------------------------------------------------------
# Vmap
# ---------------------------------------------------------------------------

class TestRMSLayerNormVmap:
    """Test vmap (per-sample gradient) precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap (forward + backward)."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        # Shared weight (not vmapped)
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

        # PyTorch vmap
        x_pt = torch.randn(
            vmap_batch, batch, seq, hidden,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )
        out_pt = torch.vmap(
            lambda x_single: pytorch_rms_layernorm(x_single, weight)
        )(x_pt)
        out_pt.sum().backward()

        # Opaque vmap
        x_op = x_pt.detach().clone().requires_grad_(True)
        out_op = torch.vmap(
            lambda x_single: opaque_rms_layernorm(x_single, weight)
        )(x_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        x_err = precision_error(x_op.grad, x_pt.grad, threshold=1e-4)

        print(f"\nRMSNorm vmap precision:")
        print(
            f"  forward: abs={fwd_err['abs_err']:.2e}, "
            f"rel={fwd_err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})"
        )
        print(
            f"  x.grad:  abs={x_err['abs_err']:.2e}, "
            f"rel={x_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})"
        )

        assert fwd_err["rel_err"] < RTOL_FORWARD, (
            f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"
        )
        assert x_err["rel_err"] < RTOL_BACKWARD, (
            f"vmap x.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        )

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory than pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x = torch.randn(
            vmap_batch, batch, seq, hidden,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )

        def pytorch_vmap_fn(x):
            return torch.vmap(
                lambda x_single: pytorch_rms_layernorm(x_single, weight)
            )(x)

        def opaque_vmap_fn(x):
            return torch.vmap(
                lambda x_single: opaque_rms_layernorm(x_single, weight)
            )(x)

        pt_stats = measure_time_and_memory(pytorch_vmap_fn, x)
        op_stats = measure_time_and_memory(opaque_vmap_fn, x)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm vmap")


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestRMSLayerNormPerformance:
    """Benchmark forward-only and forward+backward performance."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Benchmark forward-only: opaque vs pytorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        x = torch.randn(
            batch, seq, hidden, device="cuda", dtype=torch.float32,
        )
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

        # Forward-only: no requires_grad so measure_time_and_memory skips backward
        pt_stats = measure_time_and_memory(pytorch_rms_layernorm, x, weight)
        op_stats = measure_time_and_memory(opaque_rms_layernorm, x, weight)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm forward-only")

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Benchmark forward+backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch = mellum_config["batch_size"]
        seq = mellum_config["seq_len"]
        hidden = mellum_config["hidden_dim"]

        x = torch.randn(
            batch, seq, hidden,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )
        weight = torch.randn(
            hidden, device="cuda", dtype=torch.float32, requires_grad=True,
        )

        pt_stats = measure_time_and_memory(pytorch_rms_layernorm, x, weight)
        op_stats = measure_time_and_memory(opaque_rms_layernorm, x, weight)

        assert_perf_benefit(pt_stats, op_stats, label="RMSNorm forward+backward")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRMSLayerNormEdgeCases:
    """Test edge cases."""

    def test_zero_input(self):
        """Should handle zero input without NaN/Inf."""
        x = torch.zeros(2, 4, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)

        output = rms_layernorm(x, weight, eps=1e-6)
        assert not torch.isnan(output).any(), "NaN in output for zero input"
        assert not torch.isinf(output).any(), "Inf in output for zero input"

    def test_large_values(self):
        """Should handle large values with reasonable precision."""
        x = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16) * 100
        weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)

        output_triton = rms_layernorm(x, weight)
        output_pytorch = pytorch_rms_layernorm(x, weight)

        assert torch.allclose(output_triton, output_pytorch, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
