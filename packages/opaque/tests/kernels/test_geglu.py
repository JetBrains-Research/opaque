"""
GeGLU Kernel Tests (Exact and Approx)

Tests:
1. Forward pass vs PyTorch reference
2. Backward pass vs PyTorch reference
3. vmap (per-sample grad) vs PyTorch vmap
4. Performance: forward+backward time and memory vs PyTorch

Config: Mellum-4b scale (intermediate_dim=8256)
"""

import pytest
import torch
import torch.nn.functional as F

from opaque.kernels.geglu import NewStyleGeGLUExact, NewStyleGeGLUApprox

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# GeGLU uses tanh approximation in Triton, so tolerances are wider
RTOL_FORWARD = 5e-3
RTOL_BACKWARD = 2e-3


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_geglu_exact(gate, up):
    """PyTorch reference: GeGLU = gelu(gate) * up."""
    return F.gelu(gate, approximate='none') * up


def pytorch_geglu_approx(gate, up):
    """PyTorch reference: GeGLU (tanh approx) = gelu_tanh(gate) * up."""
    return F.gelu(gate, approximate='tanh') * up


def opaque_geglu_exact(gate, up):
    """Opaque Triton kernel (exact)."""
    return NewStyleGeGLUExact.apply(gate, up)


def opaque_geglu_approx(gate, up):
    """Opaque Triton kernel (tanh approx)."""
    return NewStyleGeGLUApprox.apply(gate, up)


# ============================================================================
# Exact GeGLU Tests
# ============================================================================

class TestGeGLUExactForward:
    """Test GeGLU exact forward pass."""

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)

        out_pytorch = pytorch_geglu_exact(gate, up)
        out_opaque = opaque_geglu_exact(gate, up)

        err = precision_error(out_opaque, out_pytorch, threshold=1e-4)
        print(f"\nGeGLU Exact Forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_FORWARD, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"


class TestGeGLUExactBackward:
    """Test GeGLU exact backward pass."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate_pt = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up_pt = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        out_pt = pytorch_geglu_exact(gate_pt, up_pt)
        out_pt.sum().backward()

        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opaque_geglu_exact(gate_op, up_op)
        out_op.sum().backward()

        gate_err = precision_error(gate_op.grad, gate_pt.grad, threshold=1e-4)
        up_err = precision_error(up_op.grad, up_pt.grad, threshold=1e-4)

        print(f"\nGeGLU Exact Backward:")
        print(f"  gate.grad: abs={gate_err['abs_err']:.2e}, rel={gate_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")
        print(f"  up.grad:   abs={up_err['abs_err']:.2e}, rel={up_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert gate_err["rel_err"] < RTOL_BACKWARD, f"gate.grad rel_err {gate_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        assert up_err["rel_err"] < RTOL_BACKWARD, f"up.grad rel_err {up_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"


class TestGeGLUExactVmap:
    """Test GeGLU exact vmap."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        gate_pt = gate.detach().clone().requires_grad_(True)
        up_pt = up.detach().clone().requires_grad_(True)
        out_pt = torch.vmap(pytorch_geglu_exact)(gate_pt, up_pt)
        out_pt.sum().backward()

        gate_op = gate.detach().clone().requires_grad_(True)
        up_op = up.detach().clone().requires_grad_(True)
        out_op = torch.vmap(opaque_geglu_exact)(gate_op, up_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        gate_err = precision_error(gate_op.grad, gate_pt.grad, threshold=1e-4)
        up_err = precision_error(up_op.grad, up_pt.grad, threshold=1e-4)

        print(f"\nGeGLU Exact vmap:")
        print(f"  forward:   abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")
        print(f"  gate.grad: abs={gate_err['abs_err']:.2e}, rel={gate_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")
        print(f"  up.grad:   abs={up_err['abs_err']:.2e}, rel={up_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_FORWARD, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"
        assert gate_err["rel_err"] < RTOL_BACKWARD, f"vmap gate.grad rel_err {gate_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        assert up_err["rel_err"] < RTOL_BACKWARD, f"vmap up.grad rel_err {up_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"

    def test_vmap_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """vmap: opaque should be faster or use less memory."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(g, u):
            return torch.vmap(pytorch_geglu_exact)(g, u)

        def opaque_fn(g, u):
            return torch.vmap(opaque_geglu_exact)(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Exact vmap")


# ============================================================================
# Exact GeGLU Performance Tests
# ============================================================================

class TestGeGLUExactPerformance:
    """Test GeGLU exact kernel performance (non-vmap)."""

    def test_forward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Forward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)

        def pytorch_fn(g, u):
            return pytorch_geglu_exact(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_exact(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Exact forward", max_perf_overhead=0.30)

    def test_backward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Backward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(g, u):
            return pytorch_geglu_exact(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_exact(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Exact backward")


# ============================================================================
# Approx GeGLU Tests
# ============================================================================

class TestGeGLUApproxForward:
    """Test GeGLU approx forward pass."""

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)

        out_pytorch = pytorch_geglu_approx(gate, up)
        out_opaque = opaque_geglu_approx(gate, up)

        err = precision_error(out_opaque, out_pytorch, threshold=1e-4)
        print(f"\nGeGLU Approx Forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")

        assert err["rel_err"] < RTOL_FORWARD, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"


class TestGeGLUApproxBackward:
    """Test GeGLU approx backward pass."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate_pt = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up_pt = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        out_pt = pytorch_geglu_approx(gate_pt, up_pt)
        out_pt.sum().backward()

        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opaque_geglu_approx(gate_op, up_op)
        out_op.sum().backward()

        gate_err = precision_error(gate_op.grad, gate_pt.grad, threshold=1e-4)
        up_err = precision_error(up_op.grad, up_pt.grad, threshold=1e-4)

        print(f"\nGeGLU Approx Backward:")
        print(f"  gate.grad: abs={gate_err['abs_err']:.2e}, rel={gate_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")
        print(f"  up.grad:   abs={up_err['abs_err']:.2e}, rel={up_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert gate_err["rel_err"] < RTOL_BACKWARD, f"gate.grad rel_err {gate_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        assert up_err["rel_err"] < RTOL_BACKWARD, f"up.grad rel_err {up_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"


class TestGeGLUApproxVmap:
    """Test GeGLU approx vmap."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        gate_pt = gate.detach().clone().requires_grad_(True)
        up_pt = up.detach().clone().requires_grad_(True)
        out_pt = torch.vmap(pytorch_geglu_approx)(gate_pt, up_pt)
        out_pt.sum().backward()

        gate_op = gate.detach().clone().requires_grad_(True)
        up_op = up.detach().clone().requires_grad_(True)
        out_op = torch.vmap(opaque_geglu_approx)(gate_op, up_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        gate_err = precision_error(gate_op.grad, gate_pt.grad, threshold=1e-4)
        up_err = precision_error(up_op.grad, up_pt.grad, threshold=1e-4)

        print(f"\nGeGLU Approx vmap:")
        print(f"  forward:   abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_FORWARD:.0e})")
        print(f"  gate.grad: abs={gate_err['abs_err']:.2e}, rel={gate_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")
        print(f"  up.grad:   abs={up_err['abs_err']:.2e}, rel={up_err['rel_err']:.2e} (target: <{RTOL_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_FORWARD, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_FORWARD:.0e}"
        assert gate_err["rel_err"] < RTOL_BACKWARD, f"vmap gate.grad rel_err {gate_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"
        assert up_err["rel_err"] < RTOL_BACKWARD, f"vmap up.grad rel_err {up_err['rel_err']:.2e} >= {RTOL_BACKWARD:.0e}"

    def test_vmap_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """vmap: opaque should be faster or use less memory."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(vmap_batch, batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(g, u):
            return torch.vmap(pytorch_geglu_approx)(g, u)

        def opaque_fn(g, u):
            return torch.vmap(opaque_geglu_approx)(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Approx vmap")


# ============================================================================
# Approx GeGLU Performance Tests
# ============================================================================

class TestGeGLUApproxPerformance:
    """Test GeGLU approx kernel performance (non-vmap)."""

    def test_forward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Forward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32)

        def pytorch_fn(g, u):
            return pytorch_geglu_approx(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_approx(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Approx forward", max_perf_overhead=0.30)

    def test_backward_performance(self, mellum_config, measure_time_and_memory, assert_perf_benefit):
        """Backward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = mellum_config["batch_size"], mellum_config["seq_len"], mellum_config["intermediate_dim"]

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(g, u):
            return pytorch_geglu_approx(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_approx(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Approx backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
