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
from torch.func import vmap, grad

pytest.importorskip("triton")

from opaque.compat.kernels.geglu import Opaque_GeGLU_Exact, Opaque_GeGLU_Approx

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# GeGLU uses tanh approximation in Triton, so tolerances are tighter with rtol/atol
RTOL_FORWARD = 2e-3
ATOL_FORWARD = 1e-4
RTOL_BACKWARD = 1e-2
ATOL_BACKWARD = 5e-9


# ============================================================================
# Reference Implementations
# ============================================================================


def pytorch_geglu_exact(gate, up):
    """PyTorch reference: GeGLU = gelu(gate) * up."""
    return F.gelu(gate, approximate="none") * up


def pytorch_geglu_approx(gate, up):
    """PyTorch reference: GeGLU (tanh approx) = gelu_tanh(gate) * up."""
    return F.gelu(gate, approximate="tanh") * up


def opaque_geglu_exact(gate, up):
    """Opaque Triton kernel (exact)."""
    return Opaque_GeGLU_Exact.apply(gate, up)


def opaque_geglu_approx(gate, up):
    """Opaque Triton kernel (tanh approx)."""
    return Opaque_GeGLU_Approx.apply(gate, up)


# ============================================================================
# Exact GeGLU Tests
# ============================================================================


class TestGeGLUExactForward:
    """Test GeGLU exact forward pass."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        out_pytorch = pytorch_geglu_exact(gate, up)
        out_opaque = opaque_geglu_exact(gate, up)

        print("\nGeGLU Exact Forward")
        assert_precision(
            out_opaque,
            out_pytorch,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="forward output",
        )


class TestGeGLUExactBackward:
    """Test GeGLU exact backward pass."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out_pt = pytorch_geglu_exact(gate_pt, up_pt)
        out_pt.mean().backward()

        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opaque_geglu_exact(gate_op, up_op)
        out_op.mean().backward()

        print("\nGeGLU Exact Backward")
        assert_precision(
            gate_op.grad,
            gate_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="gate.grad",
        )
        assert_precision(
            up_op.grad,
            up_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="up.grad",
        )


class TestGeGLUExactVmapForward:
    """Test GeGLU exact vmap forward."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        out_pt = vmap(pytorch_geglu_exact)(gate, up)
        out_op = vmap(opaque_geglu_exact)(gate, up)

        print("\nGeGLU Exact vmap forward")
        assert_precision(
            out_op,
            out_pt,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="vmap forward output",
        )

    def test_vmap_forward_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Triton vmap forward must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        pt_stats = measure_time_and_memory(
            lambda g, u: vmap(pytorch_geglu_exact)(g, u), gate, up
        )
        op_stats = measure_time_and_memory(
            lambda g, u: vmap(opaque_geglu_exact)(g, u), gate, up
        )

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Exact vmap forward")


class TestGeGLUExactVmapGrad:
    """Test GeGLU exact vmap(grad): the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def f_pt(g, u):
            return pytorch_geglu_exact(g, u).mean()

        def f_op(g, u):
            return opaque_geglu_exact(g, u).mean()

        grads_pt_gate, grads_pt_up = vmap(grad(f_pt, argnums=(0, 1)))(gate, up)
        grads_op_gate, grads_op_up = vmap(grad(f_op, argnums=(0, 1)))(gate, up)

        print("\nGeGLU Exact vmap(grad)")
        assert_precision(
            grads_op_gate,
            grads_pt_gate,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="gate grad",
        )
        assert_precision(
            grads_op_up,
            grads_pt_up,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="up grad",
        )

    def test_vmap_grad_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Triton vmap(grad) must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def make_pt_fn():
            def f(g, u):
                return pytorch_geglu_exact(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        def make_op_fn():
            def f(g, u):
                return opaque_geglu_exact(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        pt_stats = measure_time_and_memory(make_pt_fn(), gate, up)
        op_stats = measure_time_and_memory(make_op_fn(), gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Exact vmap(grad)")


# ============================================================================
# Exact GeGLU Performance Tests
# ============================================================================


class TestGeGLUExactPerformance:
    """Test GeGLU exact kernel performance (non-vmap)."""

    def test_forward_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Forward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        def pytorch_fn(g, u):
            return pytorch_geglu_exact(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_exact(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(
            pt_stats, op_stats, label="GeGLU Exact forward", max_perf_overhead=0.30
        )

    def test_backward_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Backward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

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

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        out_pytorch = pytorch_geglu_approx(gate, up)
        out_opaque = opaque_geglu_approx(gate, up)

        print("\nGeGLU Approx Forward")
        assert_precision(
            out_opaque,
            out_pytorch,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="forward output",
        )


class TestGeGLUApproxBackward:
    """Test GeGLU approx backward pass."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up_pt = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out_pt = pytorch_geglu_approx(gate_pt, up_pt)
        out_pt.mean().backward()

        gate_op = gate_pt.detach().clone().requires_grad_(True)
        up_op = up_pt.detach().clone().requires_grad_(True)
        out_op = opaque_geglu_approx(gate_op, up_op)
        out_op.mean().backward()

        print("\nGeGLU Approx Backward")
        assert_precision(
            gate_op.grad,
            gate_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="gate.grad",
        )
        assert_precision(
            up_op.grad,
            up_pt.grad,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="up.grad",
        )


class TestGeGLUApproxVmapForward:
    """Test GeGLU approx vmap forward."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        out_pt = vmap(pytorch_geglu_approx)(gate, up)
        out_op = vmap(opaque_geglu_approx)(gate, up)

        print("\nGeGLU Approx vmap forward")
        assert_precision(
            out_op,
            out_pt,
            rtol=RTOL_FORWARD,
            atol=ATOL_FORWARD,
            label="vmap forward output",
        )

    def test_vmap_forward_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Triton vmap forward must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        pt_stats = measure_time_and_memory(
            lambda g, u: vmap(pytorch_geglu_approx)(g, u), gate, up
        )
        op_stats = measure_time_and_memory(
            lambda g, u: vmap(opaque_geglu_approx)(g, u), gate, up
        )

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Approx vmap forward")


class TestGeGLUApproxVmapGrad:
    """Test GeGLU approx vmap(grad): the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def f_pt(g, u):
            return pytorch_geglu_approx(g, u).mean()

        def f_op(g, u):
            return opaque_geglu_approx(g, u).mean()

        grads_pt_gate, grads_pt_up = vmap(grad(f_pt, argnums=(0, 1)))(gate, up)
        grads_op_gate, grads_op_up = vmap(grad(f_op, argnums=(0, 1)))(gate, up)

        print("\nGeGLU Approx vmap(grad)")
        assert_precision(
            grads_op_gate,
            grads_pt_gate,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="gate grad",
        )
        assert_precision(
            grads_op_up,
            grads_pt_up,
            rtol=RTOL_BACKWARD,
            atol=ATOL_BACKWARD,
            label="up grad",
        )

    def test_vmap_grad_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Triton vmap(grad) must be faster or use less memory than PyTorch."""
        torch.manual_seed(42)
        vmap_batch = mellum_config["vmap_batch"]
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )
        up = torch.randn(
            vmap_batch, batch, seq, dim, device="cuda", dtype=torch.bfloat16
        )

        def make_pt_fn():
            def f(g, u):
                return pytorch_geglu_approx(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        def make_op_fn():
            def f(g, u):
                return opaque_geglu_approx(g, u).mean()

            return vmap(grad(f, argnums=(0, 1)))

        pt_stats = measure_time_and_memory(make_pt_fn(), gate, up)
        op_stats = measure_time_and_memory(make_op_fn(), gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Approx vmap(grad)")


# ============================================================================
# Approx GeGLU Performance Tests
# ============================================================================


class TestGeGLUApproxPerformance:
    """Test GeGLU approx kernel performance (non-vmap)."""

    def test_forward_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Forward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(batch, seq, dim, device="cuda", dtype=torch.bfloat16)

        def pytorch_fn(g, u):
            return pytorch_geglu_approx(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_approx(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(
            pt_stats, op_stats, label="GeGLU Approx forward", max_perf_overhead=0.30
        )

    def test_backward_performance(
        self, mellum_config, measure_time_and_memory, assert_perf_benefit
    ):
        """Backward performance: opaque vs pytorch."""
        torch.manual_seed(42)
        batch, seq, dim = (
            mellum_config["batch_size"],
            mellum_config["seq_len"],
            mellum_config["intermediate_dim"],
        )

        gate = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        up = torch.randn(
            batch, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        def pytorch_fn(g, u):
            return pytorch_geglu_approx(g, u)

        def opaque_fn(g, u):
            return opaque_geglu_approx(g, u)

        pt_stats = measure_time_and_memory(pytorch_fn, gate, up)
        op_stats = measure_time_and_memory(opaque_fn, gate, up)

        assert_perf_benefit(pt_stats, op_stats, label="GeGLU Approx backward")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
