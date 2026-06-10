"""RMSNorm Triton kernel tests.

Tests:
1. Forward pass vs PyTorch reference (Llama + Gemma variants)
2. Backward pass vs PyTorch reference (Llama)
3. vmap forward: Triton vmap vs PyTorch vmap
4. vmap(grad): per-example gradients (DP-SGD path)
5. Forward, backward, vmap forward, and vmap(grad) performance benchmarks

Performance: opaque must beat reference on time (within tolerance) or peak memory.

Config: Mellum-4b ``hidden_dim`` and sequence length via ``mellum_config``
(``conftest``), comparable to ``train_causal_lm.py --preset mellum-kstack`` geometry.
"""

import pytest
import torch
from torch.func import grad, vmap

pytest.importorskip("triton")

from opaque.api.patches.kernels.rms_norm import Opaque_RMSNorm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

RTOL_F = 0.05
ATOL_F = 0.05
RTOL_B = 2e-2
ATOL_B = 1e-3


def ref_llama_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = xf * inv
    return (normed * w.float()).to(x.dtype)


def ref_gemma_rms(
    x: torch.Tensor, w: torch.Tensor, eps: float, offset: float
) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    normed = xf * inv
    scale = w.float() + offset
    return (normed * scale).to(x.dtype)


def opaque_llama(x, w, eps=1e-6):
    return Opaque_RMSNorm.apply(x, w, eps, 0.0, "llama", False, None)


def opaque_gemma(x, w, eps=1e-6, offset=1.0):
    return Opaque_RMSNorm.apply(x, w, eps, offset, "gemma", False, None)


class TestRMSNormForward:
    def test_llama_forward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(0)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5
        y_o = opaque_llama(x, w, eps)
        y_r = ref_llama_rms(x, w, eps)
        assert_precision(y_o, y_r, rtol=RTOL_F, atol=ATOL_F, label="llama fwd")

    def test_gemma_forward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(1)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-6
        off = 1.0
        y_o = opaque_gemma(x, w, eps, off)
        y_r = ref_gemma_rms(x, w, eps, off)
        assert_precision(y_o, y_r, rtol=RTOL_F, atol=ATOL_F, label="gemma fwd")


class TestRMSNormBackward:
    def test_llama_backward_bf16(self, assert_precision, mellum_config):
        torch.manual_seed(2)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        eps = 1e-5

        x0 = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w0 = torch.randn(h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y0 = ref_llama_rms(x0, w0, eps)
        y0.mean().backward()

        x1 = x0.detach().clone().requires_grad_(True)
        w1 = w0.detach().clone().requires_grad_(True)
        y1 = opaque_llama(x1, w1, eps)
        y1.mean().backward()

        assert_precision(x1.grad, x0.grad, rtol=RTOL_B, atol=ATOL_B, label="dx llama")
        assert_precision(w1.grad, w0.grad, rtol=RTOL_B, atol=ATOL_B, label="dw llama")


class TestRMSNormVmapForward:
    """vmap over microbatch dim: Triton vmap vs PyTorch vmap (Llama)."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def ref_call(inp, wt):
            return ref_llama_rms(inp, wt, eps)

        def op_call(inp, wt):
            return opaque_llama(inp, wt, eps)

        y_pt = vmap(ref_call, in_dims=(0, None))(x, w)
        y_op = vmap(op_call, in_dims=(0, None))(x, w)

        print("\nvmap forward precision check:")
        assert_precision(y_op, y_pt, rtol=RTOL_F, atol=ATOL_F, label="vmap forward")

    def test_vmap_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def pt_fn(inp, wt):
            return vmap(lambda i, t: ref_llama_rms(i, t, eps), in_dims=(0, None))(
                inp, wt
            )

        def op_fn(inp, wt):
            return vmap(lambda i, t: opaque_llama(i, t, eps), in_dims=(0, None))(
                inp, wt
            )

        pt_stats = measure_time_and_memory(pt_fn, x, w)
        op_stats = measure_time_and_memory(op_fn, x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm vmap forward")


class TestRMSNormVmapGradPerExampleDW:
    """Regression test: vmap(grad) per-example dW must match an eager per-sample loop.

    Under DP-SGD with a trainable RMSNorm weight (full fine-tuning), the
    per-example weight gradient fed to the DP clipper must contain only that
    example's contribution.  The bug was that _RMSNormBackward.vmap summed dW
    over the entire merged (B*T, H) batch and returned it with out_dim=None,
    giving every example the batch-sum as its "per-example" dW.

    Small shapes are used deliberately so the test runs without the 24 GB GPU
    required by the mellum stress suite.
    """

    @pytest.mark.parametrize("casting_mode", ["llama", "gemma"])
    @pytest.mark.parametrize("in_place", [False, True])
    def test_vmap_grad_matches_eager_loop(self, casting_mode, in_place):
        torch.manual_seed(7)
        B, T, H = 4, 8, 64
        eps = 1e-5
        offset = 1.0 if casting_mode == "gemma" else 0.0

        x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(H, device="cuda", dtype=torch.bfloat16)

        def f(xi, wt):
            return Opaque_RMSNorm.apply(
                xi, wt, eps, offset, casting_mode, in_place, None
            ).mean()

        # vmap path (the fix under test)
        gx_vmap, gw_vmap = vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))(x, w)

        # Eager per-sample loop (ground truth)
        gx_eager = torch.stack([grad(f, argnums=0)(x[i], w) for i in range(B)])
        gw_eager = torch.stack([grad(f, argnums=1)(x[i], w) for i in range(B)])

        torch.testing.assert_close(
            gx_vmap,
            gx_eager,
            rtol=RTOL_B,
            atol=ATOL_B,
            msg=f"per-example dX mismatch ({casting_mode})",
        )
        torch.testing.assert_close(
            gw_vmap,
            gw_eager,
            rtol=RTOL_B,
            atol=ATOL_B,
            msg=f"per-example dW mismatch ({casting_mode}) — "
            "vmap dW must not be the batch-sum",
        )


class TestRMSNormVmapGrad:
    """vmap(grad): per-example gradients (Llama)."""

    def test_vmap_grad_llama(self, assert_precision, mellum_config):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def f_pt(inp, wt):
            return ref_llama_rms(inp, wt, eps).mean()

        def f_op(inp, wt):
            return opaque_llama(inp, wt, eps).mean()

        gx_pt, gw_pt = vmap(grad(f_pt, argnums=(0, 1)), in_dims=(0, None))(x, w)
        gx_op, gw_op = vmap(grad(f_op, argnums=(0, 1)), in_dims=(0, None))(x, w)

        print("\nvmap(grad) precision check:")
        assert_precision(gx_op, gx_pt, rtol=RTOL_B, atol=ATOL_B, label="vmap gx")
        assert_precision(gw_op, gw_pt, rtol=RTOL_B, atol=ATOL_B, label="vmap gw")

    def test_vmap_grad_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        n = mellum_config["vmap_batch"]
        eps = 1e-5

        x = torch.randn(n, b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)

        def make_pt_fn():
            def f(inp, wt):
                return ref_llama_rms(inp, wt, eps).mean()

            return vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))

        def make_op_fn():
            def f(inp, wt):
                return opaque_llama(inp, wt, eps).mean()

            return vmap(grad(f, argnums=(0, 1)), in_dims=(0, None))

        pt_stats = measure_time_and_memory(make_pt_fn(), x, w)
        op_stats = measure_time_and_memory(make_op_fn(), x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm vmap(grad)")


class TestRMSNormPerformance:
    """Benchmark forward-only and forward+backward vs PyTorch reference."""

    def test_forward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(b, s, h, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16)
        eps = 1e-5

        def pt_fn(inp, wt):
            return ref_llama_rms(inp, wt, eps)

        def op_fn(inp, wt):
            return opaque_llama(inp, wt, eps)

        pt_stats = measure_time_and_memory(pt_fn, x, w)
        op_stats = measure_time_and_memory(op_fn, x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm forward")

    def test_backward_performance(
        self, measure_time_and_memory, assert_perf_benefit, mellum_config
    ):
        torch.manual_seed(42)
        h = mellum_config["hidden_dim"]
        b, s = mellum_config["batch_size"], mellum_config["seq_len"]
        x = torch.randn(
            b, s, h, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        eps = 1e-5

        def pt_fn(inp, wt):
            return ref_llama_rms(inp, wt, eps)

        def op_fn(inp, wt):
            return opaque_llama(inp, wt, eps)

        pt_stats = measure_time_and_memory(pt_fn, x, w)
        op_stats = measure_time_and_memory(op_fn, x, w)
        assert_perf_benefit(pt_stats, op_stats, label="rmsnorm backward")
