"""
LoRA Kernel Tests

Tests:
1. Forward pass vs PyTorch reference (non-vmap)
2. Backward pass vs PyTorch reference (non-vmap)
3. vmap (per-sample grad) precision vs PyTorch vmap
4. Performance: forward+backward time and memory vs PyTorch

Covers three LoRA variants:
- LoRA_W: Single linear projection with LoRA
- LoRA_QKV: Fused Q, K, V projections with LoRA
- LoRA_MLP: Fused MLP (gate, up, down) with SwiGLU and LoRA

Config: Mellum-4b scale (hidden=3072, intermediate=8256, rank=64)
"""

import math
import pytest
import torch
import torch.nn.functional as F
from torch.func import vmap, grad
from opaque.kernels.lora import Opaque_LoRA_W, Opaque_LoRA_QKV, Opaque_LoRA_MLP, ACTIVATION_SWIGLU

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

SCALING = 0.1

# Tolerances (atol + rtol formula: |a - b| <= atol + rtol * |b|)
# W/QKV: addmm_ fused accumulation differs from separate add+matmul+scale
RTOL_LORA_FWD = 1e-2
ATOL_LORA_FWD = 4e-2
RTOL_LORA_BWD = 2e-2
ATOL_LORA_BWD = 5e-6

# MLP: 3-matmul chain with SwiGLU
RTOL_LORA_MLP_FWD = 2e-2
ATOL_LORA_MLP_FWD = 1e-1
RTOL_LORA_MLP_BWD = 1e-1
ATOL_LORA_MLP_BWD = 5e-6


# ============================================================================
# Helpers
# ============================================================================

def _kaiming_weight(out_features, in_features, **kwargs):
    """Create weight matrix with Kaiming initialization (realistic scale)."""
    return torch.randn(out_features, in_features, **kwargs) * math.sqrt(2.0 / in_features)


def _lora_weight(dim, rank, **kwargs):
    """Create LoRA weight with proper initialization scale."""
    return torch.randn(dim, rank, **kwargs) * math.sqrt(1.0 / dim)


def _make_qkv_weights(hidden, rank, **kw):
    """Create Q, K, V weight sets with proper initialization."""
    Wq = _kaiming_weight(hidden, hidden, **kw)
    Aq = _lora_weight(hidden, rank, **kw)
    Bq = _lora_weight(rank, hidden, **kw)
    Wk = _kaiming_weight(hidden, hidden, **kw)
    Ak = _lora_weight(hidden, rank, **kw)
    Bk = _lora_weight(rank, hidden, **kw)
    Wv = _kaiming_weight(hidden, hidden, **kw)
    Av = _lora_weight(hidden, rank, **kw)
    Bv = _lora_weight(rank, hidden, **kw)
    return Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv


# ============================================================================
# Reference Implementations
# ============================================================================

def pytorch_lora_linear(X, W, A, B, scaling):
    """PyTorch reference LoRA linear: out = X @ W.T + (X @ A @ B) * scaling."""
    out = F.linear(X, W)
    if A is not None and B is not None:
        out = out + (X @ A) @ B * scaling
    return out


def opaque_lora_linear(X, W, A, B, scaling):
    """Opaque kernel implementation."""
    return Opaque_LoRA_W.apply(X, W, A, B, scaling)


def pytorch_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """PyTorch reference for Q, K, V projections."""
    Q = pytorch_lora_linear(X, Wq, Aq, Bq, Sq)
    K = pytorch_lora_linear(X, Wk, Ak, Bk, Sk)
    V = pytorch_lora_linear(X, Wv, Av, Bv, Sv)
    return Q, K, V


def opaque_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """Opaque kernel implementation."""
    return Opaque_LoRA_QKV.apply(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)


def pytorch_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """PyTorch reference for MLP with SwiGLU."""
    gate = pytorch_lora_linear(X, Wg, Ag, Bg, Sg)
    up = pytorch_lora_linear(X, Wu, Au, Bu, Su)
    h = F.silu(gate) * up
    out = pytorch_lora_linear(h, Wd, Ad, Bd, Sd)
    return out


def opaque_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """Opaque kernel implementation."""
    result = Opaque_LoRA_MLP.apply(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, ACTIVATION_SWIGLU)
    return result[0]


# ============================================================================
# LoRA-W Tests (single projection)
# ============================================================================

class TestLoRAWForward:
    """Test LoRA-W forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, bfloat16)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        out_pt = pytorch_lora_linear(X, W, A, B, SCALING)
        out_op = opaque_lora_linear(X, W, A, B, SCALING)

        print("\nLoRA-W Forward:")
        assert_precision(out_op, out_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="output")


class TestLoRAWBackward:
    """Test LoRA-W backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (non-vmap, test X.grad, A.grad, B.grad)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A_pt = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        B_pt = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        out_pt = pytorch_lora_linear(X_pt, W, A_pt, B_pt, SCALING)
        out_pt.mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        A_op = A_pt.detach().clone().requires_grad_(True)
        B_op = B_pt.detach().clone().requires_grad_(True)

        out_op = opaque_lora_linear(X_op, W, A_op, B_op, SCALING)
        out_op.mean().backward()

        print("\nLoRA-W Backward:")
        assert_precision(X_op.grad, X_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad")
        assert_precision(A_op.grad, A_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="A.grad")
        assert_precision(B_op.grad, B_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="B.grad")


class TestLoRAWVmapForward:
    """Test LoRA-W vmap forward precision."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        out_pt = vmap(lambda x: pytorch_lora_linear(x, W, A, B, SCALING))(X)
        out_op = vmap(lambda x: opaque_lora_linear(x, W, A, B, SCALING))(X)

        print("\nLoRA-W vmap forward:")
        assert_precision(out_op, out_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="output")


class TestLoRAWVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def f_pt(x):
            return pytorch_lora_linear(x, W, A, B, SCALING).mean()

        def f_op(x):
            return opaque_lora_linear(x, W, A, B, SCALING).mean()

        grads_pt = vmap(grad(f_pt))(X)
        grads_op = vmap(grad(f_op))(X)

        print("\nLoRA-W vmap(grad):")
        assert_precision(grads_op, grads_pt, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad")

    def test_vmap_grad_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap(grad) must be faster or use less memory."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def make_pt_fn():
            def f(x):
                return pytorch_lora_linear(x, W, A, B, SCALING).mean()
            return vmap(grad(f))

        def make_op_fn():
            def f(x):
                return opaque_lora_linear(x, W, A, B, SCALING).mean()
            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), X)
        op_stats = measure_time_and_memory(make_op_fn(), X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-W vmap(grad)", max_perf_overhead=0.20)


class TestLoRAWPerformance:
    """Test LoRA-W kernel performance (non-vmap)."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw)
        B = _lora_weight(RANK, HIDDEN, **kw)

        def pytorch_fn(x):
            return pytorch_lora_linear(x, W, A, B, SCALING)

        def opaque_fn(x):
            return opaque_lora_linear(x, W, A, B, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-W forward", max_perf_overhead=0.50)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Backward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        W = _kaiming_weight(HIDDEN, HIDDEN, **kw)
        A = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        B = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        def pytorch_fn(x, a, b):
            return pytorch_lora_linear(x, W, a, b, SCALING)

        def opaque_fn(x, a, b):
            return opaque_lora_linear(x, W, a, b, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X, A, B)
        op_stats = measure_time_and_memory(opaque_fn, X, A, B)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-W backward", max_perf_overhead=0.20)


# ============================================================================
# LoRA-QKV Tests (fused Q, K, V projections)
# ============================================================================

class TestLoRAQKVForward:
    """Test LoRA-QKV forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, check Q, K, V each)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)

        Q_pt, K_pt, V_pt = pytorch_lora_qkv(
            X, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
        )
        Q_op, K_op, V_op = opaque_lora_qkv(
            X, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
        )

        print("\nLoRA-QKV Forward:")
        assert_precision(Q_op, Q_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="Q")
        assert_precision(K_op, K_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="K")
        assert_precision(V_op, V_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="V")


class TestLoRAQKVBackward:
    """Test LoRA-QKV backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (non-vmap)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        Wq, Aq_pt, Bq_pt, Wk, Ak_pt, Bk_pt, Wv, Av_pt, Bv_pt = _make_qkv_weights(HIDDEN, RANK, **kw)
        Aq_pt = Aq_pt.requires_grad_(True)
        Bq_pt = Bq_pt.requires_grad_(True)
        Ak_pt = Ak_pt.requires_grad_(True)
        Bk_pt = Bk_pt.requires_grad_(True)
        Av_pt = Av_pt.requires_grad_(True)
        Bv_pt = Bv_pt.requires_grad_(True)

        Q_pt, K_pt, V_pt = pytorch_lora_qkv(
            X_pt, Wq, Aq_pt, Bq_pt, SCALING, Wk, Ak_pt, Bk_pt, SCALING, Wv, Av_pt, Bv_pt, SCALING
        )
        (Q_pt + K_pt + V_pt).mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        Aq_op = Aq_pt.detach().clone().requires_grad_(True)
        Bq_op = Bq_pt.detach().clone().requires_grad_(True)
        Ak_op = Ak_pt.detach().clone().requires_grad_(True)
        Bk_op = Bk_pt.detach().clone().requires_grad_(True)
        Av_op = Av_pt.detach().clone().requires_grad_(True)
        Bv_op = Bv_pt.detach().clone().requires_grad_(True)

        Q_op, K_op, V_op = opaque_lora_qkv(
            X_op, Wq, Aq_op, Bq_op, SCALING, Wk, Ak_op, Bk_op, SCALING, Wv, Av_op, Bv_op, SCALING
        )
        (Q_op + K_op + V_op).mean().backward()

        print("\nLoRA-QKV Backward:")
        assert_precision(X_op.grad, X_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad")
        assert_precision(Aq_op.grad, Aq_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="Aq.grad")
        assert_precision(Bq_op.grad, Bq_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="Bq.grad")
        assert_precision(Ak_op.grad, Ak_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="Ak.grad")
        assert_precision(Bk_op.grad, Bk_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="Bk.grad")
        assert_precision(Av_op.grad, Av_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="Av.grad")
        assert_precision(Bv_op.grad, Bv_pt.grad, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="Bv.grad")


class TestLoRAQKVVmapForward:
    """Test LoRA-QKV vmap forward precision."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        Q_pt, K_pt, V_pt = vmap(
            lambda x: pytorch_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
        )(X)
        Q_op, K_op, V_op = vmap(
            lambda x: opaque_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
        )(X)

        print("\nLoRA-QKV vmap forward:")
        assert_precision(Q_op, Q_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="Q")
        assert_precision(K_op, K_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="K")
        assert_precision(V_op, V_pt, rtol=RTOL_LORA_FWD, atol=ATOL_LORA_FWD, label="V")


class TestLoRAQKVVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def f_pt(x):
            Q, K, V = pytorch_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
            return (Q + K + V).mean()

        def f_op(x):
            Q, K, V = opaque_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
            return (Q + K + V).mean()

        grads_pt = vmap(grad(f_pt))(X)
        grads_op = vmap(grad(f_op))(X)

        print("\nLoRA-QKV vmap(grad):")
        assert_precision(grads_op, grads_pt, rtol=RTOL_LORA_BWD, atol=ATOL_LORA_BWD, label="X.grad")

    def test_vmap_grad_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap(grad) must be faster or use less memory."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def make_pt_fn():
            def f(x):
                Q, K, V = pytorch_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
                return (Q + K + V).mean()
            return vmap(grad(f))

        def make_op_fn():
            def f(x):
                Q, K, V = opaque_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
                return (Q + K + V).mean()
            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), X)
        op_stats = measure_time_and_memory(make_op_fn(), X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-QKV vmap(grad)", max_perf_overhead=0.20)


class TestLoRAQKVPerformance:
    """Test LoRA-QKV kernel performance (non-vmap)."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)

        def pytorch_fn(x):
            Q, K, V = pytorch_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
            return Q + K + V

        def opaque_fn(x):
            Q, K, V = opaque_lora_qkv(
                x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
            )
            return Q + K + V

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-QKV forward", max_perf_overhead=0.20)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Backward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)
        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = _make_qkv_weights(HIDDEN, RANK, **kw)
        Aq = Aq.requires_grad_(True)
        Bq = Bq.requires_grad_(True)
        Ak = Ak.requires_grad_(True)
        Bk = Bk.requires_grad_(True)
        Av = Av.requires_grad_(True)
        Bv = Bv.requires_grad_(True)

        def pytorch_fn(x, aq, bq, ak, bk, av, bv):
            Q, K, V = pytorch_lora_qkv(
                x, Wq, aq, bq, SCALING, Wk, ak, bk, SCALING, Wv, av, bv, SCALING
            )
            return Q + K + V

        def opaque_fn(x, aq, bq, ak, bk, av, bv):
            Q, K, V = opaque_lora_qkv(
                x, Wq, aq, bq, SCALING, Wk, ak, bk, SCALING, Wv, av, bv, SCALING
            )
            return Q + K + V

        pt_stats = measure_time_and_memory(pytorch_fn, X, Aq, Bq, Ak, Bk, Av, Bv)
        op_stats = measure_time_and_memory(opaque_fn, X, Aq, Bq, Ak, Bk, Av, Bv)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-QKV backward", max_perf_overhead=0.20)


# ============================================================================
# LoRA-MLP Tests (fused gate, up, down with SwiGLU)
# ============================================================================

class TestLoRAMLPForward:
    """Test LoRA-MLP forward pass precision."""

    def test_forward_matches_pytorch(self, assert_precision, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, bfloat16)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        out_pt = pytorch_lora_mlp(
            X, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )
        out_op = opaque_lora_mlp(
            X, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )

        print("\nLoRA-MLP Forward:")
        assert_precision(out_op, out_pt, rtol=RTOL_LORA_MLP_FWD, atol=ATOL_LORA_MLP_FWD, label="output")


class TestLoRAMLPBackward:
    """Test LoRA-MLP backward pass precision."""

    def test_backward_matches_pytorch(self, assert_precision, mellum_config):
        """Backward: opaque vs pytorch (non-vmap)."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag_pt = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bg_pt = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au_pt = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bu_pt = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad_pt = _lora_weight(INTERMEDIATE, RANK, **kw).requires_grad_(True)
        Bd_pt = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        out_pt = pytorch_lora_mlp(
            X_pt, Wg, Ag_pt, Bg_pt, SCALING, Wu, Au_pt, Bu_pt, SCALING, Wd, Ad_pt, Bd_pt, SCALING
        )
        out_pt.mean().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        Ag_op = Ag_pt.detach().clone().requires_grad_(True)
        Bg_op = Bg_pt.detach().clone().requires_grad_(True)
        Au_op = Au_pt.detach().clone().requires_grad_(True)
        Bu_op = Bu_pt.detach().clone().requires_grad_(True)
        Ad_op = Ad_pt.detach().clone().requires_grad_(True)
        Bd_op = Bd_pt.detach().clone().requires_grad_(True)

        out_op = opaque_lora_mlp(
            X_op, Wg, Ag_op, Bg_op, SCALING, Wu, Au_op, Bu_op, SCALING, Wd, Ad_op, Bd_op, SCALING
        )
        out_op.mean().backward()

        print("\nLoRA-MLP Backward:")
        assert_precision(X_op.grad, X_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="X.grad")
        assert_precision(Ag_op.grad, Ag_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="Ag.grad")
        assert_precision(Bg_op.grad, Bg_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="Bg.grad")
        assert_precision(Au_op.grad, Au_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="Au.grad")
        assert_precision(Bu_op.grad, Bu_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="Bu.grad")
        assert_precision(Ad_op.grad, Ad_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="Ad.grad")
        assert_precision(Bd_op.grad, Bd_pt.grad, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="Bd.grad")


class TestLoRAMLPVmapForward:
    """Test LoRA-MLP vmap forward precision."""

    def test_vmap_forward_precision(self, assert_precision, mellum_config):
        """Batched forward: opaque Triton vmap vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        out_pt = vmap(
            lambda x: pytorch_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)
        )(X)
        out_op = vmap(
            lambda x: opaque_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)
        )(X)

        print("\nLoRA-MLP vmap forward:")
        assert_precision(out_op, out_pt, rtol=RTOL_LORA_MLP_FWD, atol=ATOL_LORA_MLP_FWD, label="output")


class TestLoRAMLPVmapGrad:
    """Test vmap(grad): per-example gradients — the DP-SGD path."""

    def test_vmap_grad_precision(self, assert_precision, mellum_config):
        """Per-example gradients: opaque Triton vs PyTorch reference."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def f_pt(x):
            return pytorch_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING).mean()

        def f_op(x):
            return opaque_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING).mean()

        grads_pt = vmap(grad(f_pt))(X)
        grads_op = vmap(grad(f_op))(X)

        print("\nLoRA-MLP vmap(grad):")
        assert_precision(grads_op, grads_pt, rtol=RTOL_LORA_MLP_BWD, atol=ATOL_LORA_MLP_BWD, label="X.grad")

    def test_vmap_grad_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Triton vmap(grad) must be faster or use less memory."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, **kw)

        def make_pt_fn():
            def f(x):
                return pytorch_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING).mean()
            return vmap(grad(f))

        def make_op_fn():
            def f(x):
                return opaque_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING).mean()
            return vmap(grad(f))

        pt_stats = measure_time_and_memory(make_pt_fn(), X)
        op_stats = measure_time_and_memory(make_op_fn(), X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-MLP vmap(grad)", max_perf_overhead=0.20)


class TestLoRAMLPPerformance:
    """Test LoRA-MLP kernel performance (non-vmap)."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw)
        Bd = _lora_weight(RANK, HIDDEN, **kw)

        def pytorch_fn(x):
            return pytorch_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)

        def opaque_fn(x):
            return opaque_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-MLP forward", max_perf_overhead=0.50)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Backward performance: opaque vs pytorch."""
        BATCH = mellum_config["batch_size"]
        SEQ = mellum_config["seq_len"]
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)
        kw = dict(device="cuda", dtype=torch.bfloat16)

        X = torch.randn(BATCH, SEQ, HIDDEN, **kw, requires_grad=True)

        Wg = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Ag = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bg = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wu = _kaiming_weight(INTERMEDIATE, HIDDEN, **kw)
        Au = _lora_weight(HIDDEN, RANK, **kw).requires_grad_(True)
        Bu = _lora_weight(RANK, INTERMEDIATE, **kw).requires_grad_(True)

        Wd = _kaiming_weight(HIDDEN, INTERMEDIATE, **kw)
        Ad = _lora_weight(INTERMEDIATE, RANK, **kw).requires_grad_(True)
        Bd = _lora_weight(RANK, HIDDEN, **kw).requires_grad_(True)

        def pytorch_fn(x, ag, bg, au, bu, ad, bd):
            return pytorch_lora_mlp(x, Wg, ag, bg, SCALING, Wu, au, bu, SCALING, Wd, ad, bd, SCALING)

        def opaque_fn(x, ag, bg, au, bu, ad, bd):
            return opaque_lora_mlp(x, Wg, ag, bg, SCALING, Wu, au, bu, SCALING, Wd, ad, bd, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X, Ag, Bg, Au, Bu, Ad, Bd)
        op_stats = measure_time_and_memory(opaque_fn, X, Ag, Bg, Au, Bu, Ad, Bd)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-MLP backward", max_perf_overhead=0.20)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
