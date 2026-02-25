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

import pytest
import torch
import torch.nn.functional as F
from opaque.kernels.lora import NewStyleLoRAW, NewStyleLoRAQKV, NewStyleLoRAMLP, ACTIVATION_SWIGLU

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Test dimensions (Mellum-4b scale, manageable for GPU memory)
BATCH = 2
SEQ = 128
SCALING = 0.1

# Tolerances
# W/QKV: identical PyTorch ops, but addmm_ reorders accumulation in backward
RTOL_LORA = 1e-5
RTOL_LORA_BACKWARD = 1e-2

# MLP: Triton activation diffs get amplified through large matrix projections
RTOL_LORA_MLP = 5e-2
RTOL_LORA_MLP_BACKWARD = 5e-1


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
    return NewStyleLoRAW.apply(X, W, A, B, scaling)


def pytorch_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """PyTorch reference for Q, K, V projections."""
    Q = pytorch_lora_linear(X, Wq, Aq, Bq, Sq)
    K = pytorch_lora_linear(X, Wk, Ak, Bk, Sk)
    V = pytorch_lora_linear(X, Wv, Av, Bv, Sv)
    return Q, K, V


def opaque_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """Opaque kernel implementation."""
    return NewStyleLoRAQKV.apply(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)


def pytorch_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """PyTorch reference for MLP with SwiGLU."""
    gate = pytorch_lora_linear(X, Wg, Ag, Bg, Sg)
    up = pytorch_lora_linear(X, Wu, Au, Bu, Su)
    h = F.silu(gate) * up
    out = pytorch_lora_linear(h, Wd, Ad, Bd, Sd)
    return out


def opaque_lora_mlp(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """Opaque kernel implementation."""
    result = NewStyleLoRAMLP.apply(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, ACTIVATION_SWIGLU)
    return result[0]


# ============================================================================
# LoRA-W Tests (single projection)
# ============================================================================

class TestLoRAWForward:
    """Test LoRA-W forward pass precision."""

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, float32)."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32)
        W = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        A = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        B = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        out_pt = pytorch_lora_linear(X, W, A, B, SCALING)
        out_op = opaque_lora_linear(X, W, A, B, SCALING)

        err = precision_error(out_op, out_pt, threshold=1e-4)
        print(f"\nLoRA-W Forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")

        assert err["rel_err"] < RTOL_LORA, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_LORA:.0e}"


class TestLoRAWBackward:
    """Test LoRA-W backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch (non-vmap, test X.grad, A.grad, B.grad)."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)
        W = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        A_pt = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        B_pt = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        out_pt = pytorch_lora_linear(X_pt, W, A_pt, B_pt, SCALING)
        out_pt.sum().backward()

        X_op = X_pt.detach().clone().requires_grad_(True)
        A_op = A_pt.detach().clone().requires_grad_(True)
        B_op = B_pt.detach().clone().requires_grad_(True)

        out_op = opaque_lora_linear(X_op, W, A_op, B_op, SCALING)
        out_op.sum().backward()

        x_err = precision_error(X_op.grad, X_pt.grad, threshold=1e-4)
        a_err = precision_error(A_op.grad, A_pt.grad, threshold=1e-4)
        b_err = precision_error(B_op.grad, B_pt.grad, threshold=1e-4)

        print(f"\nLoRA-W Backward:")
        print(f"  X.grad: abs={x_err['abs_err']:.2e}, rel={x_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  A.grad: abs={a_err['abs_err']:.2e}, rel={a_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  B.grad: abs={b_err['abs_err']:.2e}, rel={b_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")

        assert x_err["rel_err"] < RTOL_LORA_BACKWARD, f"X.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert a_err["rel_err"] < RTOL_LORA_BACKWARD, f"A.grad rel_err {a_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert b_err["rel_err"] < RTOL_LORA_BACKWARD, f"B.grad rel_err {b_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"


class TestLoRAWVmap:
    """Test LoRA-W vmap precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap (forward + backward)."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)

        W = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        A = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        B = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        X_pt = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)
        X_op = X_pt.detach().clone().requires_grad_(True)

        out_pt = torch.vmap(lambda x: pytorch_lora_linear(x, W, A, B, SCALING))(X_pt)
        out_pt.sum().backward()

        out_op = torch.vmap(lambda x: opaque_lora_linear(x, W, A, B, SCALING))(X_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        bwd_err = precision_error(X_op.grad, X_pt.grad, threshold=1e-4)

        print(f"\nLoRA-W vmap:")
        print(f"  forward: abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")
        print(f"  X.grad:  abs={bwd_err['abs_err']:.2e}, rel={bwd_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_LORA, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"
        assert bwd_err["rel_err"] < RTOL_LORA_BACKWARD, f"vmap X.grad rel_err {bwd_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)

        W = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        A = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        B = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(x):
            return torch.vmap(lambda xi: pytorch_lora_linear(xi, W, A, B, SCALING))(x)

        def opaque_fn(x):
            return torch.vmap(lambda xi: opaque_lora_linear(xi, W, A, B, SCALING))(x)

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-W vmap", max_perf_overhead=0.20)


class TestLoRAWPerformance:
    """Test LoRA-W kernel performance (non-vmap)."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward performance: opaque vs pytorch."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32)
        W = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        A = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        B = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        def pytorch_fn(x):
            return pytorch_lora_linear(x, W, A, B, SCALING)

        def opaque_fn(x):
            return opaque_lora_linear(x, W, A, B, SCALING)

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-W forward", max_perf_overhead=0.50)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Backward performance: opaque vs pytorch."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)
        W = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        A = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        B = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

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

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, check Q, K, V each)."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32)

        Wq = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Aq = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bq = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wk = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Ak = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bk = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wv = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Av = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bv = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Q_pt, K_pt, V_pt = pytorch_lora_qkv(
            X, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
        )
        Q_op, K_op, V_op = opaque_lora_qkv(
            X, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING
        )

        q_err = precision_error(Q_op, Q_pt, threshold=1e-4)
        k_err = precision_error(K_op, K_pt, threshold=1e-4)
        v_err = precision_error(V_op, V_pt, threshold=1e-4)

        print(f"\nLoRA-QKV Forward:")
        print(f"  Q: abs={q_err['abs_err']:.2e}, rel={q_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")
        print(f"  K: abs={k_err['abs_err']:.2e}, rel={k_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")
        print(f"  V: abs={v_err['abs_err']:.2e}, rel={v_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")

        assert q_err["rel_err"] < RTOL_LORA, f"Q rel_err {q_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"
        assert k_err["rel_err"] < RTOL_LORA, f"K rel_err {k_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"
        assert v_err["rel_err"] < RTOL_LORA, f"V rel_err {v_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"


class TestLoRAQKVBackward:
    """Test LoRA-QKV backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch (non-vmap)."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wq = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Aq_pt = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bq_pt = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wk = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Ak_pt = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bk_pt = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wv = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Av_pt = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bv_pt = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Q_pt, K_pt, V_pt = pytorch_lora_qkv(
            X_pt, Wq, Aq_pt, Bq_pt, SCALING, Wk, Ak_pt, Bk_pt, SCALING, Wv, Av_pt, Bv_pt, SCALING
        )
        (Q_pt + K_pt + V_pt).sum().backward()

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
        (Q_op + K_op + V_op).sum().backward()

        x_err = precision_error(X_op.grad, X_pt.grad, threshold=1e-4)
        aq_err = precision_error(Aq_op.grad, Aq_pt.grad, threshold=1e-4)
        bq_err = precision_error(Bq_op.grad, Bq_pt.grad, threshold=1e-4)
        ak_err = precision_error(Ak_op.grad, Ak_pt.grad, threshold=1e-4)
        bk_err = precision_error(Bk_op.grad, Bk_pt.grad, threshold=1e-4)
        av_err = precision_error(Av_op.grad, Av_pt.grad, threshold=1e-4)
        bv_err = precision_error(Bv_op.grad, Bv_pt.grad, threshold=1e-4)

        print(f"\nLoRA-QKV Backward:")
        print(f"  X.grad:  abs={x_err['abs_err']:.2e}, rel={x_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  Aq.grad: abs={aq_err['abs_err']:.2e}, rel={aq_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  Bq.grad: abs={bq_err['abs_err']:.2e}, rel={bq_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  Ak.grad: abs={ak_err['abs_err']:.2e}, rel={ak_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  Bk.grad: abs={bk_err['abs_err']:.2e}, rel={bk_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  Av.grad: abs={av_err['abs_err']:.2e}, rel={av_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")
        print(f"  Bv.grad: abs={bv_err['abs_err']:.2e}, rel={bv_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")

        assert x_err["rel_err"] < RTOL_LORA_BACKWARD, f"X.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert aq_err["rel_err"] < RTOL_LORA_BACKWARD, f"Aq.grad rel_err {aq_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert bq_err["rel_err"] < RTOL_LORA_BACKWARD, f"Bq.grad rel_err {bq_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert ak_err["rel_err"] < RTOL_LORA_BACKWARD, f"Ak.grad rel_err {ak_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert bk_err["rel_err"] < RTOL_LORA_BACKWARD, f"Bk.grad rel_err {bk_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert av_err["rel_err"] < RTOL_LORA_BACKWARD, f"Av.grad rel_err {av_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"
        assert bv_err["rel_err"] < RTOL_LORA_BACKWARD, f"Bv.grad rel_err {bv_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"


class TestLoRAQKVVmap:
    """Test LoRA-QKV vmap precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap (forward + backward)."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)

        Wq = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Aq = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bq = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wk = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Ak = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bk = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wv = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Av = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bv = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        X_pt = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)
        X_op = X_pt.detach().clone().requires_grad_(True)

        # PyTorch vmap
        Q_pt, K_pt, V_pt = torch.vmap(
            lambda x: pytorch_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
        )(X_pt)
        (Q_pt + K_pt + V_pt).sum().backward()

        # Opaque vmap
        Q_op, K_op, V_op = torch.vmap(
            lambda x: opaque_lora_qkv(x, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
        )(X_op)
        (Q_op + K_op + V_op).sum().backward()

        # Check forward precision for each output
        q_err = precision_error(Q_op, Q_pt, threshold=1e-4)
        k_err = precision_error(K_op, K_pt, threshold=1e-4)
        v_err = precision_error(V_op, V_pt, threshold=1e-4)
        bwd_err = precision_error(X_op.grad, X_pt.grad, threshold=1e-4)

        print(f"\nLoRA-QKV vmap:")
        print(f"  Q:      abs={q_err['abs_err']:.2e}, rel={q_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")
        print(f"  K:      abs={k_err['abs_err']:.2e}, rel={k_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")
        print(f"  V:      abs={v_err['abs_err']:.2e}, rel={v_err['rel_err']:.2e} (target: <{RTOL_LORA:.0e})")
        print(f"  X.grad: abs={bwd_err['abs_err']:.2e}, rel={bwd_err['rel_err']:.2e} (target: <{RTOL_LORA_BACKWARD:.0e})")

        assert q_err["rel_err"] < RTOL_LORA, f"vmap Q rel_err {q_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"
        assert k_err["rel_err"] < RTOL_LORA, f"vmap K rel_err {k_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"
        assert v_err["rel_err"] < RTOL_LORA, f"vmap V rel_err {v_err['rel_err']:.2e} >= {RTOL_LORA:.0e}"
        assert bwd_err["rel_err"] < RTOL_LORA_BACKWARD, f"vmap X.grad rel_err {bwd_err['rel_err']:.2e} >= {RTOL_LORA_BACKWARD:.0e}"

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)

        Wq = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Aq = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bq = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wk = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Ak = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bk = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wv = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Av = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bv = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(x):
            Q, K, V = torch.vmap(
                lambda xi: pytorch_lora_qkv(xi, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
            )(x)
            return Q + K + V

        def opaque_fn(x):
            Q, K, V = torch.vmap(
                lambda xi: opaque_lora_qkv(xi, Wq, Aq, Bq, SCALING, Wk, Ak, Bk, SCALING, Wv, Av, Bv, SCALING)
            )(x)
            return Q + K + V

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-QKV vmap", max_perf_overhead=0.20)


class TestLoRAQKVPerformance:
    """Test LoRA-QKV kernel performance (non-vmap)."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward performance: opaque vs pytorch."""
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32)

        Wq = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Aq = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bq = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wk = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Ak = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bk = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        Wv = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Av = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bv = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

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
        HIDDEN = mellum_config["hidden_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wq = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Aq = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bq = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wk = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Ak = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bk = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wv = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.float32)
        Av = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bv = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

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

    def test_forward_matches_pytorch(self, precision_error, mellum_config):
        """Forward: opaque vs pytorch (non-vmap, float32)."""
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32)

        Wg = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Ag = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bg = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wu = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Au = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bu = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wd = torch.randn(HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32)
        Ad = torch.randn(INTERMEDIATE, RANK, device="cuda", dtype=torch.float32)
        Bd = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        out_pt = pytorch_lora_mlp(
            X, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )
        out_op = opaque_lora_mlp(
            X, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
        )

        err = precision_error(out_op, out_pt, threshold=1e-4)
        print(f"\nLoRA-MLP Forward: abs={err['abs_err']:.2e}, rel={err['rel_err']:.2e} (target: <{RTOL_LORA_MLP:.0e})")

        assert err["rel_err"] < RTOL_LORA_MLP, f"Forward rel_err {err['rel_err']:.2e} >= {RTOL_LORA_MLP:.0e}"


class TestLoRAMLPBackward:
    """Test LoRA-MLP backward pass precision."""

    def test_backward_matches_pytorch(self, precision_error, mellum_config):
        """Backward: opaque vs pytorch (non-vmap)."""
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X_pt = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wg = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Ag_pt = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bg_pt = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32, requires_grad=True)

        Wu = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Au_pt = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bu_pt = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32, requires_grad=True)

        Wd = torch.randn(HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32)
        Ad_pt = torch.randn(INTERMEDIATE, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bd_pt = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        out_pt = pytorch_lora_mlp(
            X_pt, Wg, Ag_pt, Bg_pt, SCALING, Wu, Au_pt, Bu_pt, SCALING, Wd, Ad_pt, Bd_pt, SCALING
        )
        out_pt.sum().backward()

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
        out_op.sum().backward()

        x_err = precision_error(X_op.grad, X_pt.grad, threshold=1e-4)
        ag_err = precision_error(Ag_op.grad, Ag_pt.grad, threshold=1e-4)
        bg_err = precision_error(Bg_op.grad, Bg_pt.grad, threshold=1e-4)
        au_err = precision_error(Au_op.grad, Au_pt.grad, threshold=1e-4)
        bu_err = precision_error(Bu_op.grad, Bu_pt.grad, threshold=1e-4)
        ad_err = precision_error(Ad_op.grad, Ad_pt.grad, threshold=1e-4)
        bd_err = precision_error(Bd_op.grad, Bd_pt.grad, threshold=1e-4)

        print(f"\nLoRA-MLP Backward:")
        print(f"  X.grad:  abs={x_err['abs_err']:.2e}, rel={x_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")
        print(f"  Ag.grad: abs={ag_err['abs_err']:.2e}, rel={ag_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")
        print(f"  Bg.grad: abs={bg_err['abs_err']:.2e}, rel={bg_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")
        print(f"  Au.grad: abs={au_err['abs_err']:.2e}, rel={au_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")
        print(f"  Bu.grad: abs={bu_err['abs_err']:.2e}, rel={bu_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")
        print(f"  Ad.grad: abs={ad_err['abs_err']:.2e}, rel={ad_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")
        print(f"  Bd.grad: abs={bd_err['abs_err']:.2e}, rel={bd_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")

        assert x_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"X.grad rel_err {x_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"
        assert ag_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"Ag.grad rel_err {ag_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"
        assert bg_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"Bg.grad rel_err {bg_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"
        assert au_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"Au.grad rel_err {au_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"
        assert bu_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"Bu.grad rel_err {bu_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"
        assert ad_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"Ad.grad rel_err {ad_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"
        assert bd_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"Bd.grad rel_err {bd_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"


class TestLoRAMLPVmap:
    """Test LoRA-MLP vmap precision and performance."""

    def test_vmap_precision(self, precision_error, mellum_config):
        """vmap: opaque vs pytorch vmap (forward + backward)."""
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)

        Wg = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Ag = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bg = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wu = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Au = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bu = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wd = torch.randn(HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32)
        Ad = torch.randn(INTERMEDIATE, RANK, device="cuda", dtype=torch.float32)
        Bd = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        X_pt = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)
        X_op = X_pt.detach().clone().requires_grad_(True)

        # PyTorch vmap
        out_pt = torch.vmap(
            lambda x: pytorch_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)
        )(X_pt)
        out_pt.sum().backward()

        # Opaque vmap
        out_op = torch.vmap(
            lambda x: opaque_lora_mlp(x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)
        )(X_op)
        out_op.sum().backward()

        fwd_err = precision_error(out_op, out_pt, threshold=1e-4)
        bwd_err = precision_error(X_op.grad, X_pt.grad, threshold=1e-4)

        print(f"\nLoRA-MLP vmap:")
        print(f"  forward: abs={fwd_err['abs_err']:.2e}, rel={fwd_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP:.0e})")
        print(f"  X.grad:  abs={bwd_err['abs_err']:.2e}, rel={bwd_err['rel_err']:.2e} (target: <{RTOL_LORA_MLP_BACKWARD:.0e})")

        assert fwd_err["rel_err"] < RTOL_LORA_MLP, f"vmap forward rel_err {fwd_err['rel_err']:.2e} >= {RTOL_LORA_MLP:.0e}"
        assert bwd_err["rel_err"] < RTOL_LORA_MLP_BACKWARD, f"vmap X.grad rel_err {bwd_err['rel_err']:.2e} >= {RTOL_LORA_MLP_BACKWARD:.0e}"

    def test_vmap_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """vmap: opaque should be faster or use less memory."""
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        VMAP_BATCH = mellum_config["vmap_batch"]
        torch.manual_seed(42)

        Wg = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Ag = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bg = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wu = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Au = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bu = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wd = torch.randn(HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32)
        Ad = torch.randn(INTERMEDIATE, RANK, device="cuda", dtype=torch.float32)
        Bd = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        X = torch.randn(VMAP_BATCH, BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(x):
            return torch.vmap(
                lambda xi: pytorch_lora_mlp(xi, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)
            )(x)

        def opaque_fn(x):
            return torch.vmap(
                lambda xi: opaque_lora_mlp(xi, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING)
            )(x)

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-MLP vmap", max_perf_overhead=0.20)


class TestLoRAMLPPerformance:
    """Test LoRA-MLP kernel performance (non-vmap)."""

    def test_forward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Forward performance: opaque vs pytorch."""
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32)

        Wg = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Ag = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bg = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wu = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Au = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32)
        Bu = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32)

        Wd = torch.randn(HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32)
        Ad = torch.randn(INTERMEDIATE, RANK, device="cuda", dtype=torch.float32)
        Bd = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32)

        def pytorch_fn(x):
            return pytorch_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            )

        def opaque_fn(x):
            return opaque_lora_mlp(
                x, Wg, Ag, Bg, SCALING, Wu, Au, Bu, SCALING, Wd, Ad, Bd, SCALING
            )

        pt_stats = measure_time_and_memory(pytorch_fn, X)
        op_stats = measure_time_and_memory(opaque_fn, X)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-MLP forward", max_perf_overhead=0.20)

    def test_backward_performance(self, measure_time_and_memory, assert_perf_benefit, mellum_config):
        """Backward performance: opaque vs pytorch."""
        HIDDEN = mellum_config["hidden_dim"]
        INTERMEDIATE = mellum_config["intermediate_dim"]
        RANK = mellum_config["rank"]
        torch.manual_seed(42)

        X = torch.randn(BATCH, SEQ, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        Wg = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Ag = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bg = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32, requires_grad=True)

        Wu = torch.randn(INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32)
        Au = torch.randn(HIDDEN, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bu = torch.randn(RANK, INTERMEDIATE, device="cuda", dtype=torch.float32, requires_grad=True)

        Wd = torch.randn(HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32)
        Ad = torch.randn(INTERMEDIATE, RANK, device="cuda", dtype=torch.float32, requires_grad=True)
        Bd = torch.randn(RANK, HIDDEN, device="cuda", dtype=torch.float32, requires_grad=True)

        def pytorch_fn(x, ag, bg, au, bu, ad, bd):
            return pytorch_lora_mlp(
                x, Wg, ag, bg, SCALING, Wu, au, bu, SCALING, Wd, ad, bd, SCALING
            )

        def opaque_fn(x, ag, bg, au, bu, ad, bd):
            return opaque_lora_mlp(
                x, Wg, ag, bg, SCALING, Wu, au, bu, SCALING, Wd, ad, bd, SCALING
            )

        pt_stats = measure_time_and_memory(pytorch_fn, X, Ag, Bg, Au, Bu, Ad, Bd)
        op_stats = measure_time_and_memory(opaque_fn, X, Ag, Bg, Au, Bu, Ad, Bd)

        assert_perf_benefit(pt_stats, op_stats, label="LoRA-MLP backward", max_perf_overhead=0.20)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
