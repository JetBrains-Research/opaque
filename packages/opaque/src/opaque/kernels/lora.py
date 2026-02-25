"""LoRA (Low-Rank Adaptation) kernels with vmap support for DP-SGD.

Ported from unsloth/kernels/fast_lora.py for vmap compatibility.
Uses new-style autograd API with setup_context for vmap support.

Three implementations:
1. LoRA_W: Generic LoRA for single projection (O-proj, etc.)
2. LoRA_QKV: Fused LoRA for Q, K, V projections
3. LoRA_MLP: Fused LoRA for MLP (gate, up, down) with SwiGLU

Optimizations (from Unsloth):
- Triton SwiGLU kernels for activation in LoRA_MLP (forward + backward)
- addmm_ for fused gradient accumulation (avoids temporary tensors)

For DP-SGD:
- Base weights (W) are frozen, only LoRA weights (A, B) are trained
- vmap computes per-example gradients for A and B
"""

import torch
import torch.nn.functional as F

from .swiglu import triton_swiglu_forward, triton_swiglu_backward
from .geglu import (
    triton_geglu_exact_forward,
    triton_geglu_exact_backward,
    triton_geglu_approx_forward,
    triton_geglu_approx_backward,
)

# Activation types for LoRA_MLP
ACTIVATION_SWIGLU = 0
ACTIVATION_GEGLU_EXACT = 1
ACTIVATION_GEGLU_APPROX = 2

_ACTIVATION_FORWARD = {
    ACTIVATION_SWIGLU: triton_swiglu_forward,
    ACTIVATION_GEGLU_EXACT: triton_geglu_exact_forward,
    ACTIVATION_GEGLU_APPROX: triton_geglu_approx_forward,
}

_ACTIVATION_BACKWARD = {
    ACTIVATION_SWIGLU: triton_swiglu_backward,
    ACTIVATION_GEGLU_EXACT: triton_geglu_exact_backward,
    ACTIVATION_GEGLU_APPROX: triton_geglu_approx_backward,
}

_ACTIVATION_NAMES = {
    "swiglu": ACTIVATION_SWIGLU,
    "geglu_exact": ACTIVATION_GEGLU_EXACT,
    "geglu_approx": ACTIVATION_GEGLU_APPROX,
}

# PyTorch fallbacks for vmap path (must be autograd-tracked, not raw Triton)
_ACTIVATION_FORWARD_PYTORCH = {
    ACTIVATION_SWIGLU: lambda gate, up: F.silu(gate) * up,
    ACTIVATION_GEGLU_EXACT: lambda gate, up: F.gelu(gate, approximate='none') * up,
    ACTIVATION_GEGLU_APPROX: lambda gate, up: F.gelu(gate, approximate='tanh') * up,
}


class NewStyleLoRAW(torch.autograd.Function):
    """LoRA for single weight projection with vmap support.

    Computes: output = X @ W.T + X @ A @ B * scaling

    Used for O-projection and other single linear layers.
    """

    @staticmethod
    def forward(X, W, A, B, scaling):
        """Forward pass."""
        out = F.linear(X, W)  # X @ W.T

        if A is not None and B is not None:
            lora_out = (X @ A) @ B * scaling
            out = out + lora_out

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, W, A, B, scaling = inputs
        ctx.save_for_backward(X, W, A, B)
        ctx.scaling = scaling

    @staticmethod
    def backward(ctx, grad_out):
        X, W, A, B = ctx.saved_tensors
        scaling = ctx.scaling

        batch_shape = X.shape[:-1]
        hidden_dim = X.shape[-1]
        out_features = W.shape[0]

        X_flat = X.reshape(-1, hidden_dim)
        grad_out_flat = grad_out.reshape(-1, out_features)

        # LoRA weight gradients FIRST (before any potential buffer reuse)
        dA = dB = None
        if A is not None and B is not None:
            # dA = (X.T @ grad_out @ B.T) * scaling
            # Use addmm_ to fuse scaling and avoid temporaries (Unsloth pattern)
            grad_out_Bt = grad_out_flat @ B.t()
            dA = torch.empty_like(A)
            dA.addmm_(X_flat.t(), grad_out_Bt, alpha=scaling, beta=0)

            # dB = (A.T @ X.T @ grad_out) * scaling
            At_Xt = A.t() @ X_flat.t()
            dB = torch.empty_like(B)
            dB.addmm_(At_Xt, grad_out_flat, alpha=scaling, beta=0)

        # dX = grad_out @ W + (grad_out @ B.T @ A.T) * scaling
        dX = grad_out_flat @ W
        if A is not None and B is not None:
            # Fuse LoRA contribution into dX using addmm_
            dX.addmm_(grad_out_Bt, A.t(), alpha=scaling, beta=1)
        dX = dX.reshape(*batch_shape, hidden_dim)

        return dX, None, dA, dB, None

    @staticmethod
    def vmap(info, in_dims, X, W, A, B, scaling):
        """Efficient vmap rule: merge vmap batch into regular batch.

        This is mathematically equivalent to vmap but avoids the overhead
        of custom backward rules. For linear operations, this approach
        gives identical gradients while being just as fast as non-vmapped code.
        """
        X_bdim, W_bdim, A_bdim, B_bdim, scaling_bdim = in_dims

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")
        if W_bdim is not None:
            raise ValueError("W (base weight) should not be batched")
        if A_bdim is not None or B_bdim is not None:
            raise ValueError("LoRA weights should not be batched in vmap")
        if scaling_bdim is not None:
            raise ValueError("scaling should not be batched")

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        # Apply LoRA once using native PyTorch (bypasses custom backward overhead)
        out = F.linear(X_merged, W)
        if A is not None and B is not None:
            out = out + (X_merged @ A @ B) * scaling

        out = out.reshape(*original_shape[:-1], -1)
        return out, X_bdim


class NewStyleLoRAQKV(torch.autograd.Function):
    """Fused LoRA for Q, K, V projections with vmap support.

    Computes:
        Q = X @ Wq.T + X @ Aq @ Bq * scaling_q
        K = X @ Wk.T + X @ Ak @ Bk * scaling_k
        V = X @ Wv.T + X @ Av @ Bv * scaling_v
    """

    @staticmethod
    def forward(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        """Forward pass for Q, K, V projections."""
        Q = F.linear(X, Wq)
        if Aq is not None and Bq is not None:
            Q = Q + (X @ Aq) @ Bq * Sq

        K = F.linear(X, Wk)
        if Ak is not None and Bk is not None:
            K = K + (X @ Ak) @ Bk * Sk

        V = F.linear(X, Wv)
        if Av is not None and Bv is not None:
            V = V + (X @ Av) @ Bv * Sv

        return Q, K, V

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv = inputs
        ctx.save_for_backward(X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv)
        ctx.Sq = Sq
        ctx.Sk = Sk
        ctx.Sv = Sv

    @staticmethod
    def backward(ctx, grad_Q, grad_K, grad_V):
        X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = ctx.saved_tensors
        Sq, Sk, Sv = ctx.Sq, ctx.Sk, ctx.Sv

        batch_shape = X.shape[:-1]
        hidden_dim = X.shape[-1]

        X_flat = X.reshape(-1, hidden_dim)
        grad_Q_flat = grad_Q.reshape(-1, grad_Q.shape[-1])
        grad_K_flat = grad_K.reshape(-1, grad_K.shape[-1])
        grad_V_flat = grad_V.reshape(-1, grad_V.shape[-1])

        # LoRA weight gradients FIRST (Unsloth pattern with addmm_)
        dAq = dBq = dAk = dBk = dAv = dBv = None

        # Precompute intermediates needed for both weight grads and dX
        grad_Q_Bqt = grad_K_Bkt = grad_V_Bvt = None

        if Aq is not None and Bq is not None:
            grad_Q_Bqt = grad_Q_flat @ Bq.t()
            dAq = torch.empty_like(Aq)
            dAq.addmm_(X_flat.t(), grad_Q_Bqt, alpha=Sq, beta=0)
            dBq = torch.empty_like(Bq)
            dBq.addmm_(Aq.t() @ X_flat.t(), grad_Q_flat, alpha=Sq, beta=0)

        if Ak is not None and Bk is not None:
            grad_K_Bkt = grad_K_flat @ Bk.t()
            dAk = torch.empty_like(Ak)
            dAk.addmm_(X_flat.t(), grad_K_Bkt, alpha=Sk, beta=0)
            dBk = torch.empty_like(Bk)
            dBk.addmm_(Ak.t() @ X_flat.t(), grad_K_flat, alpha=Sk, beta=0)

        if Av is not None and Bv is not None:
            grad_V_Bvt = grad_V_flat @ Bv.t()
            dAv = torch.empty_like(Av)
            dAv.addmm_(X_flat.t(), grad_V_Bvt, alpha=Sv, beta=0)
            dBv = torch.empty_like(Bv)
            dBv.addmm_(Av.t() @ X_flat.t(), grad_V_flat, alpha=Sv, beta=0)

        # dX = sum of base weight + LoRA contributions (Unsloth accumulation pattern)
        dX = grad_Q_flat @ Wq
        dX.addmm_(grad_K_flat, Wk, beta=1, alpha=1)
        dX.addmm_(grad_V_flat, Wv, beta=1, alpha=1)

        if grad_Q_Bqt is not None:
            dX.addmm_(grad_Q_Bqt, Aq.t(), alpha=Sq, beta=1)
        if grad_K_Bkt is not None:
            dX.addmm_(grad_K_Bkt, Ak.t(), alpha=Sk, beta=1)
        if grad_V_Bvt is not None:
            dX.addmm_(grad_V_Bvt, Av.t(), alpha=Sv, beta=1)

        dX = dX.reshape(*batch_shape, hidden_dim)

        return (
            dX,
            None, dAq, dBq, None,  # Q: Wq, Aq, Bq, Sq
            None, dAk, dBk, None,  # K: Wk, Ak, Bk, Sk
            None, dAv, dBv, None,  # V: Wv, Av, Bv, Sv
        )

    @staticmethod
    def vmap(info, in_dims, X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        """Efficient vmap rule: merge vmap batch into regular batch."""
        X_bdim = in_dims[0]

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")

        for i, bdim in enumerate(in_dims[1:], 1):
            if bdim is not None:
                raise ValueError(f"Input {i} should not be batched")

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        # Apply LoRA using native PyTorch (bypasses custom backward overhead)
        Q = F.linear(X_merged, Wq)
        if Aq is not None and Bq is not None:
            Q = Q + (X_merged @ Aq @ Bq) * Sq

        K = F.linear(X_merged, Wk)
        if Ak is not None and Bk is not None:
            K = K + (X_merged @ Ak @ Bk) * Sk

        V = F.linear(X_merged, Wv)
        if Av is not None and Bv is not None:
            V = V + (X_merged @ Av @ Bv) * Sv

        # Reshape back
        Q = Q.reshape(*original_shape[:-1], -1)
        K = K.reshape(*original_shape[:-1], -1)
        V = V.reshape(*original_shape[:-1], -1)

        return (Q, K, V), (X_bdim, X_bdim, X_bdim)


class NewStyleLoRAMLP(torch.autograd.Function):
    """Fused LoRA for MLP (gate, up, down) with configurable GLU activation.

    Uses Triton kernels for the activation (matching Unsloth's callback pattern).
    Supports SwiGLU (default), GeGLU exact, and GeGLU approx via activation_type.

    Computes:
        gate = X @ Wg.T + X @ Ag @ Bg * Sg
        up = X @ Wu.T + X @ Au @ Bu * Su
        h = activation(gate) * up  # via Triton kernel
        out = h @ Wd.T + h @ Ad @ Bd * Sd
    """

    @staticmethod
    def forward(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type):
        """Forward pass for MLP with configurable GLU activation."""
        gate = F.linear(X, Wg)
        if Ag is not None and Bg is not None:
            gate = gate + (X @ Ag) @ Bg * Sg

        up = F.linear(X, Wu)
        if Au is not None and Bu is not None:
            up = up + (X @ Au) @ Bu * Su

        # Use Triton kernel for activation (Unsloth callback pattern)
        act_forward = _ACTIVATION_FORWARD[activation_type]
        h = act_forward(gate, up)

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            out = out + (h @ Ad) @ Bd * Sd

        return out, gate, up, h

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type = inputs
        out, gate, up, h = output
        ctx.save_for_backward(X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up, h)
        ctx.Sg = Sg
        ctx.Su = Su
        ctx.Sd = Sd
        ctx.activation_type = activation_type

    @staticmethod
    def backward(ctx, grad_out, grad_gate, grad_up, grad_h):
        X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up, h = ctx.saved_tensors
        Sg, Su, Sd = ctx.Sg, ctx.Su, ctx.Sd

        batch_shape = X.shape[:-1]
        hidden_dim = X.shape[-1]

        X_flat = X.reshape(-1, hidden_dim)
        grad_out_flat = grad_out.reshape(-1, grad_out.shape[-1])
        gate_flat = gate.reshape(-1, gate.shape[-1])
        up_flat = up.reshape(-1, up.shape[-1])
        h_flat = h.reshape(-1, h.shape[-1])

        # Backward through down projection
        dh = grad_out_flat @ Wd
        if Ad is not None and Bd is not None:
            dh.addmm_(grad_out_flat @ Bd.t(), Ad.t(), alpha=Sd, beta=1)

        # Backward through activation using Triton kernel (Unsloth callback pattern)
        act_backward = _ACTIVATION_BACKWARD[ctx.activation_type]
        dgate, dup = act_backward(dh, gate_flat, up_flat)

        # Backward through gate projection
        dX_gate = dgate @ Wg
        if Ag is not None and Bg is not None:
            dX_gate.addmm_(dgate @ Bg.t(), Ag.t(), alpha=Sg, beta=1)

        # Backward through up projection
        dX_up = dup @ Wu
        if Au is not None and Bu is not None:
            dX_up.addmm_(dup @ Bu.t(), Au.t(), alpha=Su, beta=1)

        dX = (dX_gate + dX_up).reshape(*batch_shape, hidden_dim)

        # LoRA weight gradients (Unsloth addmm_ pattern)
        dAg = dBg = dAu = dBu = dAd = dBd = None

        if Ag is not None and Bg is not None:
            dgate_Bgt = dgate @ Bg.t()
            dAg = torch.empty_like(Ag)
            dAg.addmm_(X_flat.t(), dgate_Bgt, alpha=Sg, beta=0)
            dBg = torch.empty_like(Bg)
            dBg.addmm_(Ag.t() @ X_flat.t(), dgate, alpha=Sg, beta=0)

        if Au is not None and Bu is not None:
            dup_But = dup @ Bu.t()
            dAu = torch.empty_like(Au)
            dAu.addmm_(X_flat.t(), dup_But, alpha=Su, beta=0)
            dBu = torch.empty_like(Bu)
            dBu.addmm_(Au.t() @ X_flat.t(), dup, alpha=Su, beta=0)

        if Ad is not None and Bd is not None:
            grad_out_Bdt = grad_out_flat @ Bd.t()
            dAd = torch.empty_like(Ad)
            dAd.addmm_(h_flat.t(), grad_out_Bdt, alpha=Sd, beta=0)
            dBd = torch.empty_like(Bd)
            dBd.addmm_(Ad.t() @ h_flat.t(), grad_out_flat, alpha=Sd, beta=0)

        return (
            dX,
            None, dAg, dBg, None,  # gate
            None, dAu, dBu, None,  # up
            None, dAd, dBd, None,  # down
            None,  # activation_type
        )

    @staticmethod
    def vmap(info, in_dims, X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type):
        """Efficient vmap rule: merge vmap batch into regular batch."""
        X_bdim = in_dims[0]

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")

        # in_dims has 14 elements (13 original + activation_type)
        for i, bdim in enumerate(in_dims[1:13], 1):
            if bdim is not None:
                raise ValueError(f"Input {i} should not be batched")

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        # Apply MLP using native PyTorch ops (correct for per-example grads).
        # Must use PyTorch activation (not raw Triton) to preserve autograd graph.
        gate = F.linear(X_merged, Wg)
        if Ag is not None and Bg is not None:
            gate = gate + (X_merged @ Ag @ Bg) * Sg

        up = F.linear(X_merged, Wu)
        if Au is not None and Bu is not None:
            up = up + (X_merged @ Au @ Bu) * Su

        # PyTorch activation (autograd-tracked for gradient computation)
        act_forward = _ACTIVATION_FORWARD_PYTORCH[activation_type]
        h = act_forward(gate, up)

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            out = out + (h @ Ad @ Bd) * Sd

        # Reshape back
        out = out.reshape(*original_shape[:-1], -1)
        gate = gate.reshape(*original_shape[:-1], -1)
        up = up.reshape(*original_shape[:-1], -1)
        h = h.reshape(*original_shape[:-1], -1)

        return (out, gate, up, h), (X_bdim, X_bdim, X_bdim, X_bdim)


# ============================================================================
# Convenience wrappers
# ============================================================================

def lora_linear_vmap(X, W, A, B, scaling):
    """Apply LoRA linear projection with vmap support.

    Args:
        X: Input tensor (batch, seq_len, hidden_dim)
        W: Base weight (out_features, in_features), frozen
        A: LoRA A weight (in_features, rank)
        B: LoRA B weight (rank, out_features)
        scaling: LoRA scaling (lora_alpha / lora_r)

    Returns:
        Output tensor (batch, seq_len, out_features)
    """
    return NewStyleLoRAW.apply(X, W, A, B, scaling)


def lora_qkv_vmap(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """Apply LoRA to Q, K, V projections with vmap support.

    Returns:
        Tuple of (Q, K, V)
    """
    return NewStyleLoRAQKV.apply(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)


def lora_mlp_vmap(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation="swiglu"):
    """Apply LoRA MLP with configurable GLU activation and vmap support.

    Args:
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd: Standard LoRA MLP inputs.
        activation: Activation type - "swiglu" (default), "geglu_exact", or "geglu_approx".

    Returns:
        Output tensor (batch, seq_len, hidden_dim)
    """
    if isinstance(activation, str):
        activation_type = _ACTIVATION_NAMES[activation]
    else:
        activation_type = activation
    result = NewStyleLoRAMLP.apply(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type)
    return result[0]  # Return only the output, not intermediate tensors
