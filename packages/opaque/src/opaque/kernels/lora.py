"""LoRA (Low-Rank Adaptation) kernels with vmap support for DP-SGD.

Simplified from unsloth/kernels/fast_lora.py for vmap compatibility.
Uses new-style autograd API with setup_context for vmap support.

Three implementations:
1. LoRA_W: Generic LoRA for single projection (O-proj, etc.)
2. LoRA_QKV: Fused LoRA for Q, K, V projections
3. LoRA_MLP: Fused LoRA for MLP (gate, up, down) with SwiGLU

For DP-SGD:
- Base weights (W) are frozen, only LoRA weights (A, B) are trained
- vmap computes per-example gradients for A and B
"""

import torch
import torch.nn.functional as F


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
            # A is (in_dim, rank), B is (rank, out_features)
            # X @ A gives (batch, seq, rank)
            # (X @ A) @ B gives (batch, seq, out_features)
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

        # dX = grad_out @ W + grad_out @ B.T @ A.T * scaling
        dX = grad_out_flat @ W
        if A is not None and B is not None:
            # Forward: out = (X @ A) @ B
            # Backward: dX += grad_out @ B.T @ A.T
            dX = dX + (grad_out_flat @ B.t() @ A.t()) * scaling
        dX = dX.reshape(*batch_shape, hidden_dim)

        # LoRA weight gradients
        dA = dB = None
        if A is not None and B is not None:
            # Forward: out = (X @ A) @ B
            # dA = X.T @ grad_out @ B.T
            # dB = A.T @ X.T @ grad_out
            dA = (X_flat.t() @ grad_out_flat @ B.t()) * scaling
            dB = (A.t() @ X_flat.t() @ grad_out_flat) * scaling

        return dX, None, dA, dB, None

    @staticmethod
    def vmap(info, in_dims, X, W, A, B, scaling):
        """Custom vmap rule for DP-SGD per-example gradients."""
        X_bdim, W_bdim, A_bdim, B_bdim, scaling_bdim = in_dims

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")
        if W_bdim is not None:
            raise ValueError("W (base weight) should not be batched")
        if A_bdim is not None or B_bdim is not None:
            raise ValueError("LoRA weights should not be batched in vmap")
        if scaling_bdim is not None:
            raise ValueError("scaling should not be batched")

        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        out = NewStyleLoRAW.apply(X_merged, W, A, B, scaling)
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

        # dX = sum of gradients from Q, K, V
        dX = grad_Q_flat @ Wq + grad_K_flat @ Wk + grad_V_flat @ Wv

        if Aq is not None and Bq is not None:
            # Forward: Q = ... + (X @ Aq) @ Bq
            # Backward: dX += grad_Q @ Bq.T @ Aq.T
            dX = dX + (grad_Q_flat @ Bq.t() @ Aq.t()) * Sq
        if Ak is not None and Bk is not None:
            dX = dX + (grad_K_flat @ Bk.t() @ Ak.t()) * Sk
        if Av is not None and Bv is not None:
            dX = dX + (grad_V_flat @ Bv.t() @ Av.t()) * Sv

        dX = dX.reshape(*batch_shape, hidden_dim)

        # LoRA weight gradients
        dAq = dBq = dAk = dBk = dAv = dBv = None

        if Aq is not None and Bq is not None:
            # Forward: Q = ... + (X @ Aq) @ Bq
            # dAq = X.T @ grad_Q @ Bq.T, dBq = Aq.T @ X.T @ grad_Q
            dAq = (X_flat.t() @ grad_Q_flat @ Bq.t()) * Sq
            dBq = (Aq.t() @ X_flat.t() @ grad_Q_flat) * Sq

        if Ak is not None and Bk is not None:
            dAk = (X_flat.t() @ grad_K_flat @ Bk.t()) * Sk
            dBk = (Ak.t() @ X_flat.t() @ grad_K_flat) * Sk

        if Av is not None and Bv is not None:
            dAv = (X_flat.t() @ grad_V_flat @ Bv.t()) * Sv
            dBv = (Av.t() @ X_flat.t() @ grad_V_flat) * Sv

        return (
            dX,
            None, dAq, dBq, None,  # Q: Wq, Aq, Bq, Sq
            None, dAk, dBk, None,  # K: Wk, Ak, Bk, Sk
            None, dAv, dBv, None,  # V: Wv, Av, Bv, Sv
        )

    @staticmethod
    def vmap(info, in_dims, X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        """Custom vmap rule for DP-SGD per-example gradients."""
        X_bdim = in_dims[0]

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")

        for i, bdim in enumerate(in_dims[1:], 1):
            if bdim is not None:
                raise ValueError(f"Input {i} should not be batched")

        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        Q, K, V = NewStyleLoRAQKV.apply(
            X_merged, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv
        )

        Q = Q.reshape(*original_shape[:-1], -1)
        K = K.reshape(*original_shape[:-1], -1)
        V = V.reshape(*original_shape[:-1], -1)

        return (Q, K, V), (X_bdim, X_bdim, X_bdim)


class NewStyleLoRAMLP(torch.autograd.Function):
    """Fused LoRA for MLP (gate, up, down) with SwiGLU activation.

    Computes:
        gate = X @ Wg.T + X @ Ag @ Bg * Sg
        up = X @ Wu.T + X @ Au @ Bu * Su
        h = silu(gate) * up  # SwiGLU
        out = h @ Wd.T + h @ Ad @ Bd * Sd
    """

    @staticmethod
    def forward(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
        """Forward pass for MLP with SwiGLU."""
        gate = F.linear(X, Wg)
        if Ag is not None and Bg is not None:
            gate = gate + (X @ Ag) @ Bg * Sg

        up = F.linear(X, Wu)
        if Au is not None and Bu is not None:
            up = up + (X @ Au) @ Bu * Su

        h = F.silu(gate) * up

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            out = out + (h @ Ad) @ Bd * Sd

        return out, gate, up, h

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd = inputs
        out, gate, up, h = output
        ctx.save_for_backward(X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up, h)
        ctx.Sg = Sg
        ctx.Su = Su
        ctx.Sd = Sd

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
            # Forward: out = ... + (h @ Ad) @ Bd
            # Backward: dh += grad_out @ Bd.T @ Ad.T
            dh = dh + (grad_out_flat @ Bd.t() @ Ad.t()) * Sd

        # Backward through SwiGLU
        sigmoid_gate = torch.sigmoid(gate_flat)
        silu_gate = gate_flat * sigmoid_gate

        dgate = dh * up_flat * sigmoid_gate * (1 + gate_flat * (1 - sigmoid_gate))
        dup = dh * silu_gate

        # Backward through gate projection
        dX_gate = dgate @ Wg
        if Ag is not None and Bg is not None:
            # Forward: gate = ... + (X @ Ag) @ Bg
            # Backward: dX += dgate @ Bg.T @ Ag.T
            dX_gate = dX_gate + (dgate @ Bg.t() @ Ag.t()) * Sg

        # Backward through up projection
        dX_up = dup @ Wu
        if Au is not None and Bu is not None:
            dX_up = dX_up + (dup @ Bu.t() @ Au.t()) * Su

        dX = (dX_gate + dX_up).reshape(*batch_shape, hidden_dim)

        # LoRA weight gradients
        dAg = dBg = dAu = dBu = dAd = dBd = None

        if Ag is not None and Bg is not None:
            # Forward: gate = ... + (X @ Ag) @ Bg
            # dAg = X.T @ dgate @ Bg.T, dBg = Ag.T @ X.T @ dgate
            dAg = (X_flat.t() @ dgate @ Bg.t()) * Sg
            dBg = (Ag.t() @ X_flat.t() @ dgate) * Sg

        if Au is not None and Bu is not None:
            dAu = (X_flat.t() @ dup @ Bu.t()) * Su
            dBu = (Au.t() @ X_flat.t() @ dup) * Su

        if Ad is not None and Bd is not None:
            dAd = (h_flat.t() @ grad_out_flat @ Bd.t()) * Sd
            dBd = (Ad.t() @ h_flat.t() @ grad_out_flat) * Sd

        return (
            dX,
            None, dAg, dBg, None,  # gate
            None, dAu, dBu, None,  # up
            None, dAd, dBd, None,  # down
        )

    @staticmethod
    def vmap(info, in_dims, X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
        """Custom vmap rule for DP-SGD per-example gradients."""
        X_bdim = in_dims[0]

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")

        for i, bdim in enumerate(in_dims[1:], 1):
            if bdim is not None:
                raise ValueError(f"Input {i} should not be batched")

        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        out, gate, up, h = NewStyleLoRAMLP.apply(
            X_merged, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd
        )

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


def lora_mlp_vmap(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd):
    """Apply LoRA MLP (with SwiGLU) with vmap support.

    Returns:
        Output tensor (batch, seq_len, hidden_dim)
    """
    result = NewStyleLoRAMLP.apply(X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd)
    return result[0]  # Return only the output, not intermediate tensors
