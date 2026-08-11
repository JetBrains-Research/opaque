# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton LoRA kernels derive from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py)
# and have been adapted to Opaque's vmap-friendly new-style autograd dispatch.
# See NOTICE in the repository root.
"""LoRA (Low-Rank Adaptation) kernels with vmap support for DP-SGD.

Implements the low-rank adapter structure from Hu et al., *LoRA: Low-Rank
Adaptation of Large Language Models* (https://arxiv.org/abs/2106.09685).

Ported from unsloth/kernels/fast_lora.py for vmap compatibility.
Uses new-style autograd API with setup_context for vmap support.

Three implementations:
1. LoRA_W: Generic LoRA for single projection (O-proj, etc.)
2. LoRA_QKV: Fused LoRA for Q, K, V projections
3. LoRA_MLP: Fused LoRA for MLP (gate, up, down) with SwiGLU / GeGLU

Optimizations (from Unsloth):
- Triton SwiGLU kernels for activation in LoRA_MLP (forward + backward)
- addmm_ for fused gradient accumulation (avoids temporary tensors)

For DP-SGD:
- Base weights (W) are frozen, only LoRA weights (A, B) are trained
- vmap computes per-example gradients for A and B
"""

import torch
import torch.nn.functional as F

from ._utils import ensure_cuda_tensors, follow_autocast
from .geglu import (
    _triton_geglu_approx_backward_fused,
    _triton_geglu_approx_forward,
    _triton_geglu_exact_backward_fused,
    _triton_geglu_exact_forward,
)
from .swiglu import _triton_swiglu_backward_fused, _triton_swiglu_forward

# Activation types for LoRA_MLP
ACTIVATION_SWIGLU = 0
ACTIVATION_GEGLU_EXACT = 1
ACTIVATION_GEGLU_APPROX = 2

_ACTIVATION_FORWARD = {
    ACTIVATION_SWIGLU: _triton_swiglu_forward,
    ACTIVATION_GEGLU_EXACT: _triton_geglu_exact_forward,
    ACTIVATION_GEGLU_APPROX: _triton_geglu_approx_forward,
}

_ACTIVATION_BACKWARD_FUSED = {
    ACTIVATION_SWIGLU: _triton_swiglu_backward_fused,
    ACTIVATION_GEGLU_EXACT: _triton_geglu_exact_backward_fused,
    ACTIVATION_GEGLU_APPROX: _triton_geglu_approx_backward_fused,
}

_ACTIVATION_NAMES = {
    "swiglu": ACTIVATION_SWIGLU,
    "geglu_exact": ACTIVATION_GEGLU_EXACT,
    "geglu_approx": ACTIVATION_GEGLU_APPROX,
}


def _validate_vmap_dims(in_dims, *, name, batched_indices):
    """Reject vmap layouts that cannot preserve the fused-kernel semantics."""
    for index, batch_dim in enumerate(in_dims):
        expected = 0 if index in batched_indices else None
        if batch_dim != expected:
            raise ValueError(
                f"{name} vmap requires inputs {sorted(batched_indices)} to be batched "
                f"at dim 0 and all other inputs to be unbatched, got {in_dims}"
            )


def _lora_w_backward_impl(grad_out, X, W, A, B, scaling):
    """Shared LoRA_W backward logic used by both forward and vmap paths.

    Returns (dX, dA, dB).
    """
    batch_shape = X.shape[:-1]
    hidden_dim = X.shape[-1]
    out_features = W.shape[0]

    X_flat = X.reshape(-1, hidden_dim)
    grad_out_flat = grad_out.reshape(-1, out_features)

    # LoRA weight gradients FIRST (before any potential buffer reuse)
    dA = dB = None
    if A is not None and B is not None:
        grad_out_Bt = grad_out_flat @ B.t()
        dA = torch.empty_like(A)
        dA.addmm_(X_flat.t(), grad_out_Bt, alpha=scaling, beta=0)

        At_Xt = A.t() @ X_flat.t()
        dB = torch.empty_like(B)
        dB.addmm_(At_Xt, grad_out_flat, alpha=scaling, beta=0)

    # dX: allocate fresh buffer (X may be shared across multiple LoRA layers)
    dX_flat = torch.mm(grad_out_flat, W)
    if A is not None and B is not None:
        dX_flat.addmm_(grad_out_Bt, A.t(), alpha=scaling, beta=1)
    dX = dX_flat.reshape(*batch_shape, hidden_dim)

    return dX, dA, dB


def _lora_w_backward_lite(grad_out, W, A, B, scaling):
    """Lightweight LoRA_W backward: only computes dX (no weight grads).

    Used when LoRA weights are frozen (don't require grad), e.g. in vmap(grad())
    for DP-SGD. Avoids saving X in setup_context, reducing peak memory.
    """
    batch_shape = grad_out.shape[:-1]
    out_features = W.shape[0]
    grad_out_flat = grad_out.reshape(-1, out_features)

    dX = torch.mm(grad_out_flat, W)
    if A is not None and B is not None:
        dX.addmm_(grad_out_flat @ B.t(), A.t(), alpha=scaling, beta=1)
    return dX.reshape(*batch_shape, W.shape[1])


class _LoRAWBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support.

    When vmap(grad(fn)) runs backward, functorch intercepts .apply() and
    routes to vmap(), where tensors are regular and in-place ops are safe.
    """

    @staticmethod
    def forward(grad_out, X, W, A, B, scaling):
        return _lora_w_backward_impl(grad_out, X, W, A, B, scaling)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_W")

    @staticmethod
    def vmap(info, in_dims, grad_out, X, W, A, B, scaling):
        _validate_vmap_dims(in_dims, name="_LoRAWBackward", batched_indices={0, 1})

        B_vmap = X.shape[0]
        hidden_dim = X.shape[-1]
        out_features = W.shape[0]

        # dX: merge approach works (spatially parallel — each token independent)
        X_merged = X.reshape(-1, *X.shape[2:])
        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        _X_flat = X_merged.reshape(-1, hidden_dim)
        grad_out_flat = grad_out_merged.reshape(-1, out_features)

        dX_flat = torch.mm(grad_out_flat, W)
        if A is not None and B is not None:
            grad_out_Bt_flat = grad_out_flat @ B.t()
            dX_flat.addmm_(grad_out_Bt_flat, A.t(), alpha=scaling, beta=1)
        dX = dX_flat.reshape(X.shape)

        # dA, dB: per-sample computation using bmm (NOT merged!)
        # Merging would sum across vmap samples, giving wrong per-sample grads.
        if A is not None and B is not None:
            X_3d = X.reshape(B_vmap, -1, hidden_dim)
            grad_out_3d = grad_out.reshape(B_vmap, -1, out_features)

            # dA[i] = X[i].T @ (grad_out[i] @ B.T) * scaling
            grad_out_Bt = grad_out_3d @ B.t()  # [B_vmap, tokens, rank]
            dA = torch.bmm(X_3d.transpose(-2, -1), grad_out_Bt) * scaling

            # dB[i] = (X[i] @ A).T @ grad_out[i] * scaling
            XA = X_3d @ A  # [B_vmap, tokens, rank]
            dB = torch.bmm(XA.transpose(-2, -1), grad_out_3d) * scaling
        else:
            dA = dB = None

        return (dX, dA, dB), (
            0,
            0 if dA is not None else None,
            0 if dB is not None else None,
        )


class _LoRAWBackwardLite(torch.autograd.Function):
    """Lightweight backward: only computes dX (no weight grads, no X needed).

    Used when LoRA weights don't require grad (e.g. under vmap(grad()) where
    grad() detaches captured parameters). Avoids retaining X, reducing peak memory.
    """

    @staticmethod
    def forward(grad_out, W, A, B, scaling):
        return _lora_w_backward_lite(grad_out, W, A, B, scaling)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_W")

    @staticmethod
    def vmap(info, in_dims, grad_out, W, A, B, scaling):
        _validate_vmap_dims(in_dims, name="_LoRAWBackwardLite", batched_indices={0})
        grad_out_bdim = in_dims[0]

        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        dX = _lora_w_backward_lite(grad_out_merged, W, A, B, scaling)
        dX = dX.reshape(*grad_out.shape[:-1], W.shape[1])
        return dX, grad_out_bdim


class Opaque_LoRA_W(torch.autograd.Function):
    """LoRA for single weight projection with vmap support.

    Computes: output = X @ W.T + X @ A @ B * scaling

    Used for O-projection and other single linear layers.
    """

    @staticmethod
    def forward(X, W, A, B, scaling):
        """Forward pass."""
        out = F.linear(X, W)  # X @ W.T

        if A is not None and B is not None:
            # Out-of-place addmm: same fused BLAS call as addmm_ but doesn't
            # mutate `out`, avoiding autograd version-counter conflicts when
            # both X and A/B require grad (e.g. manual per-sample gradients).
            XA = X @ A
            out = torch.addmm(
                out.reshape(-1, out.shape[-1]),
                XA.reshape(-1, XA.shape[-1]),
                B,
                alpha=scaling,
                beta=1,
            ).reshape(out.shape)

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, W, A, B, scaling = inputs
        # Under vmap(grad()), grad() detaches captured LoRA weights (requires_grad=False).
        # Skip saving X when weight grads aren't needed — reduces peak memory.
        needs_weight_grads = any(t is not None and t.requires_grad for t in [A, B])
        if needs_weight_grads:
            ctx.save_for_backward(X, W, A, B)
        else:
            ctx.save_for_backward(W, A, B)
        ctx.needs_weight_grads = needs_weight_grads
        ctx.scaling = scaling

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_weight_grads:
            X, W, A, B = ctx.saved_tensors
            dX, dA, dB = _LoRAWBackward.apply(grad_out, X, W, A, B, ctx.scaling)
        else:
            W, A, B = ctx.saved_tensors
            dX = _LoRAWBackwardLite.apply(grad_out, W, A, B, ctx.scaling)
            dA = dB = None
        return dX, None, dA, dB, None

    @staticmethod
    def vmap(info, in_dims, X, W, A, B, scaling):
        """Efficient vmap rule: merge vmap batch into regular batch.

        This is mathematically equivalent to vmap but avoids the overhead
        of custom backward rules. For linear operations, this approach
        gives identical gradients while being just as fast as non-vmapped code.
        """
        _validate_vmap_dims(in_dims, name="Opaque_LoRA_W", batched_indices={0})
        X_bdim = in_dims[0]

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        # Apply LoRA with addmm_ (same as non-vmap forward)
        out = F.linear(X_merged, W)
        if A is not None and B is not None:
            XA = X_merged @ A
            out.reshape(-1, out.shape[-1]).addmm_(
                XA.reshape(-1, XA.shape[-1]), B, alpha=scaling, beta=1
            )

        out = out.reshape(*original_shape[:-1], -1)
        return out, X_bdim


def _lora_qkv_backward_impl(
    grad_Q, grad_K, grad_V, X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv
):
    """Shared LoRA_QKV backward logic used by both forward and vmap paths.

    Returns (dX, dAq, dBq, dAk, dBk, dAv, dBv).
    """
    batch_shape = X.shape[:-1]
    hidden_dim = X.shape[-1]

    X_flat = X.reshape(-1, hidden_dim)
    grad_Q_flat = grad_Q.reshape(-1, grad_Q.shape[-1])
    grad_K_flat = grad_K.reshape(-1, grad_K.shape[-1])
    grad_V_flat = grad_V.reshape(-1, grad_V.shape[-1])

    # LoRA weight gradients FIRST (Unsloth pattern with addmm_)
    dAq = dBq = dAk = dBk = dAv = dBv = None
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

    # dX from base + LoRA weights (all LoRA grads above already consumed X_flat)
    dX_flat = torch.mm(grad_Q_flat, Wq)
    dX_flat.addmm_(grad_K_flat, Wk, beta=1, alpha=1)
    dX_flat.addmm_(grad_V_flat, Wv, beta=1, alpha=1)

    if grad_Q_Bqt is not None:
        dX_flat.addmm_(grad_Q_Bqt, Aq.t(), alpha=Sq, beta=1)
    if grad_K_Bkt is not None:
        dX_flat.addmm_(grad_K_Bkt, Ak.t(), alpha=Sk, beta=1)
    if grad_V_Bvt is not None:
        dX_flat.addmm_(grad_V_Bvt, Av.t(), alpha=Sv, beta=1)

    dX = dX_flat.reshape(*batch_shape, hidden_dim)
    return dX, dAq, dBq, dAk, dBk, dAv, dBv


class _LoRAQKVBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(
        grad_Q, grad_K, grad_V, X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv
    ):
        return _lora_qkv_backward_impl(
            grad_Q,
            grad_K,
            grad_V,
            X,
            Wq,
            Aq,
            Bq,
            Sq,
            Wk,
            Ak,
            Bk,
            Sk,
            Wv,
            Av,
            Bv,
            Sv,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_QKV")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_Q,
        grad_K,
        grad_V,
        X,
        Wq,
        Aq,
        Bq,
        Sq,
        Wk,
        Ak,
        Bk,
        Sk,
        Wv,
        Av,
        Bv,
        Sv,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAQKVBackward", batched_indices={0, 1, 2, 3}
        )

        B_vmap = X.shape[0]
        hidden_dim = X.shape[-1]

        # dX: merge approach works (spatially parallel)
        X_merged = X.reshape(-1, *X.shape[2:])
        grad_Q_merged = grad_Q.reshape(-1, *grad_Q.shape[2:])
        grad_K_merged = grad_K.reshape(-1, *grad_K.shape[2:])
        grad_V_merged = grad_V.reshape(-1, *grad_V.shape[2:])

        _X_flat = X_merged.reshape(-1, hidden_dim)
        grad_Q_flat = grad_Q_merged.reshape(-1, grad_Q.shape[-1])
        grad_K_flat = grad_K_merged.reshape(-1, grad_K.shape[-1])
        grad_V_flat = grad_V_merged.reshape(-1, grad_V.shape[-1])

        # dX from base weights (merged is fine — spatially parallel)
        dX_flat = torch.mm(grad_Q_flat, Wq)
        dX_flat.addmm_(grad_K_flat, Wk, beta=1, alpha=1)
        dX_flat.addmm_(grad_V_flat, Wv, beta=1, alpha=1)

        # dX from LoRA contributions
        grad_Q_Bqt = grad_K_Bkt = grad_V_Bvt = None
        if Aq is not None and Bq is not None:
            grad_Q_Bqt = grad_Q_flat @ Bq.t()
            dX_flat.addmm_(grad_Q_Bqt, Aq.t(), alpha=Sq, beta=1)
        if Ak is not None and Bk is not None:
            grad_K_Bkt = grad_K_flat @ Bk.t()
            dX_flat.addmm_(grad_K_Bkt, Ak.t(), alpha=Sk, beta=1)
        if Av is not None and Bv is not None:
            grad_V_Bvt = grad_V_flat @ Bv.t()
            dX_flat.addmm_(grad_V_Bvt, Av.t(), alpha=Sv, beta=1)

        dX = dX_flat.reshape(X.shape)

        # dA, dB: per-sample computation using bmm (NOT merged!)
        X_3d = X.reshape(B_vmap, -1, hidden_dim)
        grad_Q_3d = grad_Q.reshape(B_vmap, -1, grad_Q.shape[-1])
        grad_K_3d = grad_K.reshape(B_vmap, -1, grad_K.shape[-1])
        grad_V_3d = grad_V.reshape(B_vmap, -1, grad_V.shape[-1])

        def _per_sample_lora_grads(X_3d, grad_3d, A, B, S):
            if A is None or B is None:
                return None, None
            grad_Bt = grad_3d @ B.t()  # [B_vmap, tokens, rank]
            dA = torch.bmm(X_3d.transpose(-2, -1), grad_Bt) * S
            XA = X_3d @ A  # [B_vmap, tokens, rank]
            dB = torch.bmm(XA.transpose(-2, -1), grad_3d) * S
            return dA, dB

        dAq, dBq = _per_sample_lora_grads(X_3d, grad_Q_3d, Aq, Bq, Sq)
        dAk, dBk = _per_sample_lora_grads(X_3d, grad_K_3d, Ak, Bk, Sk)
        dAv, dBv = _per_sample_lora_grads(X_3d, grad_V_3d, Av, Bv, Sv)

        def _bdim(t):
            return 0 if t is not None else None

        return (
            (dX, dAq, dBq, dAk, dBk, dAv, dBv),
            (0, _bdim(dAq), _bdim(dBq), _bdim(dAk), _bdim(dBk), _bdim(dAv), _bdim(dBv)),
        )


def _lora_qkv_backward_lite(
    grad_Q, grad_K, grad_V, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv
):
    """Lightweight QKV backward: only computes dX (no weight grads, no X needed).

    Used when LoRA weights don't require grad (e.g. under vmap(grad()) where
    grad() detaches captured parameters). Avoids retaining X, reducing peak memory.
    """
    batch_shape = grad_Q.shape[:-1]
    grad_Q_flat = grad_Q.reshape(-1, grad_Q.shape[-1])
    grad_K_flat = grad_K.reshape(-1, grad_K.shape[-1])
    grad_V_flat = grad_V.reshape(-1, grad_V.shape[-1])

    # dX from base weights
    dX = torch.mm(grad_Q_flat, Wq)
    dX.addmm_(grad_K_flat, Wk, beta=1, alpha=1)
    dX.addmm_(grad_V_flat, Wv, beta=1, alpha=1)

    # dX from LoRA contributions
    if Aq is not None and Bq is not None:
        dX.addmm_(grad_Q_flat @ Bq.t(), Aq.t(), alpha=Sq, beta=1)
    if Ak is not None and Bk is not None:
        dX.addmm_(grad_K_flat @ Bk.t(), Ak.t(), alpha=Sk, beta=1)
    if Av is not None and Bv is not None:
        dX.addmm_(grad_V_flat @ Bv.t(), Av.t(), alpha=Sv, beta=1)

    return dX.reshape(*batch_shape, Wq.shape[1])


class _LoRAQKVBackwardLite(torch.autograd.Function):
    """Lightweight QKV backward: only dX, no weight grads, no X needed."""

    @staticmethod
    def forward(grad_Q, grad_K, grad_V, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        return _lora_qkv_backward_lite(
            grad_Q,
            grad_K,
            grad_V,
            Wq,
            Aq,
            Bq,
            Sq,
            Wk,
            Ak,
            Bk,
            Sk,
            Wv,
            Av,
            Bv,
            Sv,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_QKV")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_Q,
        grad_K,
        grad_V,
        Wq,
        Aq,
        Bq,
        Sq,
        Wk,
        Ak,
        Bk,
        Sk,
        Wv,
        Av,
        Bv,
        Sv,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAQKVBackwardLite", batched_indices={0, 1, 2}
        )
        grad_Q_bdim = in_dims[0]

        grad_Q_merged = grad_Q.reshape(-1, *grad_Q.shape[2:])
        grad_K_merged = grad_K.reshape(-1, *grad_K.shape[2:])
        grad_V_merged = grad_V.reshape(-1, *grad_V.shape[2:])

        dX = _lora_qkv_backward_lite(
            grad_Q_merged,
            grad_K_merged,
            grad_V_merged,
            Wq,
            Aq,
            Bq,
            Sq,
            Wk,
            Ak,
            Bk,
            Sk,
            Wv,
            Av,
            Bv,
            Sv,
        )
        dX = dX.reshape(*grad_Q.shape[:-1], Wq.shape[1])
        return dX, grad_Q_bdim


class Opaque_LoRA_QKV(torch.autograd.Function):
    """Fused LoRA for Q, K, V projections with vmap support.

    Computes:
        Q = X @ Wq.T + X @ Aq @ Bq * scaling_q
        K = X @ Wk.T + X @ Ak @ Bk * scaling_k
        V = X @ Wv.T + X @ Av @ Bv * scaling_v
    """

    @staticmethod
    def forward(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        """Forward pass for Q, K, V projections."""
        X_flat = X.reshape(-1, X.shape[-1])

        Q = F.linear(X, Wq)
        if Aq is not None and Bq is not None:
            Q.reshape(-1, Q.shape[-1]).addmm_(X_flat @ Aq, Bq, alpha=Sq, beta=1)

        K = F.linear(X, Wk)
        if Ak is not None and Bk is not None:
            K.reshape(-1, K.shape[-1]).addmm_(X_flat @ Ak, Bk, alpha=Sk, beta=1)

        V = F.linear(X, Wv)
        if Av is not None and Bv is not None:
            V.reshape(-1, V.shape[-1]).addmm_(X_flat @ Av, Bv, alpha=Sv, beta=1)

        return Q, K, V

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv = inputs
        # Under vmap(grad()), grad() detaches captured LoRA weights (requires_grad=False).
        # Skip saving X when weight grads aren't needed — reduces peak memory.
        needs_weight_grads = any(
            t is not None and t.requires_grad for t in [Aq, Bq, Ak, Bk, Av, Bv]
        )
        if needs_weight_grads:
            ctx.save_for_backward(X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv)
        else:
            ctx.save_for_backward(Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv)
        ctx.needs_weight_grads = needs_weight_grads
        ctx.Sq = Sq
        ctx.Sk = Sk
        ctx.Sv = Sv

    @staticmethod
    def backward(ctx, grad_Q, grad_K, grad_V):
        Sq, Sk, Sv = ctx.Sq, ctx.Sk, ctx.Sv

        if ctx.needs_weight_grads:
            X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = ctx.saved_tensors
            dX, dAq, dBq, dAk, dBk, dAv, dBv = _LoRAQKVBackward.apply(
                grad_Q,
                grad_K,
                grad_V,
                X,
                Wq,
                Aq,
                Bq,
                Sq,
                Wk,
                Ak,
                Bk,
                Sk,
                Wv,
                Av,
                Bv,
                Sv,
            )
        else:
            Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = ctx.saved_tensors
            dX = _LoRAQKVBackwardLite.apply(
                grad_Q,
                grad_K,
                grad_V,
                Wq,
                Aq,
                Bq,
                Sq,
                Wk,
                Ak,
                Bk,
                Sk,
                Wv,
                Av,
                Bv,
                Sv,
            )
            dAq = dBq = dAk = dBk = dAv = dBv = None

        return (
            dX,
            None,
            dAq,
            dBq,
            None,  # Q: Wq, Aq, Bq, Sq
            None,
            dAk,
            dBk,
            None,  # K: Wk, Ak, Bk, Sk
            None,
            dAv,
            dBv,
            None,  # V: Wv, Av, Bv, Sv
        )

    @staticmethod
    def vmap(info, in_dims, X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        """Efficient vmap rule: merge vmap batch into regular batch."""
        _validate_vmap_dims(in_dims, name="Opaque_LoRA_QKV", batched_indices={0})
        X_bdim = in_dims[0]

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        # Apply LoRA with addmm_ (same as non-vmap forward)
        X_flat = X_merged.reshape(-1, X_merged.shape[-1])

        Q = F.linear(X_merged, Wq)
        if Aq is not None and Bq is not None:
            Q.reshape(-1, Q.shape[-1]).addmm_(X_flat @ Aq, Bq, alpha=Sq, beta=1)

        K = F.linear(X_merged, Wk)
        if Ak is not None and Bk is not None:
            K.reshape(-1, K.shape[-1]).addmm_(X_flat @ Ak, Bk, alpha=Sk, beta=1)

        V = F.linear(X_merged, Wv)
        if Av is not None and Bv is not None:
            V.reshape(-1, V.shape[-1]).addmm_(X_flat @ Av, Bv, alpha=Sv, beta=1)

        # Reshape back
        Q = Q.reshape(*original_shape[:-1], -1)
        K = K.reshape(*original_shape[:-1], -1)
        V = V.reshape(*original_shape[:-1], -1)

        return (Q, K, V), (X_bdim, X_bdim, X_bdim)


def _lora_mlp_backward_impl(
    grad_out,
    X,
    Wg,
    Ag,
    Bg,
    Sg,
    Wu,
    Au,
    Bu,
    Su,
    Wd,
    Ad,
    Bd,
    Sd,
    gate,
    up,
    activation_type,
):
    """Shared LoRA_MLP backward logic used by both forward and vmap paths.

    Returns (dX, dAg, dBg, dAu, dBu, dAd, dBd).
    """
    batch_shape = X.shape[:-1]
    hidden_dim = X.shape[-1]

    X_flat = X.reshape(-1, hidden_dim)
    grad_out_flat = grad_out.reshape(-1, grad_out.shape[-1])
    gate_flat = gate.reshape(-1, gate.shape[-1])
    up_flat = up.reshape(-1, up.shape[-1])

    # Backward through down projection
    dh = grad_out_flat @ Wd
    if Ad is not None and Bd is not None:
        dh.addmm_(grad_out_flat @ Bd.t(), Ad.t(), alpha=Sd, beta=1)

    # Fused backward: recompute h, overwrite gate→dgate, up→dup (Unsloth pattern)
    act_backward_fused = _ACTIVATION_BACKWARD_FUSED[activation_type]
    h, dgate, dup = act_backward_fused(dh, gate_flat, up_flat)

    # LoRA weight gradients FIRST (before out=X_flat reuse)
    dAg = dBg = dAu = dBu = dAd = dBd = None
    dgate_Bgt = dup_But = None

    # Down LoRA grads (need recomputed h)
    if Ad is not None and Bd is not None:
        grad_out_Bdt = grad_out_flat @ Bd.t()
        dAd = torch.empty_like(Ad)
        dAd.addmm_(h.t(), grad_out_Bdt, alpha=Sd, beta=0)
        dBd = torch.empty_like(Bd)
        dBd.addmm_(Ad.t() @ h.t(), grad_out_flat, alpha=Sd, beta=0)

    # Gate LoRA grads (need X_flat and dgate)
    if Ag is not None and Bg is not None:
        dgate_Bgt = dgate @ Bg.t()
        dAg = torch.empty_like(Ag)
        dAg.addmm_(X_flat.t(), dgate_Bgt, alpha=Sg, beta=0)
        dBg = torch.empty_like(Bg)
        dBg.addmm_(Ag.t() @ X_flat.t(), dgate, alpha=Sg, beta=0)

    # Up LoRA grads (need X_flat and dup)
    if Au is not None and Bu is not None:
        dup_But = dup @ Bu.t()
        dAu = torch.empty_like(Au)
        dAu.addmm_(X_flat.t(), dup_But, alpha=Su, beta=0)
        dBu = torch.empty_like(Bu)
        dBu.addmm_(Au.t() @ X_flat.t(), dup, alpha=Su, beta=0)

    # dX from base + LoRA weights (all LoRA grads above already consumed X_flat)
    dX_flat = torch.mm(dgate, Wg)
    dX_flat.addmm_(dup, Wu, beta=1, alpha=1)
    if dgate_Bgt is not None:
        dX_flat.addmm_(dgate_Bgt, Ag.t(), alpha=Sg, beta=1)
    if dup_But is not None:
        dX_flat.addmm_(dup_But, Au.t(), alpha=Su, beta=1)

    dX = dX_flat.reshape(*batch_shape, hidden_dim)
    return dX, dAg, dBg, dAu, dBu, dAd, dBd


class _LoRAMLPBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support.

    Uses fused Triton activation backward (recomputes h, in-place gate→dgate, up→dup)
    and addmm_ for weight grads.
    """

    @staticmethod
    def forward(
        grad_out,
        X,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        gate,
        up,
        activation_type,
    ):
        return _lora_mlp_backward_impl(
            grad_out,
            X,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            Wd,
            Ad,
            Bd,
            Sd,
            gate,
            up,
            activation_type,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_MLP")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        X,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        gate,
        up,
        activation_type,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAMLPBackward", batched_indices={0, 1, 14, 15}
        )

        B_vmap = X.shape[0]
        hidden_dim = X.shape[-1]
        inter_dim = Wg.shape[0]  # intermediate dimension

        # Merge for dX and activation backward (spatially parallel)
        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        gate_merged = gate.reshape(-1, *gate.shape[2:])
        up_merged = up.reshape(-1, *up.shape[2:])

        grad_out_flat = grad_out_merged.reshape(-1, grad_out.shape[-1])
        gate_flat = gate_merged.reshape(-1, inter_dim)
        up_flat = up_merged.reshape(-1, inter_dim)

        # Backward through down projection (dX path — merged is fine)
        dh = grad_out_flat @ Wd
        if Ad is not None and Bd is not None:
            dh.addmm_(grad_out_flat @ Bd.t(), Ad.t(), alpha=Sd, beta=1)

        # Fused activation backward (merged)
        act_backward_fused = _ACTIVATION_BACKWARD_FUSED[activation_type]
        h, dgate, dup = act_backward_fused(dh, gate_flat, up_flat)

        # Per-sample LoRA weight grads using bmm (NOT merged!).  X_3d is a
        # reshape view of the input X; these grads must complete before dX
        # computation so that a fresh dX buffer (not X's storage) is used.
        X_3d = X.reshape(B_vmap, -1, hidden_dim)
        grad_out_3d = grad_out.reshape(B_vmap, -1, grad_out.shape[-1])
        h_3d = h.reshape(B_vmap, -1, inter_dim)
        dgate_3d = dgate.reshape(B_vmap, -1, inter_dim)
        dup_3d = dup.reshape(B_vmap, -1, inter_dim)

        def _per_sample_lora_grads(input_3d, grad_3d, A, B, S):
            if A is None or B is None:
                return None, None
            grad_Bt = grad_3d @ B.t()
            dA = torch.bmm(input_3d.transpose(-2, -1), grad_Bt) * S
            XA = input_3d @ A
            dB = torch.bmm(XA.transpose(-2, -1), grad_3d) * S
            return dA, dB

        # Down LoRA grads use h as input, grad_out as gradient
        dAd, dBd = _per_sample_lora_grads(h_3d, grad_out_3d, Ad, Bd, Sd)
        # Gate LoRA grads use X as input, dgate as gradient
        dAg, dBg = _per_sample_lora_grads(X_3d, dgate_3d, Ag, Bg, Sg)
        # Up LoRA grads use X as input, dup as gradient
        dAu, dBu = _per_sample_lora_grads(X_3d, dup_3d, Au, Bu, Su)

        # dX from base weights (merged — spatially parallel).  Allocates a fresh
        # dX buffer instead of reusing X's storage, so that parallel-residual
        # architectures (e.g., Cohere) where the same X feeds both attention and
        # MLP branches are not affected by one branch's backward clobbering X.
        dX_flat = torch.mm(dgate, Wg)
        dX_flat.addmm_(dup, Wu, beta=1, alpha=1)
        if Ag is not None and Bg is not None:
            dX_flat.addmm_(dgate @ Bg.t(), Ag.t(), alpha=Sg, beta=1)
        if Au is not None and Bu is not None:
            dX_flat.addmm_(dup @ Bu.t(), Au.t(), alpha=Su, beta=1)
        dX = dX_flat.reshape(X.shape)

        def _bdim(t):
            return 0 if t is not None else None

        return (
            (dX, dAg, dBg, dAu, dBu, dAd, dBd),
            (0, _bdim(dAg), _bdim(dBg), _bdim(dAu), _bdim(dBu), _bdim(dAd), _bdim(dBd)),
        )


def _lora_mlp_backward_lite(
    grad_out, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, gate, up, activation_type
):
    """Lightweight MLP backward: only computes dX (no weight grads, no X needed).

    Still needs gate/up for activation backward (recompute h, compute dgate/dup).
    """
    batch_shape = grad_out.shape[:-1]
    grad_out_flat = grad_out.reshape(-1, grad_out.shape[-1])
    gate_flat = gate.reshape(-1, gate.shape[-1])
    up_flat = up.reshape(-1, up.shape[-1])

    # Backward through down projection
    dh = grad_out_flat @ Wd
    if Ad is not None and Bd is not None:
        dh.addmm_(grad_out_flat @ Bd.t(), Ad.t(), alpha=Sd, beta=1)

    # Fused backward: recompute h, overwrite gate→dgate, up→dup
    act_backward_fused = _ACTIVATION_BACKWARD_FUSED[activation_type]
    _h, dgate, dup = act_backward_fused(dh, gate_flat, up_flat)

    # dX: fresh allocation (no X buffer to reuse)
    dX = torch.mm(dgate, Wg)
    dX.addmm_(dup, Wu, beta=1, alpha=1)
    if Ag is not None and Bg is not None:
        dX.addmm_(dgate @ Bg.t(), Ag.t(), alpha=Sg, beta=1)
    if Au is not None and Bu is not None:
        dX.addmm_(dup @ Bu.t(), Au.t(), alpha=Su, beta=1)

    return dX.reshape(*batch_shape, Wg.shape[1])


class _LoRAMLPBackwardLite(torch.autograd.Function):
    """Lightweight MLP backward: only dX, no weight grads, no X needed."""

    @staticmethod
    def forward(
        grad_out,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        gate,
        up,
        activation_type,
    ):
        return _lora_mlp_backward_lite(
            grad_out,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            Wd,
            Ad,
            Bd,
            Sd,
            gate,
            up,
            activation_type,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_MLP")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        gate,
        up,
        activation_type,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAMLPBackwardLite", batched_indices={0, 13, 14}
        )
        grad_out_bdim = in_dims[0]

        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        gate_merged = gate.reshape(-1, *gate.shape[2:])
        up_merged = up.reshape(-1, *up.shape[2:])

        dX = _lora_mlp_backward_lite(
            grad_out_merged,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            Wd,
            Ad,
            Bd,
            Sd,
            gate_merged,
            up_merged,
            activation_type,
        )
        dX = dX.reshape(*grad_out.shape[:-1], Wg.shape[1])
        return dX, grad_out_bdim


class Opaque_LoRA_MLP(torch.autograd.Function):
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
        X_flat = X.reshape(-1, X.shape[-1])

        gate = F.linear(X, Wg)
        if Ag is not None and Bg is not None:
            gate.reshape(-1, gate.shape[-1]).addmm_(X_flat @ Ag, Bg, alpha=Sg, beta=1)

        up = F.linear(X, Wu)
        if Au is not None and Bu is not None:
            up.reshape(-1, up.shape[-1]).addmm_(X_flat @ Au, Bu, alpha=Su, beta=1)

        # Use Triton kernel for activation (Unsloth callback pattern)
        act_forward = _ACTIVATION_FORWARD[activation_type]
        h = act_forward(gate, up)

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            h_flat = h.reshape(-1, h.shape[-1])
            out.reshape(-1, out.shape[-1]).addmm_(h_flat @ Ad, Bd, alpha=Sd, beta=1)

        return out, gate, up, h

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type = inputs
        _out, gate, up, _h = output
        # Don't save h — recomputed in backward via fused kernel (Unsloth pattern)
        # Under vmap(grad()), grad() detaches captured LoRA weights (requires_grad=False).
        # Skip saving X when weight grads aren't needed — reduces peak memory.
        needs_weight_grads = any(
            t is not None and t.requires_grad for t in [Ag, Bg, Au, Bu, Ad, Bd]
        )
        if needs_weight_grads:
            ctx.save_for_backward(X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up)
        else:
            ctx.save_for_backward(Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up)
        ctx.needs_weight_grads = needs_weight_grads
        ctx.Sg = Sg
        ctx.Su = Su
        ctx.Sd = Sd
        ctx.activation_type = activation_type

    @staticmethod
    def backward(ctx, grad_out, grad_gate, grad_up, grad_h):
        Sg, Su, Sd = ctx.Sg, ctx.Su, ctx.Sd

        if ctx.needs_weight_grads:
            X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up = ctx.saved_tensors
            dX, dAg, dBg, dAu, dBu, dAd, dBd = _LoRAMLPBackward.apply(
                grad_out,
                X,
                Wg,
                Ag,
                Bg,
                Sg,
                Wu,
                Au,
                Bu,
                Su,
                Wd,
                Ad,
                Bd,
                Sd,
                gate,
                up,
                ctx.activation_type,
            )
        else:
            Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, gate, up = ctx.saved_tensors
            dX = _LoRAMLPBackwardLite.apply(
                grad_out,
                Wg,
                Ag,
                Bg,
                Sg,
                Wu,
                Au,
                Bu,
                Su,
                Wd,
                Ad,
                Bd,
                Sd,
                gate,
                up,
                ctx.activation_type,
            )
            dAg = dBg = dAu = dBu = dAd = dBd = None

        return (
            dX,
            None,
            dAg,
            dBg,
            None,  # gate
            None,
            dAu,
            dBu,
            None,  # up
            None,
            dAd,
            dBd,
            None,  # down
            None,  # activation_type
        )

    @staticmethod
    def vmap(
        info,
        in_dims,
        X,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        activation_type,
    ):
        """Efficient vmap rule: merge vmap batch into regular batch.

        With the two-level pattern, backward is handled by _LoRAMLPBackward.apply()
        dispatched from Opaque_LoRA_MLP.backward(). No autograd graph needed here,
        so we can use addmm_ and direct Triton activation kernels.
        """
        _validate_vmap_dims(in_dims, name="Opaque_LoRA_MLP", batched_indices={0})
        X_bdim = in_dims[0]

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])
        X_flat = X_merged.reshape(-1, X_merged.shape[-1])

        # Apply MLP with addmm_ + direct Triton activation (no autograd needed)
        gate = F.linear(X_merged, Wg)
        if Ag is not None and Bg is not None:
            gate.reshape(-1, gate.shape[-1]).addmm_(X_flat @ Ag, Bg, alpha=Sg, beta=1)

        up = F.linear(X_merged, Wu)
        if Au is not None and Bu is not None:
            up.reshape(-1, up.shape[-1]).addmm_(X_flat @ Au, Bu, alpha=Su, beta=1)

        # Direct Triton activation kernel (safe: backward handled by _LoRAMLPBackward)
        act_forward = _ACTIVATION_FORWARD[activation_type]
        h = act_forward(gate, up)

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            h_flat = h.reshape(-1, h.shape[-1])
            out.reshape(-1, out.shape[-1]).addmm_(h_flat @ Ad, Bd, alpha=Sd, beta=1)

        # Reshape back
        out = out.reshape(*original_shape[:-1], -1)
        gate = gate.reshape(*original_shape[:-1], -1)
        up = up.reshape(*original_shape[:-1], -1)
        h = h.reshape(*original_shape[:-1], -1)

        return (out, gate, up, h), (X_bdim, X_bdim, X_bdim, X_bdim)


# ============================================================================
# Convenience wrappers
# ============================================================================


def opaque_lora_w(X, W, A, B, scaling):
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
    ensure_cuda_tensors(X, W, A, B, fn_name="opaque_lora_w")
    X, W, A, B = follow_autocast(X, W, A, B)
    return Opaque_LoRA_W.apply(X, W, A, B, scaling)


def opaque_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
    """Apply LoRA to Q, K, V projections with vmap support.

    Returns:
        Tuple of (Q, K, V)
    """
    ensure_cuda_tensors(
        X,
        Wq,
        Aq,
        Bq,
        Wk,
        Ak,
        Bk,
        Wv,
        Av,
        Bv,
        fn_name="opaque_lora_qkv",
    )
    X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv = follow_autocast(
        X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv
    )
    return Opaque_LoRA_QKV.apply(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv)


def opaque_lora_mlp(
    X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation="swiglu"
):
    """Apply LoRA MLP with configurable GLU activation and vmap support.

    Args:
        X: MLP input activations.
        Wg: Base gate-projection weight.
        Ag: Gate-projection LoRA down-projection weight.
        Bg: Gate-projection LoRA up-projection weight.
        Sg: Gate-projection LoRA scaling factor.
        Wu: Base up-projection weight.
        Au: Up-projection LoRA down-projection weight.
        Bu: Up-projection LoRA up-projection weight.
        Su: Up-projection LoRA scaling factor.
        Wd: Base down-projection weight.
        Ad: Down-projection LoRA down-projection weight.
        Bd: Down-projection LoRA up-projection weight.
        Sd: Down-projection LoRA scaling factor.
        activation: Activation type - "swiglu" (default), "geglu_exact", or "geglu_approx".

    Returns:
        Output tensor (batch, seq_len, hidden_dim)
    """
    ensure_cuda_tensors(
        X,
        Wg,
        Ag,
        Bg,
        Wu,
        Au,
        Bu,
        Wd,
        Ad,
        Bd,
        fn_name="opaque_lora_mlp",
    )
    X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd = follow_autocast(
        X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd
    )
    if isinstance(activation, str):
        activation_type = _ACTIVATION_NAMES[activation]
    else:
        activation_type = activation
    result = Opaque_LoRA_MLP.apply(
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type
    )
    return result[0]  # Return only the output, not intermediate tensors
